from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import TRADING_DAYS, Window
from .data import CACHE_DIR, DataError, load_close, load_risk_free

TREND_PAIRS = {
    "SPY / BIL": ("SPY", "BIL", "US Equities Trend (S&P 500) vs T-Bills"),
    "QQQ / BIL": ("QQQ", "BIL", "Tech / Growth Trend vs T-Bills"),
    "TLT / BIL": ("TLT", "BIL", "Long-Term Treasury Trend vs T-Bills"),
    "GLD / BIL": ("GLD", "BIL", "Gold Trend vs T-Bills"),
    "VNQ / BIL": ("VNQ", "BIL", "Real Estate Trend vs T-Bills"),
    "IWM / BIL": ("IWM", "BIL", "Russell 2000 Small-Cap Trend vs T-Bills"),
}


def load_pair_dataset(asset_ticker: str, safe_ticker: str = "BIL", refresh: bool = False) -> pd.DataFrame:
    """Download and cache closing prices for an asset, a cash-proxy (e.g. BIL), and risk-free rate."""
    asset_close = load_close(asset_ticker, refresh=refresh)
    try:
        safe_close = load_close(safe_ticker, refresh=refresh)
    except Exception:
        # Fallback to T-bill rate compounding if safe asset is missing
        safe_close = None

    rf = load_risk_free(refresh=refresh)
    df = pd.DataFrame({"asset_close": asset_close})
    if safe_close is not None:
        df["safe_close"] = safe_close.reindex(df.index).ffill()
    else:
        df["safe_close"] = np.nan

    df["rf"] = rf.reindex(df.index, method="ffill").shift(1).fillna(0.0)
    df["asset_ret"] = df["asset_close"].pct_change()
    
    if safe_close is not None and not df["safe_close"].dropna().empty:
        df["safe_ret"] = df["safe_close"].pct_change().fillna(df["rf"])
    else:
        df["safe_ret"] = df["rf"]

    df = df.dropna(subset=["asset_close", "asset_ret"])
    return df


@dataclass
class DualMomentumResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary_stats: Dict[str, Dict[str, float]]
    lookback_days: int
    signal_type: str


def run_trend_backtest(
    df: pd.DataFrame,
    lookback: int = 200,
    signal_type: str = "SMA",  # 'SMA', 'EMA', or 'ROC' (Rate of Change / Absolute Momentum)
    vol_target_ann: float = 0.15,  # 15% volatility targeting
    enable_vol_target: bool = True,
    cost_bps: float = 3.0,
) -> DualMomentumResult:
    """Run an adaptive trend-following / absolute momentum backtest with volatility targeting."""
    d = df.copy()
    close = d["asset_close"]

    # Calculate Signal
    if signal_type == "SMA":
        trend_line = close.rolling(lookback).mean()
        # Lag by 1 day so signal generated at yesterday's close is traded today
        signal = (close > trend_line).astype(float).shift(1).fillna(0.0)
    elif signal_type == "EMA":
        trend_line = close.ewm(span=lookback, adjust=False).mean()
        signal = (close > trend_line).astype(float).shift(1).fillna(0.0)
    elif signal_type == "ROC":
        roc = close.pct_change(lookback)
        signal = (roc > 0).astype(float).shift(1).fillna(0.0)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    # Volatility Targeting
    if enable_vol_target and vol_target_ann > 0:
        # Realized 20-day volatility annualized
        realized_vol = (d["asset_ret"].rolling(20).std() * np.sqrt(TRADING_DAYS)).shift(1)
        # Cap leverage at 1.5x, floor weight at 0.1
        vol_scalar = (vol_target_ann / realized_vol.replace(0, np.nan)).clip(lower=0.1, upper=1.5).fillna(1.0)
        target_weight = signal * vol_scalar
    else:
        target_weight = signal

    # Calculate turnover and transaction costs
    weight_change = target_weight.diff().abs().fillna(0.0)
    cost = weight_change * (cost_bps / 1e4)

    # Asset return + Safe asset / cash return on unallocated capital
    asset_pnl = target_weight * d["asset_ret"]
    safe_pnl = (1.0 - target_weight).clip(lower=0.0) * d["safe_ret"]
    gross_ret = asset_pnl + safe_pnl
    net_ret = gross_ret - cost

    d["signal"] = signal
    d["target_weight"] = target_weight
    d["strat_ret"] = net_ret
    d["strat_excess"] = net_ret - d["rf"]
    d["bh_ret"] = d["asset_ret"]
    d["bh_excess"] = d["asset_ret"] - d["rf"]

    # Drop warm-up period
    d = d.iloc[lookback + 25 :].copy()

    # Calculate summary metrics
    def calc_metrics(r_series: pd.Series, excess_series: pd.Series) -> Dict[str, float]:
        n = len(r_series)
        if n == 0:
            return {}
        cagr = (1 + r_series).prod() ** (TRADING_DAYS / n) - 1
        vol = excess_series.std() * np.sqrt(TRADING_DAYS)
        mean_excess = excess_series.mean() * TRADING_DAYS
        sharpe = mean_excess / vol if vol > 1e-6 else 0.0
        
        cum = (1 + r_series).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        return {
            "CAGR": cagr,
            "Excess return (ann.)": mean_excess,
            "Volatility (ann.)": vol,
            "Sharpe": sharpe,
            "Max drawdown": max_dd,
        }

    strat_stats = calc_metrics(d["strat_ret"], d["strat_excess"])
    bh_stats = calc_metrics(d["bh_ret"], d["bh_excess"])

    # Calculate trade state changes
    position_changes = (signal.diff() != 0) & (signal.notna())
    num_trades = int(signal.diff().abs().sum() / 2.0)
    strat_stats["Trades"] = num_trades
    strat_stats["Time in market"] = float((target_weight > 0.05).mean())
    bh_stats["Time in market"] = 1.0

    trades_list = []
    # Identify trade episodes
    entry_dates = d.index[(d["signal"] == 1.0) & (d["signal"].shift(1) == 0.0)]
    exit_dates = d.index[(d["signal"] == 0.0) & (d["signal"].shift(1) == 1.0)]

    for en in entry_dates:
        ex_candidates = exit_dates[exit_dates > en]
        ex = ex_candidates[0] if len(ex_candidates) > 0 else d.index[-1]
        t_slice = d.loc[en:ex]
        gross_trade = (1 + t_slice["asset_ret"]).prod() - 1
        trades_list.append({
            "entry_date": en,
            "exit_date": ex,
            "days_held": len(t_slice),
            "trade_return": gross_trade,
            "annualized": (1 + gross_trade) ** (TRADING_DAYS / max(len(t_slice), 1)) - 1,
        })
    trades_df = pd.DataFrame(trades_list)
    if not trades_df.empty:
        strat_stats["Hit rate"] = float((trades_df["trade_return"] > 0).mean())
        strat_stats["Avg trade return"] = float(trades_df["trade_return"].mean())
    else:
        strat_stats["Hit rate"] = 0.0
        strat_stats["Avg trade return"] = 0.0

    stats = {
        "Adaptive Trend Strategy": strat_stats,
        "Buy & Hold Underlying": bh_stats,
    }

    return DualMomentumResult(
        daily=d,
        trades=trades_df,
        summary_stats=stats,
        lookback_days=lookback,
        signal_type=signal_type,
    )


def trend_lookback_grid(
    df: pd.DataFrame,
    lookbacks: List[int] = [50, 100, 150, 200, 250],
    vol_targets: List[float] = [0.10, 0.12, 0.15, 0.18, 0.20],
    signal_type: str = "SMA",
    cost_bps: float = 3.0,
) -> pd.DataFrame:
    """Generate a parameter sensitivity grid of Sharpe ratios across Lookback vs Vol Target."""
    grid = np.zeros((len(lookbacks), len(vol_targets)))
    for i, lb in enumerate(lookbacks):
        for j, vt in enumerate(vol_targets):
            try:
                res = run_trend_backtest(
                    df,
                    lookback=lb,
                    signal_type=signal_type,
                    vol_target_ann=vt,
                    enable_vol_target=True,
                    cost_bps=cost_bps,
                )
                grid[i, j] = res.summary_stats["Adaptive Trend Strategy"]["Sharpe"]
            except Exception:
                grid[i, j] = np.nan
    return pd.DataFrame(grid, index=lookbacks, columns=[f"{int(v*100)}% Vol" for v in vol_targets])
