from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .backtest import TRADING_DAYS
from .data import DataError, load_close

DEFAULT_TREND_ASSETS = {
    "SPY": "US large-cap equities",
    "TLT": "Long-duration US Treasuries",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "UUP": "US dollar",
}


@dataclass
class DiversifiedTrendResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    summary_stats: Dict[str, Dict[str, float]]


def load_trend_prices(tickers: Sequence[str], refresh: bool = False) -> pd.DataFrame:
    """Load dividend-adjusted ETF proxy closes on their shared trading calendar."""
    if not tickers:
        raise ValueError("Select at least one trend asset.")

    closes = []
    for ticker in tickers:
        try:
            closes.append(load_close(ticker, refresh=refresh).rename(ticker))
        except Exception as exc:
            raise DataError(f"Could not load trend asset {ticker}: {exc}") from exc

    prices = pd.concat(closes, axis=1, join="inner").sort_index().dropna()
    if prices.empty:
        raise DataError("Selected assets have no overlapping price history.")
    return prices


def portfolio_metrics(total_returns: pd.Series, excess_returns: pd.Series) -> Dict[str, float]:
    if total_returns.empty:
        return {}

    volatility = float(excess_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    annual_excess = float(excess_returns.mean() * TRADING_DAYS)
    sharpe = annual_excess / volatility if volatility > 1e-12 else 0.0
    growth = (1.0 + total_returns).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    years = len(total_returns) / TRADING_DAYS
    ending_value = float(growth.iloc[-1])
    cagr = ending_value ** (1.0 / years) - 1.0 if ending_value > 0 and years > 0 else np.nan

    return {
        "CAGR": float(cagr),
        "Excess return (ann.)": annual_excess,
        "Volatility (ann.)": volatility,
        "Sharpe": float(sharpe),
        "Max drawdown": float(drawdown.min()),
    }


def run_diversified_trend_backtest(
    prices: pd.DataFrame,
    risk_free: pd.Series,
    lookback_days: int = 252,
    volatility_lookback: int = 63,
    target_volatility: float = 0.10,
    cost_bps: float = 2.0,
    max_asset_weight: float = 1.0,
    max_gross_leverage: float = 2.5,
) -> DiversifiedTrendResult:
    """Backtest monthly time-series momentum on ETF proxies, net of turnover costs.

    Signals and volatility estimates are formed at month-end and held from the
    next trading session. Portfolio returns are cash plus weighted asset excess
    returns, approximating fully collateralized futures exposure.
    """
    if prices.empty or prices.columns.has_duplicates:
        raise ValueError("Prices must be non-empty and have unique asset columns.")
    if lookback_days < 2 or volatility_lookback < 2:
        raise ValueError("Lookbacks must be at least 2 trading days.")
    if target_volatility <= 0 or cost_bps < 0 or max_asset_weight <= 0 or max_gross_leverage <= 0:
        raise ValueError("Volatility, weight, and leverage limits must be positive; costs cannot be negative.")

    prices = prices.sort_index().dropna(how="any").astype(float)
    if (prices <= 0).any().any():
        raise ValueError("Prices must all be positive.")
    returns = prices.pct_change(fill_method=None)
    rf = risk_free.reindex(prices.index).ffill().shift(1).fillna(0.0).astype(float)
    excess = returns.sub(rf, axis=0)

    momentum = prices.pct_change(lookback_days, fill_method=None)
    realized_volatility = (
        returns.rolling(volatility_lookback, min_periods=volatility_lookback)
        .std(ddof=1)
        .mul(np.sqrt(TRADING_DAYS))
    )
    valid = momentum.notna() & realized_volatility.notna() & realized_volatility.gt(0)
    active_assets = valid.sum(axis=1).replace(0, np.nan)
    asset_risk_budget = target_volatility / np.sqrt(active_assets)
    raw_weights = np.sign(momentum).mul(asset_risk_budget, axis=0).div(realized_volatility)
    raw_weights = raw_weights.where(valid, 0.0).clip(-max_asset_weight, max_asset_weight)

    gross = raw_weights.abs().sum(axis=1)
    gross_scale = (max_gross_leverage / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    raw_weights = raw_weights.mul(gross_scale, axis=0)

    months = pd.Series(prices.index.to_period("M"), index=prices.index)
    month_end = months.ne(months.shift(-1))
    month_end.iloc[-1] = False
    targets = raw_weights.where(month_end, axis=0).ffill()
    weights = targets.shift(1).fillna(0.0)

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    trading_cost = turnover * cost_bps / 10_000.0
    strategy_gross_excess = (weights * excess).sum(axis=1)
    strategy_excess = strategy_gross_excess - trading_cost
    strategy_returns = rf + strategy_excess
    equal_weight_excess = excess.mean(axis=1)
    equal_weight_returns = rf + equal_weight_excess

    daily = pd.DataFrame(
        {
            "strategy_ret": strategy_returns,
            "strategy_excess": strategy_excess,
            "strategy_gross_ret": rf + strategy_gross_excess,
            "strategy_gross_excess": strategy_gross_excess,
            "equal_weight_ret": equal_weight_returns,
            "equal_weight_excess": equal_weight_excess,
            "rf": rf,
            "turnover": turnover,
            "cost": trading_cost,
        },
        index=prices.index,
    )

    active = weights.abs().sum(axis=1).gt(0)
    if not active.any():
        raise ValueError("Not enough price history to generate a trend position.")
    first_active = active[active].index[0]
    daily = daily.loc[first_active:].copy()
    weights = weights.loc[first_active:].copy()

    stats = {
        "Diversified trend": portfolio_metrics(daily["strategy_ret"], daily["strategy_excess"])
        | {"Turnover (ann.)": float(daily["turnover"].mean() * TRADING_DAYS)},
        "Diversified trend (before costs)": portfolio_metrics(daily["strategy_gross_ret"], daily["strategy_gross_excess"]),
        "Equal-weight buy & hold": portfolio_metrics(daily["equal_weight_ret"], daily["equal_weight_excess"]),
    }
    return DiversifiedTrendResult(daily=daily, weights=weights, summary_stats=stats)


def trend_subperiods(result: DiversifiedTrendResult) -> pd.DataFrame:
    """Return comparable strategy and benchmark Sharpe ratios by fixed periods."""
    periods = [
        ("Full sample", "1900-01-01", "2260-01-01"),
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020 onward", "2020-01-01", "2260-01-01"),
        ("2019 onward", "2019-01-01", "2260-01-01"),
    ]
    rows = []
    for label, start, end in periods:
        daily = result.daily.loc[start:end]
        if len(daily) < 60:
            continue
        trend = portfolio_metrics(daily["strategy_ret"], daily["strategy_excess"])
        benchmark = portfolio_metrics(daily["equal_weight_ret"], daily["equal_weight_excess"])
        rows.append(
            {
                "Period": label,
                "Start": daily.index[0].date(),
                "End": daily.index[-1].date(),
                "Trend Sharpe": trend["Sharpe"],
                "Equal-weight Sharpe": benchmark["Sharpe"],
                "Trend excess (ann.)": trend["Excess return (ann.)"],
                "Trend max drawdown": trend["Max drawdown"],
                "Days": len(daily),
            }
        )
    return pd.DataFrame(rows)