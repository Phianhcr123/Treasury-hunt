"""Month-end call overlay backtest on TLT with synthetic option prices.

Free historical TLT option quotes don't exist, so options are priced the way the hypothesis
assumes the market prices them: an American CRR tree with a standard flat volatility.
The volatility is the MOVE index converted to TLT price volatility
(MOVE in bp/yr x TLT duration), which tracks Cboe's TLT implied-vol index (VXTLT) closely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Window, month_offsets
from .data import load_dataset, load_option_inputs
from .pricing import black_scholes_delta, crr_price, drift_tree


@dataclass(frozen=True)
class OptionSpec:
    dte: int = 30
    """Calendar days to expiry at entry."""
    moneyness: float = 1.0
    """Strike / spot at entry (1.0 = at the money, 1.02 = 2% out of the money)."""
    half_spread: float = 0.01
    """Dollars per share paid on each side relative to model value (TLT ATM spreads are ~$0.01-0.03)."""
    commission: float = 0.0065
    """Dollars per share per side ($0.65 per contract)."""
    duration: float = 16.0
    """TLT modified duration used to convert MOVE (bp) into price volatility."""
    strike_step: float = 0.5


def iv_proxy(move: pd.Series, duration: float) -> pd.Series:
    return move * duration / 10_000


def _divs_between(divs: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[tuple[float, float], ...]:
    sel = divs[(divs.index > start) & (divs.index <= end)]
    return tuple(((d - start).days / 365.0, float(a)) for d, a in sel.items())


def run_options_backtest(
    window: Window = Window(3, 0),
    spec: OptionSpec = OptionSpec(),
    excess_drift_bps: float = 11.0,
    etf_cost_bps: float = 2.0,
    ticker: str = "TLT",
    start: str | None = None,
    end: str | None = None,
    with_model: bool = True,
) -> pd.DataFrame:
    """One row per month: ETF window return vs buying a call at entry and selling at exit."""
    opt = load_option_inputs(ticker).loc[start:end]
    adj = load_dataset(ticker).loc[start:end]
    rf_annual = (adj["rf"] * 252).reindex(opt.index).ffill().bfill()
    divs = opt.loc[opt["dividend"] > 0, "dividend"]
    iv = iv_proxy(opt["move"], spec.duration)

    off = month_offsets(opt.index)
    period = opt.index.to_period("M")
    entries = opt.index[off["to_end"] == -window.entry]
    exits_same = {p: d for p, d in zip(period[off["to_end"] == min(window.exit, 0)], opt.index[off["to_end"] == min(window.exit, 0)])}
    exits_next = {p - 1: d for p, d in zip(period[off["from_end"] == window.exit], opt.index[off["from_end"] == window.exit])} if window.exit > 0 else {}

    cost = spec.half_spread + spec.commission
    rows = []
    for t0 in entries:
        p = t0.to_period("M")
        t1 = exits_next.get(p) if window.exit > 0 else exits_same.get(p)
        if t1 is None or t1 <= t0:
            continue
        s0, s1 = float(opt.at[t0, "close"]), float(opt.at[t1, "close"])
        v0, v1 = float(iv.at[t0]), float(iv.at[t1])
        r0, r1 = float(rf_annual.at[t0]), float(rf_annual.at[t1])
        expiry = t0 + pd.Timedelta(days=spec.dte)
        T0 = spec.dte / 365.0
        T1 = (expiry - t1).days / 365.0
        K = round(s0 * spec.moneyness / spec.strike_step) * spec.strike_step
        d0, d1 = _divs_between(divs, t0, expiry), _divs_between(divs, t1, expiry)

        c0 = crr_price(s0, K, T0, r0, v0, dividends=d0)
        c1 = crr_price(s1, K, T1, r1, v1, dividends=d1)
        delta = black_scholes_delta(s0, K, T0, r0, v0, dividends=d0)
        buy, sell = c0 + cost, max(c1 - cost, 0.0)

        held = adj.loc[(adj.index > t0) & (adj.index <= t1)]
        rf_win = float((1 + held["rf"]).prod() - 1)
        etf_ret = float((1 + held["ret"]).prod() - 1) - 2 * etf_cost_bps / 10_000

        # Same dollar delta as $1 in the ETF; the rest of the dollar sits in T-bills.
        alloc = buy / (delta * s0) if delta > 0 else np.nan
        option_ret = sell / buy - 1
        alloc_gross = c0 / (delta * s0) if delta > 0 else np.nan
        option_ret_gross = c1 / c0 - 1
        row = {
            "entry": t0.date(),
            "exit": t1.date(),
            "spot_entry": s0,
            "strike": K,
            "iv_entry": v0,
            "call_entry": c0,
            "call_exit": c1,
            "delta": delta,
            "etf_ret": etf_ret,
            "rf_window": rf_win,
            "call_ret": option_ret,
            "call_overlay_ret": alloc * option_ret + (1 - alloc) * rf_win,
            "etf_ret_gross": etf_ret + 2 * etf_cost_bps / 10_000,
            "call_ret_gross": option_ret_gross,
            "call_overlay_ret_gross": alloc_gross * option_ret_gross + (1 - alloc_gross) * rf_win,
            "delta_hedged_pnl": ((c1 - c0) - delta * (s1 - s0)) / s0,
            "cost_share_of_premium": 2 * cost / c0,
        }
        if with_model:
            m = drift_tree(s0, K, T0, r0, v0, horizon=window.holding_days / 252, excess_drift=excess_drift_bps / 1e4 * 252, dividends=d0)
            row["model_expected_ret"] = m.expected_return
            row["model_expected_ret_net"] = (m.expected_exit_value - cost) / (m.price + cost) - 1
        rows.append(row)
    return pd.DataFrame(rows)


def options_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-trade statistics; Sharpe is annualized assuming one trade per month."""
    def stats(r: pd.Series) -> dict[str, float]:
        sd = r.std(ddof=1)
        return {
            "Avg per trade": r.mean(),
            "Std per trade": sd,
            "Sharpe (ann.)": r.mean() / sd * np.sqrt(12) if sd > 0 else np.nan,
            "Hit rate": (r > 0).mean(),
            "Worst trade": r.min(),
            "t-stat": r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else np.nan,
        }

    rf = trades["rf_window"]
    out = {
        "ETF in window (excess)": stats(trades["etf_ret"] - rf),
        "Call, % of premium (excess)": stats(trades["call_ret"] - rf),
        "Delta-matched call overlay (excess)": stats(trades["call_overlay_ret"] - rf),
        "Delta-hedged call (% of spot)": stats(trades["delta_hedged_pnl"]),
    }
    df = pd.DataFrame(out).T
    df["Trades"] = len(trades)
    return df


def iv_proxy_check(ticker: str = "TLT", duration: float = 16.0, end: str | None = None) -> dict[str, float]:
    """How the MOVE-based implied-vol proxy compares with subsequent realized TLT volatility."""
    opt = load_option_inputs(ticker).loc[:end]
    adj = load_dataset(ticker).loc[:end]
    iv = iv_proxy(opt["move"], duration).reindex(adj.index)
    fwd_rv = adj["ret"][::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252)
    both = pd.concat([iv, fwd_rv], axis=1, keys=["iv", "rv"]).dropna()
    return {
        "avg_iv_proxy": float(both["iv"].mean()),
        "avg_next_21d_realized": float(both["rv"].mean()),
        "correlation": float(both["iv"].corr(both["rv"])),
    }
