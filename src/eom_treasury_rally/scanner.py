"""Live TLT call scanner for the next month-end window.

For every listed call it backs out the market's implied volatility (American CRR, projected
monthly dividends), then runs the drift tree over the next month-end window, assuming spot and
implied vol are unchanged by the entry date. The quoted spread is charged on entry and exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from .backtest import Window
from .pricing import crr_price, drift_tree, implied_vol


class NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_DAY = CustomBusinessDay(calendar=NYSEHolidays())


def trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=NYSE_DAY)


@dataclass
class NextWindow:
    entry: pd.Timestamp
    exit: pd.Timestamp
    holding_days: int


def next_window(today: pd.Timestamp, window: Window = Window(3, 0)) -> NextWindow:
    """Entry/exit closes of the next month-end window that hasn't started yet."""
    today = pd.Timestamp(today).normalize()
    for months_ahead in range(0, 3):
        month_start = (today + pd.offsets.MonthBegin(0) if today.day == 1 else today - pd.offsets.MonthBegin(1)) + pd.DateOffset(months=months_ahead)
        days = trading_days(month_start, month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10))
        in_month = days[days.month == month_start.month]
        entry = in_month[-1 - window.entry]
        if window.exit <= 0:
            exit_ = in_month[-1 + window.exit]
        else:
            exit_ = days[days > in_month[-1]][window.exit - 1]
        if entry >= today:
            return NextWindow(entry, exit_, window.holding_days)
    raise RuntimeError("Could not find the next window.")


def projected_dividends(ticker: yf.Ticker, today: pd.Timestamp, until: pd.Timestamp) -> list[tuple[pd.Timestamp, float]]:
    """TLT pays monthly, ex-date on the first trading day of the month; project the last amount forward."""
    hist = ticker.dividends
    amount = float(hist.iloc[-1]) if len(hist) else 0.0
    out = []
    m = today + pd.offsets.MonthBegin(1)
    while m <= until:
        out.append((trading_days(m, m + pd.Timedelta(days=7))[0], amount))
        m += pd.offsets.MonthBegin(1)
    return out


def scan_calls(
    symbol: str = "TLT",
    window: Window = Window(3, 0),
    excess_drift_bps: float = 11.0,
    risk_free: float = 0.04,
    max_days: int = 75,
    strike_band: float = 0.05,
    commission: float = 0.0065,
    fallback_spread: float = 0.02,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    tk = yf.Ticker(symbol)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price data for {symbol}.")
    spot = float(hist["Close"].iloc[-1])
    today = pd.Timestamp(today or pd.Timestamp.now().normalize())
    win = next_window(today, window)
    horizon = win.holding_days / 252

    expiries = [pd.Timestamp(e) for e in tk.options]
    expiries = [e for e in expiries if win.exit < e <= today + pd.Timedelta(days=max_days)]
    divs = projected_dividends(tk, today, today + pd.Timedelta(days=max_days + 1))

    rows = []
    for exp_date in expiries:
        calls = tk.option_chain(exp_date.strftime("%Y-%m-%d")).calls
        calls = calls[(calls["strike"] >= spot * (1 - strike_band)) & (calls["strike"] <= spot * (1 + strike_band))]
        for _, c in calls.iterrows():
            bid, ask, last = float(c["bid"] or 0), float(c["ask"] or 0), float(c["lastPrice"] or 0)
            live = bid > 0 and ask > bid
            mid = (bid + ask) / 2 if live else last
            spread = ask - bid if live else fallback_spread
            if mid <= 0.02:
                continue
            K = float(c["strike"])
            T_now = (exp_date - today).days / 365
            d_now = tuple(((d - today).days / 365, a) for d, a in divs if today < d <= exp_date)
            iv = implied_vol(mid, spot, K, T_now, risk_free, dividends=d_now, steps=80)
            if not np.isfinite(iv):
                continue
            T_entry = (exp_date - win.entry).days / 365
            d_entry = tuple(((d - win.entry).days / 365, a) for d, a in divs if win.entry < d <= exp_date)
            m = drift_tree(spot, K, T_entry, risk_free, iv, horizon, excess_drift_bps / 1e4 * 252, dividends=d_entry, value_steps=80)
            cost = spread / 2 + commission
            net = (m.expected_exit_value - cost) / (m.price + cost) - 1
            rows.append(
                {
                    "contract": c["contractSymbol"],
                    "expiry": exp_date.date(),
                    "strike": K,
                    "bid": bid,
                    "ask": ask,
                    "quote": "live" if live else "last trade (market closed)",
                    "market_iv": iv,
                    "fair_at_entry": m.price,
                    "expected_value_at_exit": m.expected_exit_value,
                    "delta": m.delta,
                    "elasticity": m.elasticity,
                    "expected_ret_net": net,
                    "ret_sd": m.return_sd,
                    "edge_per_risk": net / m.return_sd if m.return_sd > 0 else np.nan,
                    "cost_bps_of_spot_per_delta": 2 * cost / (m.delta * spot) * 1e4 if m.delta > 0 else np.nan,
                    "volume": int(c["volume"]) if pd.notna(c["volume"]) else 0,
                    "open_interest": int(c["openInterest"]) if pd.notna(c["openInterest"]) else 0,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge_per_risk", ascending=False).reset_index(drop=True)
    etf_drift = excess_drift_bps * win.holding_days
    etf_sd = float(np.nanmedian(df["market_iv"])) * np.sqrt(horizon) if not df.empty else np.nan
    meta = {
        "spot": spot,
        "today": today.date(),
        "window_entry": win.entry.date(),
        "window_exit": win.exit.date(),
        "expected_window_drift_bps": etf_drift,
        "etf_edge_per_risk": etf_drift / 1e4 / etf_sd if etf_sd else np.nan,
        "next_ex_dividend": divs[0][0].date() if divs else None,
    }
    return df, meta
