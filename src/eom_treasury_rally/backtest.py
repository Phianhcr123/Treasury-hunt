from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Window:
    """Holding window anchored on the last trading day of each month.

    entry: enter at the close `entry` trading days before the month's last close.
    exit:  exit at the close `exit` trading days relative to the month's last close
           (0 = last trading day, -1 = the day before, +1 = first day of next month).
    """

    entry: int = 3
    exit: int = 0

    def __post_init__(self) -> None:
        if self.entry < 1:
            raise ValueError("entry must be >= 1")
        if self.exit <= -self.entry:
            raise ValueError("exit must come after entry")

    @property
    def holding_days(self) -> int:
        return self.entry + self.exit


def month_offsets(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Position of each trading day relative to month boundaries.

    to_end:   0 on the month's last trading day, -1 the day before, ...
    from_end: 1 on the first trading day of a month, 2 on the second, ...
              (i.e. trading days elapsed since the previous month's last close)
    """
    period = index.to_period("M")
    grp = pd.Series(np.arange(len(index)), index=index).groupby(period)
    to_end = grp.cumcount(ascending=False).to_numpy() * -1
    from_end = grp.cumcount().to_numpy() + 1
    off = pd.DataFrame({"to_end": to_end, "from_end": from_end}, index=index)
    # If the data stops before the month is over, the true month-end is unknown.
    last = index[-1]
    if last < last + pd.offsets.BMonthEnd(0):
        off.loc[period == last.to_period("M"), "to_end"] = -10_000
    return off


def positions(index: pd.DatetimeIndex, window: Window) -> pd.Series:
    """1 on days whose close-to-close return is held, else 0.

    Entering at the close of day -entry means the first held return is day -(entry-1).
    The schedule is known in advance from the calendar, so there is no look-ahead.
    """
    off = month_offsets(index)
    held = (off["to_end"] > -window.entry) & (off["to_end"] <= min(window.exit, 0))
    if window.exit > 0:
        held |= off["from_end"] <= window.exit
    # A window already in progress on the first day of data has no entry price.
    if held.iloc[0]:
        held.iloc[: int(np.argmin(held.to_numpy()))] = False
    return held.astype(int).rename("pos")


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    window: Window
    cost_bps: float


def run_backtest(df: pd.DataFrame, window: Window, cost_bps: float = 2.0) -> BacktestResult:
    """Long the ETF inside the window, earn the T-bill rate otherwise.

    cost_bps is charged per side on each entry and exit (spread + commission + slippage).
    """
    pos = positions(df.index, window)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost = turnover * cost_bps / 10_000

    out = pd.DataFrame(index=df.index)
    out["ret"] = df["ret"]
    out["rf"] = df["rf"]
    out["pos"] = pos
    out["strategy"] = pos * df["ret"] + (1 - pos) * df["rf"] - cost
    out["strategy_excess"] = out["strategy"] - out["rf"]
    out["buy_hold"] = df["ret"]
    out["buy_hold_excess"] = df["ret"] - df["rf"]
    out["rest_of_month_excess"] = (1 - pos) * (df["ret"] - df["rf"])

    return BacktestResult(daily=out, trades=_trades(out, cost_bps), window=window, cost_bps=cost_bps)


def _trades(daily: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    held = daily[daily["pos"] == 1]
    if held.empty:
        return pd.DataFrame(columns=["entry_close", "exit_close", "days", "gross", "excess", "net_excess"])
    run_id = (daily["pos"].diff().fillna(1) != 0).cumsum()[daily["pos"] == 1]
    rows = []
    for _, g in held.groupby(run_id):
        gross = (1 + g["ret"]).prod() - 1
        rf = (1 + g["rf"]).prod() - 1
        entry_idx = daily.index.get_loc(g.index[0]) - 1
        rows.append(
            {
                "entry_close": daily.index[entry_idx] if entry_idx >= 0 else pd.NaT,
                "exit_close": g.index[-1],
                "days": len(g),
                "gross": gross,
                "excess": gross - rf,
                "net_excess": gross - rf - 2 * cost_bps / 10_000,
            }
        )
    return pd.DataFrame(rows)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def summary(res: BacktestResult) -> dict[str, dict[str, float]]:
    d = res.daily
    years = len(d) / TRADING_DAYS

    def block(total: pd.Series, excess: pd.Series) -> dict[str, float]:
        vol = excess.std() * np.sqrt(TRADING_DAYS)
        return {
            "CAGR": (1 + total).prod() ** (1 / years) - 1,
            "Excess return (ann.)": excess.mean() * TRADING_DAYS,
            "Volatility (ann.)": vol,
            "Sharpe": excess.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
            "Max drawdown": _max_drawdown(total),
        }

    strat = block(d["strategy"], d["strategy_excess"])
    strat["Time in market"] = d["pos"].mean()
    t = res.trades
    if len(t):
        strat["Trades"] = float(len(t))
        strat["Avg trade (net excess)"] = t["net_excess"].mean()
        strat["Hit rate"] = (t["net_excess"] > 0).mean()
        strat["t-stat (per trade)"] = t["net_excess"].mean() / (t["net_excess"].std(ddof=1) / np.sqrt(len(t)))

    return {
        "Month-end strategy": strat,
        "Buy & hold": block(d["buy_hold"], d["buy_hold_excess"]),
        "Rest of month only": block(d["rest_of_month_excess"] + d["rf"], d["rest_of_month_excess"]),
    }


def day_of_month_profile(df: pd.DataFrame, before: int = 10, after: int = 5) -> pd.DataFrame:
    """Mean daily excess return by trading day relative to month-end, with 95% CI.

    Offset 0 is the last trading day of the month; +1 is the first day of the next month.
    """
    off = month_offsets(df.index)
    excess = df["ret"] - df["rf"]
    label = pd.Series(np.nan, index=df.index)
    near_end = off["to_end"] > -before
    label[near_end] = off["to_end"][near_end]
    near_start = (off["from_end"] <= after) & ~near_end
    label[near_start] = off["from_end"][near_start]
    g = excess.groupby(label.dropna().astype(int))
    stats = g.agg(["mean", "std", "count"])
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    stats["t_stat"] = stats["mean"] / (stats["std"] / np.sqrt(stats["count"]))
    stats.index = stats.index.astype(int)
    stats.index.name = "offset"
    return stats[["mean", "ci95", "t_stat", "count"]]


def permutation_test(df: pd.DataFrame, window: Window, n_sims: int = 2000, seed: int = 7) -> dict[str, float]:
    """Is the month-end window special, or would any same-length window each month do as well?

    Each simulation picks a random contiguous window of the same length inside every month
    and records the average excess return per trade. The p-value is the share of
    simulations that match or beat the real month-end window.
    """
    excess = (df["ret"] - df["rf"]).to_numpy()
    log_ex = np.log1p(excess)
    months = df.index.to_period("M")
    codes, uniques = pd.factorize(months)
    n = window.holding_days

    starts, lengths = [], []
    for m in range(len(uniques)):
        idx = np.flatnonzero(codes == m)
        starts.append(idx[0])
        lengths.append(len(idx))
    starts = np.array(starts)
    lengths = np.array(lengths)
    valid = lengths >= n + 1
    starts, lengths = starts[valid], lengths[valid]

    csum = np.concatenate([[0.0], np.cumsum(log_ex)])

    def window_return(first: np.ndarray) -> np.ndarray:
        return np.expm1(csum[first + n] - csum[first])

    pos = positions(df.index, window).to_numpy()
    real = []
    run_start = None
    for i, p in enumerate(pos):
        if p and run_start is None:
            run_start = i
        if not p and run_start is not None:
            real.append(np.expm1(csum[i] - csum[run_start]))
            run_start = None
    actual = float(np.mean(real)) if real else np.nan

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for k in range(n_sims):
        first = starts + (rng.random(len(starts)) * (lengths - n + 1)).astype(int)
        sims[k] = window_return(first).mean()

    return {
        "actual_avg_trade": actual,
        "random_avg_trade": float(sims.mean()),
        "p_value": float((np.sum(sims >= actual) + 1) / (n_sims + 1)),
        "n_sims": float(n_sims),
    }


def sensitivity_grid(
    df: pd.DataFrame,
    entries: range = range(1, 11),
    exits: range = range(-2, 4),
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Sharpe ratio for every (entry, exit) window, to check the result is not one lucky cell."""
    grid = pd.DataFrame(index=pd.Index(list(entries), name="entry"), columns=pd.Index(list(exits), name="exit"), dtype=float)
    for e in entries:
        for x in exits:
            if x <= -e:
                continue
            res = run_backtest(df, Window(e, x), cost_bps)
            grid.loc[e, x] = summary(res)["Month-end strategy"]["Sharpe"]
    return grid


DEFAULT_PERIODS = [
    ("Full sample", None, None),
    ("2002–2014", None, "2014-12-31"),
    ("2015–present", "2015-01-01", None),
    ("In paper's sample (≤2018)", None, "2018-12-31"),
    ("After paper's sample (2019+)", "2019-01-01", None),
    ("Last 3 years", "LAST3Y", None),
]


def subperiod_table(df: pd.DataFrame, window: Window, cost_bps: float = 2.0) -> pd.DataFrame:
    rows = []
    for label, start, end in DEFAULT_PERIODS:
        if start == "LAST3Y":
            start = (df.index[-1] - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        sub = df.loc[start:end]
        if len(sub) < TRADING_DAYS:
            continue
        res = run_backtest(sub, window, cost_bps)
        s = summary(res)
        strat, bh = s["Month-end strategy"], s["Buy & hold"]
        rows.append(
            {
                "Period": label,
                "Start": sub.index[0].date(),
                "End": sub.index[-1].date(),
                "Strategy Sharpe": strat["Sharpe"],
                "Buy & hold Sharpe": bh["Sharpe"],
                "Strategy excess (ann.)": strat["Excess return (ann.)"],
                "Avg trade (net)": strat.get("Avg trade (net excess)", np.nan),
                "Hit rate": strat.get("Hit rate", np.nan),
                "Trades": int(strat.get("Trades", 0)),
            }
        )
    return pd.DataFrame(rows)


def yearly_returns(res: BacktestResult) -> pd.DataFrame:
    d = res.daily
    yr = d.groupby(d.index.year)
    return pd.DataFrame(
        {
            "Strategy excess": yr["strategy_excess"].apply(lambda r: (1 + r).prod() - 1),
            "Buy & hold excess": yr["buy_hold_excess"].apply(lambda r: (1 + r).prod() - 1),
        }
    )
