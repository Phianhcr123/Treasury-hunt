import numpy as np
import pandas as pd
import pytest

from eom_treasury_rally.backtest import (
    Window,
    day_of_month_profile,
    holdout_start,
    month_offsets,
    permutation_test,
    positions,
    run_backtest,
    slice_result,
    summary,
)


def make_df(start="2020-01-01", end="2021-12-31", ret=0.0, rf=0.0):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"close": 100.0, "ret": ret, "rf": rf}, index=idx)


def test_month_offsets_last_and_first_days():
    df = make_df("2024-01-01", "2024-03-29")
    off = month_offsets(df.index)
    assert off.loc["2024-01-31", "to_end"] == 0
    assert off.loc["2024-01-30", "to_end"] == -1
    assert off.loc["2024-02-01", "from_end"] == 1
    assert off.loc["2024-02-02", "from_end"] == 2


def test_incomplete_final_month_is_never_treated_as_month_end():
    df = make_df("2024-01-01", "2024-02-07")
    pos = positions(df.index, Window(3, 0))
    assert pos.loc["2024-02-01":].sum() == 0
    assert pos.loc["2024-01-29":"2024-01-31"].tolist() == [1, 1, 1]


def test_window_in_progress_at_data_start_is_skipped():
    df = make_df("2024-01-30", "2024-02-29")
    pos = positions(df.index, Window(3, 0))
    assert pos.loc["2024-01"].sum() == 0
    res = run_backtest(df, Window(3, 0))
    assert res.trades["entry_close"].notna().all()


def test_positions_hold_last_n_days():
    df = make_df("2024-01-01", "2024-02-29")
    pos = positions(df.index, Window(3, 0))
    jan = pos.loc["2024-01"]
    assert jan.sum() == 3
    assert jan.loc["2024-01-29":"2024-01-31"].tolist() == [1, 1, 1]
    assert pos.loc["2024-01-26"] == 0


def test_positions_with_exit_after_month_end():
    df = make_df("2024-01-01", "2024-02-29")
    pos = positions(df.index, Window(2, 1))
    assert pos.loc["2024-01-30":"2024-02-01"].tolist() == [1, 1, 1]
    assert pos.loc["2024-02-02"] == 0


def test_positions_with_exit_before_month_end():
    df = make_df("2024-01-01", "2024-02-29")
    pos = positions(df.index, Window(4, -1))
    assert pos.loc["2024-01-25":"2024-01-31"].tolist() == [0, 1, 1, 1, 0]


@pytest.mark.parametrize("entry,exit_", [(0, 0), (2, -2), (3, -5)])
def test_invalid_windows(entry, exit_):
    with pytest.raises(ValueError):
        Window(entry, exit_)


def test_strategy_earns_rf_out_of_market_and_pays_costs():
    df = make_df(ret=0.001, rf=0.0001)
    res = run_backtest(df, Window(3, 0), cost_bps=0)
    d = res.daily
    assert np.allclose(d.loc[d.pos == 1, "strategy"], 0.001)
    assert np.allclose(d.loc[d.pos == 0, "strategy"], 0.0001)

    res_cost = run_backtest(df, Window(3, 0), cost_bps=5)
    diff = (res.daily["strategy"] - res_cost.daily["strategy"]).sum()
    assert diff == pytest.approx(len(res.trades) * 2 * 5 / 10_000)


def test_costs_land_on_entry_and_exit_days_and_match_trades():
    df = make_df("2024-01-01", "2024-03-29", ret=0.001)
    res = run_backtest(df, Window(3, 0), cost_bps=5)
    d = res.daily
    assert d.loc["2024-01-29":"2024-01-31", "turnover"].tolist() == [1, 0, 1]
    assert d.loc["2024-02-01", "turnover"] == 0
    for _, t in res.trades.iterrows():
        held = d.loc[d.index > t["entry_close"]].loc[: t["exit_close"]]
        net = (1 + held["strategy_excess"]).prod() - 1
        # Daily compounding of the cost differs from the trade table's additive cost by a hair.
        assert net == pytest.approx(t["net_excess"], abs=1e-5)


def test_gross_series_is_strategy_before_costs():
    df = make_df(ret=0.001, rf=0.0001)
    d = run_backtest(df, Window(3, 0), cost_bps=5).daily
    assert np.allclose(d["strategy_gross"] - d["strategy"], d["turnover"] * 5 / 10_000)
    assert np.allclose(d["strategy_gross_excess"], d["strategy_gross"] - d["rf"])
    s = summary(run_backtest(df, Window(3, 0), cost_bps=5))
    assert s["Month-end strategy (before costs)"]["Excess return (ann.)"] > s["Month-end strategy"]["Excess return (ann.)"]
    # 12 round trips a year, each side counted once.
    assert s["Month-end strategy"]["Turnover (ann.)"] == pytest.approx(24, rel=0.05)


def test_holdout_is_two_years_for_long_histories():
    idx = pd.bdate_range("2002-07-31", "2026-10-02")
    start = holdout_start(idx)
    assert start == pd.Timestamp("2024-10-01")
    assert (idx[-1] - start).days >= 2 * 365


def test_holdout_is_twenty_percent_for_short_histories():
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    start = holdout_start(idx)
    # 20% of 5 years is about a year (cut lands on 2023-12-31), moved back to the start of its month.
    assert start == pd.Timestamp("2023-12-01")
    assert pd.Timedelta(days=365) < idx[-1] - start <= pd.Timedelta(days=365 + 31)


def test_holdout_boundary_never_splits_a_month():
    idx = pd.bdate_range("2010-01-01", "2026-06-17")
    start = holdout_start(idx)
    before = idx[idx < start][-1]
    assert before.to_period("M") != start.to_period("M")
    assert idx[-1] - start >= pd.Timedelta(days=int(365.25 * 2))


def test_slice_result_splits_daily_and_trades_cleanly():
    df = make_df("2018-01-01", "2024-12-31", ret=0.001, rf=0.0001)
    res = run_backtest(df, Window(3, 0), cost_bps=2)
    cut = holdout_start(df.index)
    ins = slice_result(res, end=df.index[df.index < cut][-1])
    oos = slice_result(res, start=cut)
    assert len(ins.daily) + len(oos.daily) == len(res.daily)
    assert len(ins.trades) + len(oos.trades) == len(res.trades)
    assert (oos.trades["entry_close"] >= ins.daily.index[-1]).all()
    assert ins.daily["strategy"].sum() + oos.daily["strategy"].sum() == pytest.approx(res.daily["strategy"].sum())


def test_detects_planted_month_end_effect():
    rng = np.random.default_rng(0)
    df = make_df("2005-01-01", "2024-12-31")
    df["ret"] = rng.normal(0, 0.005, len(df))
    off = month_offsets(df.index)
    df.loc[off["to_end"] > -3, "ret"] += 0.002
    window = Window(3, 0)
    s = summary(run_backtest(df, window, cost_bps=1))
    assert s["Month-end strategy"]["Sharpe"] > 1
    assert s["Rest of month only"]["Sharpe"] < s["Month-end strategy"]["Sharpe"]
    assert permutation_test(df, window, n_sims=300)["p_value"] < 0.01
    prof = day_of_month_profile(df)
    assert prof.loc[0, "mean"] > prof.loc[-5, "mean"]


def test_no_effect_gives_unremarkable_p_value():
    rng = np.random.default_rng(1)
    df = make_df("2005-01-01", "2024-12-31")
    df["ret"] = rng.normal(0, 0.005, len(df))
    assert permutation_test(df, Window(3, 0), n_sims=300)["p_value"] > 0.05
