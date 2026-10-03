import numpy as np
import pandas as pd
import pytest

from eom_treasury_rally.backtest import (
    Window,
    day_of_month_profile,
    month_offsets,
    permutation_test,
    positions,
    run_backtest,
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
    # The final trade is still open on the last day of data, so it has no exit cost.
    assert diff == pytest.approx((len(res.trades) * 2 - 1) * 5 / 10_000)


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
