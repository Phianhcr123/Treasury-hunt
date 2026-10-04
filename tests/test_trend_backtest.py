import numpy as np
import pandas as pd
import pytest

from eom_treasury_rally.trend_backtest import run_trend_backtest, trend_metrics


def make_pair(periods=1_200, seed=3):
    index = pd.bdate_range("2012-01-02", periods=periods)
    rng = np.random.default_rng(seed)
    close = 100 * (1 + pd.Series(rng.normal(0.0003, 0.011, periods), index=index)).cumprod()
    df = pd.DataFrame({"asset_close": close, "rf": 0.00005})
    df["asset_ret"] = df["asset_close"].pct_change()
    df["safe_ret"] = df["rf"]
    return df.dropna()


def test_gross_minus_cost_is_net_and_turnover_is_weight_change():
    res = run_trend_backtest(make_pair(), lookback=50, cost_bps=5)
    d = res.daily

    assert np.allclose(d["strat_gross_ret"] - d["cost"], d["strat_ret"])
    assert np.allclose(d["strat_gross_excess"] - d["strat_excess"], d["cost"])
    assert np.allclose(d["cost"], d["turnover"] * 5 / 1e4)
    assert np.allclose(d["turnover"].iloc[1:], d["target_weight"].diff().abs().iloc[1:])
    assert res.summary_stats["Adaptive Trend Strategy"]["Turnover (ann.)"] == pytest.approx(d["turnover"].mean() * 252)


def test_one_run_cut_by_date_matches_a_run_on_the_shorter_sample():
    df = make_pair()
    cut = df.index[900]
    full = run_trend_backtest(df, lookback=50, cost_bps=3).daily.loc[:cut]
    short = run_trend_backtest(df.loc[:cut], lookback=50, cost_bps=3).daily

    pd.testing.assert_frame_equal(full, short)
    assert trend_metrics(full["strat_ret"], full["strat_excess"]) == pytest.approx(trend_metrics(short["strat_ret"], short["strat_excess"]))
