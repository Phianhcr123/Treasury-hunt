import numpy as np
import pandas as pd
import pytest

from eom_treasury_rally.diversified_trend import (
    run_diversified_trend_backtest,
    trend_subperiods,
)


def make_prices(periods=900, seed=7):
    index = pd.bdate_range("2010-01-04", periods=periods)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.009, size=(periods, 3))
    returns[:, 0] += 0.0003
    returns[:, 1] -= 0.0002
    prices = 100 * (1 + pd.DataFrame(returns, index=index, columns=["UP", "DOWN", "MIXED"])).cumprod()
    return prices


def run(prices, cost_bps=0, **kwargs):
    risk_free = pd.Series(0.00005, index=prices.index)
    return run_diversified_trend_backtest(
        prices,
        risk_free,
        lookback_days=20,
        volatility_lookback=20,
        target_volatility=0.10,
        cost_bps=cost_bps,
        **kwargs,
    )


def test_month_end_signal_is_held_starting_next_session():
    prices = make_prices()
    prices["UP"] = 100 * (1.001 ** np.arange(len(prices)))
    result = run(prices)

    first_active = result.weights.index[0]
    prior_session = prices.index[prices.index.get_loc(first_active) - 1]
    assert prior_session.to_period("M") != first_active.to_period("M")
    assert result.weights.loc[first_active, "UP"] > 0

    rebalances = result.weights.diff().abs().sum(axis=1).gt(1e-12)
    for rebalance_date in result.weights.index[rebalances]:
        prior_date = prices.index[prices.index.get_loc(rebalance_date) - 1]
        assert prior_date.to_period("M") != rebalance_date.to_period("M")


def test_transaction_costs_reduce_returns_on_rebalance_days():
    prices = make_prices()
    gross = run(prices, cost_bps=0)
    net = run(prices, cost_bps=5)

    traded = net.daily["turnover"] > 0
    assert traded.any()
    assert (net.daily.loc[traded, "strategy_ret"] < gross.daily.loc[traded, "strategy_ret"]).all()
    assert net.daily["cost"].sum() > 0


def test_position_and_portfolio_gross_exposure_are_capped():
    result = run(make_prices(), max_asset_weight=0.25, max_gross_leverage=0.40)

    assert result.weights.abs().max().max() <= 0.25 + 1e-12
    assert result.weights.abs().sum(axis=1).max() <= 0.40 + 1e-12


def test_summary_compares_same_dates_and_includes_post_2019():
    result = run(make_prices(periods=2_500))
    subperiods = trend_subperiods(result)

    assert result.summary_stats["Diversified trend"]["Sharpe"] == pytest.approx(
        result.daily["strategy_excess"].mean() * np.sqrt(252)
        / result.daily["strategy_excess"].std(ddof=1)
    )
    assert "2019 onward" in subperiods["Period"].tolist()
    assert subperiods.loc[subperiods["Period"] == "2019 onward", "Days"].iloc[0] > 60