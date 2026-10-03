import pandas as pd
import pytest

from eom_treasury_rally.backtest import Window
from eom_treasury_rally.pricing import black_scholes, crr_price, drift_tree, implied_vol
from eom_treasury_rally.scanner import next_window

S, K, T, R, SIG = 77.5, 77.5, 30 / 365, 0.04, 0.16
DIV = ((4 / 365, 0.31),)


def test_crr_european_converges_to_black_scholes():
    for call in (True, False):
        assert crr_price(S, K, T, R, SIG, call, american=False, steps=800) == pytest.approx(black_scholes(S, K, T, R, SIG, call), abs=2e-3)


def test_american_call_without_dividends_equals_european():
    assert crr_price(S, K, T, R, SIG, steps=400) == pytest.approx(crr_price(S, K, T, R, SIG, american=False, steps=400), abs=1e-9)


def test_american_put_worth_more_than_european():
    assert crr_price(S, 85, T, R, SIG, call=False) > crr_price(S, 85, T, R, SIG, call=False, american=False)


def test_dividend_creates_early_exercise_premium_for_deep_itm_call():
    am = crr_price(S, 70, T, R, SIG, dividends=DIV)
    eu = crr_price(S, 70, T, R, SIG, american=False, dividends=DIV)
    assert am - eu > 0.05


def test_implied_vol_round_trip():
    price = crr_price(S, 80, T, R, 0.21, dividends=DIV, steps=100)
    assert implied_vol(price, S, 80, T, R, dividends=DIV, steps=100) == pytest.approx(0.21, abs=1e-3)


def test_drift_tree_without_drift_reproduces_fair_price():
    res = drift_tree(S, K, T, R, SIG, horizon=3 / 252, excess_drift=0.0, dividends=DIV)
    assert res.rn_check == pytest.approx(res.price, rel=2e-3)
    assert abs(res.expected_return) < 5e-3


def test_drift_raises_call_value_by_roughly_elasticity_times_drift():
    res = drift_tree(S, K, T, R, SIG, horizon=3 / 252, excess_drift=0.0011 * 252, dividends=DIV)
    assert res.expected_return == pytest.approx(res.elasticity * 0.0033, rel=0.25)
    put = drift_tree(S, K, T, R, SIG, horizon=3 / 252, excess_drift=0.0011 * 252, call=False, dividends=DIV)
    assert put.expected_return < 0


def test_next_window_dates():
    w = next_window(pd.Timestamp("2026-10-03"), Window(3, 0))
    assert (w.entry, w.exit) == (pd.Timestamp("2026-10-27"), pd.Timestamp("2026-10-30"))
    w = next_window(pd.Timestamp("2026-10-28"), Window(3, 0))
    assert (w.entry, w.exit) == (pd.Timestamp("2026-11-24"), pd.Timestamp("2026-11-30"))
    w = next_window(pd.Timestamp("2026-10-03"), Window(2, 1))
    assert (w.entry, w.exit) == (pd.Timestamp("2026-10-28"), pd.Timestamp("2026-11-02"))
