import numpy as np
import pandas as pd
import pytest

from eom_treasury_rally.backtest import Window
from eom_treasury_rally.data import CACHE_DIR
from eom_treasury_rally.options_backtest import OptionSpec, run_options_backtest
from eom_treasury_rally.options_cli import contract_grid

pytestmark = pytest.mark.skipif(
    not (CACHE_DIR / "tlt_option_inputs.csv").exists() or not (CACHE_DIR / "tlt.csv").exists(),
    reason="needs cached TLT prices and option inputs",
)


def test_gross_columns_drop_exactly_the_costs():
    t = run_options_backtest(Window(3, 0), OptionSpec(), etf_cost_bps=2.0, start="2015-01-01", end="2016-12-31", with_model=False)
    assert len(t) > 20

    assert np.allclose(t["etf_ret_gross"] - t["etf_ret"], 4 / 10_000)
    assert np.allclose(t["call_ret_gross"], t["call_exit"] / t["call_entry"] - 1)
    assert (t["call_overlay_ret_gross"] > t["call_overlay_ret"]).all()

    free = run_options_backtest(Window(3, 0), OptionSpec(half_spread=0.0, commission=0.0), etf_cost_bps=0.0, start="2015-01-01", end="2016-12-31", with_model=False)
    assert np.allclose(free["call_overlay_ret"], free["call_overlay_ret_gross"])
    assert np.allclose(free["etf_ret"], free["etf_ret_gross"])


def test_contract_grid_only_uses_trades_inside_the_dates(monkeypatch):
    seen = []

    def fake_summary(trades):
        seen.append(trades)
        return pd.DataFrame(np.nan, index=["Call, % of premium (excess)", "Delta-matched call overlay (excess)", "ETF in window (excess)"], columns=["Avg per trade", "Sharpe (ann.)"])

    monkeypatch.setattr("eom_treasury_rally.options_cli.options_summary", fake_summary)
    monkeypatch.setattr("eom_treasury_rally.options_cli.GRID_DTE", (30,))
    monkeypatch.setattr("eom_treasury_rally.options_cli.GRID_MONEYNESS", (1.0,))
    contract_grid(Window(3, 0), 0.01, start="2010-01-01", end="2012-06-30")

    (t,) = seen
    assert pd.to_datetime(t["entry"]).min() >= pd.Timestamp("2010-01-01")
    assert pd.to_datetime(t["exit"]).max() <= pd.Timestamp("2012-06-30")
    assert len(t) == 30
