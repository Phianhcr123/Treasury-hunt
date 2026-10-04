import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.data import load_risk_free
from eom_treasury_rally.diversified_trend import (
    DEFAULT_TREND_ASSETS,
    load_trend_prices,
    run_diversified_trend_backtest,
    trend_subperiods,
)

st.set_page_config(page_title="Diversified Trend Research", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_inputs(tickers: tuple[str, ...], refresh_token: int):
    return load_trend_prices(tickers, refresh=refresh_token > 0), load_risk_free(refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def get_sensitivity(prices, risk_free, volatility_lookback, target_volatility, max_asset_weight, max_gross_leverage):
    rows = []
    for lookback_days in (126, 252, 378):
        for sensitivity_cost in (0.0, 2.0, 5.0):
            sensitivity = run_diversified_trend_backtest(
                prices,
                risk_free,
                lookback_days=lookback_days,
                volatility_lookback=volatility_lookback,
                target_volatility=target_volatility,
                cost_bps=sensitivity_cost,
                max_asset_weight=max_asset_weight,
                max_gross_leverage=max_gross_leverage,
            )
            recent = sensitivity.daily.loc["2019-01-01":]
            rows.append(
                {
                    "Lookback (months)": lookback_days // 21,
                    "Cost (bps)": sensitivity_cost,
                    "Full-sample Sharpe": sensitivity.summary_stats["Diversified trend"]["Sharpe"],
                    "Post-2019 Sharpe": (
                        recent["strategy_excess"].mean() * np.sqrt(252) / recent["strategy_excess"].std(ddof=1)
                        if len(recent) > 1 and recent["strategy_excess"].std(ddof=1) > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


if "trend_refresh" not in st.session_state:
    st.session_state.trend_refresh = 0

st.title("Diversified Trend Research")
st.caption("Monthly time-series momentum across liquid, exchange-traded asset proxies")

with st.sidebar:
    st.header("Portfolio")
    tickers = st.multiselect(
        "Asset proxies",
        options=list(DEFAULT_TREND_ASSETS),
        default=list(DEFAULT_TREND_ASSETS),
        format_func=lambda ticker: f"{ticker} · {DEFAULT_TREND_ASSETS[ticker]}",
    )
    lookback_months = st.slider("Momentum lookback (months)", 3, 18, 12)
    volatility_lookback = st.slider("Volatility lookback (trading days)", 20, 126, 63, 5)
    target_volatility = st.slider("Portfolio volatility target (%)", 5, 20, 10) / 100.0
    cost_bps = st.slider("Cost per one-way notional turnover (bps)", 0.0, 10.0, 2.0, 0.5)
    max_asset_weight = st.slider("Maximum absolute weight per asset", 0.25, 2.0, 1.0, 0.25)
    max_gross_leverage = st.slider("Maximum gross exposure", 0.5, 4.0, 2.5, 0.25)
    if st.button("Refresh market data", width="stretch"):
        st.session_state.trend_refresh += 1
        st.cache_data.clear()

if not tickers:
    st.info("Select at least one asset proxy to run the research backtest.")
    st.stop()

try:
    with st.spinner("Loading adjusted prices and T-bill rates…"):
        prices, risk_free = load_inputs(tuple(tickers), st.session_state.trend_refresh)
    result = run_diversified_trend_backtest(
        prices,
        risk_free,
        lookback_days=lookback_months * 21,
        volatility_lookback=volatility_lookback,
        target_volatility=target_volatility,
        cost_bps=cost_bps,
        max_asset_weight=max_asset_weight,
        max_gross_leverage=max_gross_leverage,
    )
except Exception as exc:
    st.error(f"Research run failed: {exc}")
    st.stop()

strategy = result.summary_stats["Diversified trend"]
benchmark = result.summary_stats["Equal-weight buy & hold"]
daily = result.daily

st.caption(
    f"Shared sample: {daily.index[0].date()} to {daily.index[-1].date()} · "
    f"{len(tickers)} assets · signals rebalance monthly from prior-close data"
)
metrics = st.columns(5)
metrics[0].metric("Net Sharpe", f"{strategy['Sharpe']:.2f}", f"benchmark {benchmark['Sharpe']:.2f}")
metrics[1].metric("Annual excess", f"{strategy['Excess return (ann.)']:.2%}")
metrics[2].metric("Annual volatility", f"{strategy['Volatility (ann.)']:.2%}")
metrics[3].metric("Max drawdown", f"{strategy['Max drawdown']:.1%}")
metrics[4].metric("Turnover cost", f"{daily['cost'].sum():.2%}", "sum of daily cost drag")

tab_performance, tab_exposure, tab_robustness, tab_method = st.tabs(
    ["Performance", "Exposure", "Subperiods", "Method & evidence"]
)

with tab_performance:
    curve = go.Figure()
    curve.add_trace(
        go.Scatter(
            x=daily.index,
            y=(1 + daily["strategy_excess"]).cumprod(),
            name="Diversified trend, net",
            line=dict(width=2.5, color="#176b87"),
        )
    )
    curve.add_trace(
        go.Scatter(
            x=daily.index,
            y=(1 + daily["equal_weight_excess"]).cumprod(),
            name="Equal-weight buy & hold",
            line=dict(width=1.5, color="#d17a22"),
        )
    )
    curve.update_layout(
        title="Growth of $1 in excess of T-bills",
        yaxis_type="log",
        height=430,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(curve, width="stretch")

    summary = pd.DataFrame(result.summary_stats).T
    st.dataframe(
        summary.style.format(
            {
                "CAGR": "{:.2%}",
                "Excess return (ann.)": "{:.2%}",
                "Volatility (ann.)": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.1%}",
            }
        ),
        width="stretch",
    )

    yearly = daily.groupby(daily.index.year).agg(
        Trend_excess=("strategy_excess", lambda values: (1 + values).prod() - 1),
        Benchmark_excess=("equal_weight_excess", lambda values: (1 + values).prod() - 1),
    )
    yearly.columns = ["Trend excess", "Benchmark excess"]
    st.dataframe(yearly.style.format("{:.2%}"), width="stretch")

with tab_exposure:
    exposure = go.Figure()
    for ticker in result.weights.columns:
        exposure.add_trace(
            go.Scatter(x=result.weights.index, y=result.weights[ticker], name=ticker, stackgroup=None)
        )
    exposure.update_layout(
        title="Signed asset weights (positive = long, negative = short)",
        yaxis_title="Weight of portfolio NAV",
        height=430,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(exposure, width="stretch")
    st.metric("Average gross exposure", f"{result.weights.abs().sum(axis=1).mean():.2f}x")

with tab_robustness:
    subperiods = trend_subperiods(result)
    st.dataframe(
        subperiods.style.format(
            {
                "Trend Sharpe": "{:.2f}",
                "Equal-weight Sharpe": "{:.2f}",
                "Trend excess (ann.)": "{:.2%}",
                "Trend max drawdown": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Sharpe ratios are recalculated on each period's daily excess returns; short slices are omitted.")
    sensitivity = get_sensitivity(
        prices,
        risk_free,
        volatility_lookback,
        target_volatility,
        max_asset_weight,
        max_gross_leverage,
    )
    st.subheader("Signal-horizon and cost sensitivity")
    st.dataframe(
        sensitivity.style.format(
            {
                "Cost (bps)": "{:.0f}",
                "Full-sample Sharpe": "{:.2f}",
                "Post-2019 Sharpe": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Fixed 6-, 12-, and 18-month horizons; this is a sensitivity check, not an optimized selection.")

with tab_method:
    st.markdown(
        """
### Backtest definition
- Signal is the sign of each asset's trailing total return. Signals and trailing volatility are measured at each month-end close and positions apply from the next session.
- Each available asset receives an equal share of the target risk budget, scaled by trailing 63-day volatility. Per-asset and total gross exposure are capped.
- Strategy return is T-bill return plus signed weighted asset excess returns. Costs are charged against each change in notional weight.
- The comparator is an equal-weight, long-only portfolio of the selected assets over exactly the same days, with no turnover costs.

### Important limits
These are adjusted ETF prices, not futures returns. They provide an accessible first test of cross-asset trend, but omit futures-specific roll yield, financing, margin, and market-impact details. The selected ETFs also do not reproduce the 58-market academic portfolio. Yahoo prices, proxy choice, fees, and parameter choices all affect results; treat the Sharpe as a hypothesis check, not an expected live return.

### Research references
- [Moskowitz, Ooi & Pedersen, Time Series Momentum (2012)](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- [Hurst, Ooi & Pedersen, A Century of Evidence on Trend-Following Investing (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)
- [Daniel & Moskowitz, Momentum Crashes (NBER)](https://www.nber.org/papers/w20439)
"""
    )