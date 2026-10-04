import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.data import load_risk_free
from eom_treasury_rally.diversified_trend import (
    DEFAULT_TREND_ASSETS,
    load_trend_prices,
    portfolio_metrics,
    run_diversified_trend_backtest,
    trend_subperiods,
)
from eom_treasury_rally.holdout_ui import (
    SPX_LINE,
    SPX_NAME,
    Line,
    comparison_table,
    equity_chart,
    growth,
    locked_message,
    log_evaluation,
    setup_holdout,
    show_equity_chart,
    spx_growth,
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


def comparison_rows(sample: str, d: pd.DataFrame, d_2x: pd.DataFrame, cost_bps: float) -> list[dict]:
    turnover = float(d["turnover"].mean() * 252)

    def row(series: str, total: pd.Series, excess: pd.Series, turn: float) -> dict:
        m = portfolio_metrics(total, excess)
        return {"Sample": sample, "Series": series, **{k: m.get(k, np.nan) for k in ["Excess return (ann.)", "Volatility (ann.)", "Sharpe", "Max drawdown"]}, "Turnover (×/yr)": turn}

    return [
        row(f"Diversified trend, net of {cost_bps:g} bps", d["strategy_ret"], d["strategy_excess"], turnover),
        row(f"Diversified trend, net of {2 * cost_bps:g} bps (2× costs)", d_2x["strategy_ret"], d_2x["strategy_excess"], turnover),
        row("Diversified trend, before costs", d["strategy_gross_ret"], d["strategy_gross_excess"], turnover),
        row("Equal-weight buy & hold", d["equal_weight_ret"], d["equal_weight_excess"], 0.0),
    ]


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
        prices_all, risk_free = load_inputs(tuple(tickers), st.session_state.trend_refresh)
except Exception as exc:
    st.error(f"Research run failed: {exc}")
    st.stop()

with st.sidebar:
    min_d, max_d = prices_all.index[0].date(), prices_all.index[-1].date()
    date_range = st.slider(
        "Sample period", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM",
        help="Starts where every selected asset has prices. The holdout is cut from the end of whatever sample you pick.",
    )

prices = prices_all.loc[str(date_range[0]) : str(date_range[1])]
if len(prices) < 2 * 252:
    st.warning("Pick a sample of at least two years.")
    st.stop()

config = (tuple(tickers), lookback_months, volatility_lookback, target_volatility, cost_bps, max_asset_weight, max_gross_leverage, str(date_range[0]), str(date_range[1]))
h = setup_holdout("Diversified trend · " + "/".join(tickers), prices.index, config)
prices_is = prices.loc[: h.is_end]

params = dict(
    lookback_days=lookback_months * 21,
    volatility_lookback=volatility_lookback,
    target_volatility=target_volatility,
    max_asset_weight=max_asset_weight,
    max_gross_leverage=max_gross_leverage,
)
try:
    # Signals use month-end closes and apply from the next session, so one run over the sample can be cut by date.
    result_all = run_diversified_trend_backtest(prices, risk_free, cost_bps=cost_bps, **params)
    result_all_2x = run_diversified_trend_backtest(prices, risk_free, cost_bps=2 * cost_bps, **params)
    result = run_diversified_trend_backtest(prices_is, risk_free, cost_bps=cost_bps, **params)
except Exception as exc:
    st.error(f"Research run failed: {exc}. The in-sample period must be longer than the momentum lookback.")
    st.stop()
spx = spx_growth(prices.index, risk_free.reindex(prices.index).ffill().shift(1).fillna(0.0))

daily_all = result_all.daily
daily = result.daily
daily_oos = daily_all.loc[h.oos_start :]
if h.revealed:
    log_evaluation(
        h,
        portfolio_metrics(daily_oos["strategy_ret"], daily_oos["strategy_excess"])["Sharpe"],
        f"{'/'.join(tickers)}, {lookback_months}m momentum, {volatility_lookback}d vol, {target_volatility:.0%} target, "
        f"{cost_bps:g} bps, max weight {max_asset_weight:g}, max gross {max_gross_leverage:g}",
    )

strategy = result.summary_stats["Diversified trend"]
benchmark = result.summary_stats["Equal-weight buy & hold"]

st.caption(
    f"In-sample: {daily.index[0].date()} to {daily.index[-1].date()} · "
    f"{len(tickers)} assets · signals rebalance monthly from prior-close data"
)
metrics = st.columns(5)
metrics[0].metric("Net Sharpe", f"{strategy['Sharpe']:.2f}", f"benchmark {benchmark['Sharpe']:.2f}")
metrics[1].metric("Annual excess", f"{strategy['Excess return (ann.)']:.2%}")
metrics[2].metric("Annual volatility", f"{strategy['Volatility (ann.)']:.2%}")
metrics[3].metric("Max drawdown", f"{strategy['Max drawdown']:.1%}")
metrics[4].metric("Turnover cost", f"{daily['cost'].sum():.2%}", f"{strategy['Turnover (ann.)']:.1f}×/yr turnover", delta_color="off", delta_arrow="off")

tab_performance, tab_oos, tab_exposure, tab_robustness, tab_method = st.tabs(
    ["Performance", "Out-of-sample", "Exposure", "Subperiods", "Method & evidence"]
)

with tab_performance:
    lines = [
        Line(f"Diversified trend, net of {cost_bps:g} bps", growth(daily_all["strategy_excess"]), dict(width=2.5, color="#176b87")),
        Line("Diversified trend, before costs", growth(daily_all["strategy_gross_excess"]), dict(width=1.5, color="#176b87", dash="dot")),
        Line("Equal-weight buy & hold", growth(daily_all["equal_weight_excess"]), dict(width=1.5, color="#d17a22")),
    ]
    if spx is not None:
        s = spx.loc[daily_all.index]
        lines.append(Line(SPX_NAME, s / s.iloc[0], SPX_LINE))
    curve, target = equity_chart(lines, h, "Growth of $1 in excess of T-bills", height=430)
    show_equity_chart(curve, target, h)
    st.caption(
        "Dotted line: the same positions before trading costs; the gap to the solid line is the cost drag. "
        "Dashed amber line: where the out-of-sample holdout starts. Green line: the S&P 500 (SPY with dividends) for reference."
    )

    summary = pd.DataFrame(result.summary_stats).T
    st.dataframe(
        summary.style.format(
            {
                "CAGR": "{:.2%}",
                "Excess return (ann.)": "{:.2%}",
                "Volatility (ann.)": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.1%}",
                "Turnover (ann.)": "{:.1f}×",
            },
            na_rep="–",
        ),
        width="stretch",
    )
    st.caption("In-sample only.")

    yearly = daily.groupby(daily.index.year).agg(
        Trend_excess=("strategy_excess", lambda values: (1 + values).prod() - 1),
        Benchmark_excess=("equal_weight_excess", lambda values: (1 + values).prod() - 1),
    )
    yearly.columns = ["Trend excess", "Benchmark excess"]
    st.dataframe(yearly.style.format("{:.2%}"), width="stretch")

with tab_oos:
    if not h.revealed:
        locked_message(h)
    else:
        daily_oos_2x = result_all_2x.daily.loc[h.oos_start :]
        st.markdown(f"#### Out-of-sample: {daily_oos.index[0]:%Y-%m-%d} to {daily_oos.index[-1]:%Y-%m-%d}")
        rows = comparison_rows("In-sample", daily, result_all_2x.daily.loc[daily.index[0] : h.is_end], cost_bps) + comparison_rows("Out-of-sample", daily_oos, daily_oos_2x, cost_bps)
        st.dataframe(comparison_table(rows), width="stretch", hide_index=True)

        is_sharpe = strategy["Sharpe"]
        oos_sharpe = portfolio_metrics(daily_oos["strategy_ret"], daily_oos["strategy_excess"])["Sharpe"]
        se = 1 / np.sqrt(len(daily_oos) / 252)
        st.markdown(
            f"Net Sharpe **{is_sharpe:.2f}** in-sample against **{oos_sharpe:.2f}** out-of-sample. "
            f"Over {len(daily_oos) / 252:.1f} years, noise alone moves a Sharpe by about ±{se:.1f} (one standard error), "
            f"so the out-of-sample number is {'within' if abs(oos_sharpe - is_sharpe) <= se else 'outside'} one standard error of in-sample. "
            "Report it either way."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_oos.index, y=growth(daily_oos["strategy_excess"]), name=f"Diversified trend, net of {cost_bps:g} bps", line=dict(width=2.5, color="#176b87")))
        fig.add_trace(go.Scatter(x=daily_oos.index, y=growth(daily_oos["strategy_gross_excess"]), name="Diversified trend, before costs", line=dict(width=1.5, color="#176b87", dash="dot")))
        fig.add_trace(go.Scatter(x=daily_oos.index, y=growth(daily_oos_2x["strategy_excess"]), name=f"Diversified trend, net of {2 * cost_bps:g} bps", line=dict(width=1.5, color="#7c3aed")))
        fig.add_trace(go.Scatter(x=daily_oos.index, y=growth(daily_oos["equal_weight_excess"]), name="Equal-weight buy & hold", line=dict(width=1.5, color="#d17a22")))
        if spx is not None:
            s = spx.loc[daily_oos.index]
            fig.add_trace(go.Scatter(x=daily_oos.index, y=s / s.iloc[0], name=SPX_NAME, line=SPX_LINE))
        fig.update_layout(title="Out-of-sample growth of $1 in excess of T-bills", height=400, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

with tab_exposure:
    weights = result_all.weights if h.revealed else result.weights
    exposure = go.Figure()
    for ticker in weights.columns:
        exposure.add_trace(
            go.Scatter(x=weights.index, y=weights[ticker], name=ticker, stackgroup=None)
        )
    exposure.update_layout(
        title="Signed asset weights (positive = long, negative = short)",
        yaxis_title="Weight of portfolio NAV",
        height=430,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(exposure, width="stretch")
    st.metric("Average gross exposure", f"{weights.abs().sum(axis=1).mean():.2f}x")

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
        prices_is,
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
    st.caption(
        "Fixed 6-, 12-, and 18-month horizons; this is a sensitivity check, not an optimized selection. "
        f"Subperiods and sensitivity use in-sample data only (up to {h.is_end:%Y-%m-%d}), so \"Full sample\" means the full in-sample period."
    )

with tab_method:
    st.markdown(
        """
### Backtest definition
- Signal is the sign of each asset's trailing total return. Signals and trailing volatility are measured at each month-end close and positions apply from the next session.
- Each available asset receives an equal share of the target risk budget, scaled by trailing 63-day volatility. Per-asset and total gross exposure are capped.
- Strategy return is T-bill return plus signed weighted asset excess returns. Costs are charged against each change in notional weight.
- The comparator is an equal-weight, long-only portfolio of the selected assets over exactly the same days, with no turnover costs.
- The most recent 20% of the selected sample or 2 years, whichever is shorter, is held out behind the amber window on the equity chart. Everything else on this page uses the in-sample period until you click it.

### Important limits
These are adjusted ETF prices, not futures returns. They provide an accessible first test of cross-asset trend, but omit futures-specific roll yield, financing, margin, and market-impact details. The selected ETFs also do not reproduce the 58-market academic portfolio. Yahoo prices, proxy choice, fees, and parameter choices all affect results; treat the Sharpe as a hypothesis check, not an expected live return.

### Research references
- [Moskowitz, Ooi & Pedersen, Time Series Momentum (2012)](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- [Hurst, Ooi & Pedersen, A Century of Evidence on Trend-Following Investing (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)
- [Daniel & Moskowitz, Momentum Crashes (NBER)](https://www.nber.org/papers/w20439)
"""
    )
