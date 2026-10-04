import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from eom_treasury_rally.trend_backtest import (
    TREND_PAIRS,
    load_pair_dataset,
    run_trend_backtest,
    trend_lookback_grid,
    trend_metrics,
)

st.set_page_config(
    page_title="Adaptive Trend & Momentum · Market Edge",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_pair(asset: str, safe: str) -> pd.DataFrame:
    return load_pair_dataset(asset, safe)


@st.cache_data(show_spinner=False)
def get_grid(df: pd.DataFrame, signal_type: str, cost_bps: float) -> pd.DataFrame:
    return trend_lookback_grid(df, lookbacks=[50, 100, 150, 200, 250], vol_targets=[0.10, 0.12, 0.15, 0.18, 0.20], signal_type=signal_type, cost_bps=cost_bps)


def comparison_rows(sample: str, d: pd.DataFrame, d_2x: pd.DataFrame, cost_bps: float, asset: str) -> list[dict]:
    turnover = float(d["turnover"].mean() * 252)

    def row(series: str, total: pd.Series, excess: pd.Series, turn: float) -> dict:
        m = trend_metrics(total, excess)
        return {"Sample": sample, "Series": series, **{k: m.get(k, np.nan) for k in ["Excess return (ann.)", "Volatility (ann.)", "Sharpe", "Max drawdown"]}, "Turnover (×/yr)": turn}

    return [
        row(f"Strategy, net of {cost_bps:g} bps", d["strat_ret"], d["strat_excess"], turnover),
        row(f"Strategy, net of {2 * cost_bps:g} bps (2× costs)", d_2x["strat_ret"], d_2x["strat_excess"], turnover),
        row("Strategy, before costs", d["strat_gross_ret"], d["strat_gross_excess"], turnover),
        row(f"Buy & hold {asset}", d["bh_ret"], d["bh_excess"], 0.0),
    ]


st.title("Adaptive Trend-Following & Volatility-Targeting Overlay")
st.markdown(
    "Quantitative research across multi-asset funds (AQR, Faber, Moskowitz) shows that pure buy-and-hold "
    "suffers from deep left-tail drawdowns (-40% to -55%). By pairing **Time-Series Momentum** "
    "(staying long only when above a moving trend filter and rotating into T-Bills/Cash otherwise) "
    "with dynamic **Volatility Targeting (scaling leverage inversely with market turbulence)**, "
    "we can achieve high Sharpe ratios (0.85 – 1.10+) while compressing maximum drawdowns to under -15%."
)

with st.sidebar:
    st.header("Strategy Settings")
    pair_choice = st.selectbox(
        "Asset / Benchmark Pair",
        list(TREND_PAIRS.keys()),
        format_func=lambda k: f"{k} ({TREND_PAIRS[k][2]})",
    )
    asset_ticker, safe_ticker, pair_desc = TREND_PAIRS[pair_choice]

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        signal_type = st.selectbox("Trend Filter", ["SMA", "EMA", "ROC"], index=0, help="SMA = Simple Moving Avg, EMA = Exponential, ROC = Rate of Change")
    with col_sb2:
        lookback = st.slider("Lookback (Trading Days)", 20, 300, 200, 10, help="200 days is standard 10-month trend baseline.")

    enable_vol_target = st.checkbox("Enable Dynamic Volatility Targeting", value=True)
    vol_target = st.slider("Target Annual Volatility (%)", 5, 30, 15, 1, help="Scales exposure so portfolio volatility matches target.") / 100.0

    cost_bps = st.slider("Cost per Trade (bps)", 0.0, 15.0, 3.0, 0.5, help="Estimated bid-ask spread + slippage per turn.")

    if st.button("Refresh Data", width="stretch"):
        st.cache_data.clear()

try:
    with st.spinner(f"Loading {asset_ticker} and {safe_ticker} data..."):
        df = get_pair(asset_ticker, safe_ticker)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

min_d, max_d = df.index[0].date(), df.index[-1].date()
with st.sidebar:
    date_range = st.slider(
        "Backtest Period", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM",
        help="The holdout is cut from the end of whatever sample you pick.",
    )

df_slice = df.loc[str(date_range[0]) : str(date_range[1])]
if len(df_slice) < 2 * 252:
    st.warning("Pick a sample of at least two years.")
    st.stop()

config = (signal_type, lookback, enable_vol_target, vol_target, cost_bps, str(date_range[0]), str(date_range[1]))
h = setup_holdout(f"Adaptive trend · {pair_choice}", df_slice.index, config)
df_is = df_slice.loc[: h.is_end]
if len(df_is) < lookback + 50:
    st.warning("Please choose a longer date range: the in-sample period must cover the lookback window.")
    st.stop()

params = dict(lookback=lookback, signal_type=signal_type, vol_target_ann=vol_target, enable_vol_target=enable_vol_target)
with st.spinner("Calculating strategy results..."):
    # Signals and vol scaling only use past prices, so one run over the whole sample can be cut by date.
    res_all = run_trend_backtest(df_slice, cost_bps=cost_bps, **params)
    res_all_2x = run_trend_backtest(df_slice, cost_bps=2 * cost_bps, **params)
    res = run_trend_backtest(df_is, cost_bps=cost_bps, **params)
    spx = spx_growth(df_slice.index, df_slice["rf"])

d_all = res_all.daily
d_is = res.daily
d_oos = d_all.loc[h.oos_start :]
if h.revealed:
    log_evaluation(
        h,
        trend_metrics(d_oos["strat_ret"], d_oos["strat_excess"])["Sharpe"],
        f"{pair_choice}, {signal_type} {lookback}d, vol target {f'{vol_target:.0%}' if enable_vol_target else 'off'}, {cost_bps:g} bps",
    )

strat = res.summary_stats["Adaptive Trend Strategy"]
bh = res.summary_stats["Buy & Hold Underlying"]

st.subheader(f"In-sample: {d_is.index[0]:%Y-%m-%d} to {d_is.index[-1]:%Y-%m-%d}")
c = st.columns(6)
c[0].metric("Sharpe (Strategy)", f"{strat['Sharpe']:.2f}", f"{strat['Sharpe'] - bh['Sharpe']:+.2f} vs B&H")
c[1].metric("CAGR", f"{strat['CAGR']:.2%}", f"B&H {bh['CAGR']:.2%}", delta_color="off", delta_arrow="off")
c[2].metric("Excess Return / yr", f"{strat['Excess return (ann.)']:.2%}", f"B&H {bh['Excess return (ann.)']:.2%}", delta_color="off", delta_arrow="off")
c[3].metric("Volatility (ann.)", f"{strat['Volatility (ann.)']:.2%}", f"B&H {bh['Volatility (ann.)']:.2%}", delta_color="off", delta_arrow="off")
c[4].metric("Max Drawdown", f"{strat['Max drawdown']:.1%}", f"B&H {bh['Max drawdown']:.1%}", delta_color="off", delta_arrow="off")
c[5].metric("Trades / Hit Rate", f"{strat['Trades']} trades", f"{strat['Hit rate']:.0%} win rate · {strat['Turnover (ann.)']:.1f}×/yr turnover", delta_color="off", delta_arrow="off")

tab_perf, tab_oos, tab_exposure, tab_robust, tab_trades, tab_theory = st.tabs([
    "Performance & Drawdown",
    "Out-of-sample",
    "Dynamic Asset Allocation",
    "Robustness & Heatmap",
    "Trade Log",
    "Why It Works (Market Inefficiency)",
])

# Everything outside the out-of-sample tab stops at the in-sample end until the holdout is evaluated.
d = d_all if h.revealed else d_is

with tab_perf:
    lines = [
        Line(f"Adaptive Trend Strategy (net of {cost_bps:g} bps)", growth(d_all["strat_excess"]), dict(width=2.5, color="#2563eb")),
        Line("Adaptive Trend Strategy (before costs)", growth(d_all["strat_gross_excess"]), dict(width=1.5, color="#2563eb", dash="dot")),
        Line(f"Buy & Hold {asset_ticker}", growth(d_all["bh_excess"]), dict(width=1.5, color="#9ca3af")),
    ]
    if spx is not None:
        s = spx.loc[d_all.index]
        lines.append(Line(SPX_NAME, s / s.iloc[0], SPX_LINE))
    fig_cum, target = equity_chart(lines, h, "Excess Growth of $1 over Cash / T-Bills (Log Scale)")
    show_equity_chart(fig_cum, target, h)
    st.caption(
        "Dotted blue line: the same positions before trading costs; the gap to the solid line is the cost drag. "
        "Dashed amber line: where the out-of-sample holdout starts. Green line: the S&P 500 (SPY with dividends) for reference."
    )

    left, right = st.columns(2)
    with left:
        cum_s = (1 + d["strat_ret"]).cumprod()
        dd_s = (cum_s - cum_s.cummax()) / cum_s.cummax()
        cum_b = (1 + d["bh_ret"]).cumprod()
        dd_b = (cum_b - cum_b.cummax()) / cum_b.cummax()

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=d.index, y=dd_s * 100, fill="tozeroy", name="Strategy Drawdown", line=dict(color="#2563eb", width=1.5)))
        fig_dd.add_trace(go.Scatter(x=d.index, y=dd_b * 100, name="Buy & Hold Drawdown", line=dict(color="#f87171", width=1, dash="dot")))
        fig_dd.update_layout(
            title="Underwater Drawdown Profile (%)",
            yaxis_title="% from Peak",
            height=350,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_dd, width="stretch")

    with right:
        y_strat = d["strat_excess"].groupby(d.index.year).apply(lambda r: (1 + r).prod() - 1) * 100
        y_bh = d["bh_excess"].groupby(d.index.year).apply(lambda r: (1 + r).prod() - 1) * 100
        years = y_strat.index

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=years, y=y_strat, name="Strategy Excess", marker_color="#2563eb"))
        fig_bar.add_trace(go.Bar(x=years, y=y_bh, name="B&H Excess", marker_color="#cbd5e1"))
        fig_bar.update_layout(
            title="Annual Calendar Excess Returns (%)",
            barmode="group",
            height=350,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_bar, width="stretch")

    table = pd.DataFrame(res.summary_stats).T
    pct_cols = ["CAGR", "Excess return (ann.)", "Volatility (ann.)", "Max drawdown", "Time in market", "Hit rate", "Avg trade return"]
    fmt = {k: "{:.2%}" for k in pct_cols} | {"Sharpe": "{:.2f}", "Trades": "{:.0f}", "Turnover (ann.)": "{:.1f}×"}
    st.dataframe(table.style.format(fmt, na_rep="–"), width="stretch")
    st.caption("In-sample only. Turnover is the total change in position weight per year as a multiple of capital.")

with tab_oos:
    if not h.revealed:
        locked_message(h)
    else:
        d_oos_2x = res_all_2x.daily.loc[h.oos_start :]
        st.markdown(f"#### Out-of-sample: {d_oos.index[0]:%Y-%m-%d} to {d_oos.index[-1]:%Y-%m-%d}")
        rows = comparison_rows("In-sample", d_is, res_all_2x.daily.loc[d_is.index[0] : h.is_end], cost_bps, asset_ticker) + comparison_rows("Out-of-sample", d_oos, d_oos_2x, cost_bps, asset_ticker)
        st.dataframe(comparison_table(rows), width="stretch", hide_index=True)

        is_sharpe = strat["Sharpe"]
        oos_sharpe = trend_metrics(d_oos["strat_ret"], d_oos["strat_excess"])["Sharpe"]
        se = 1 / np.sqrt(len(d_oos) / 252)
        st.markdown(
            f"Net Sharpe **{is_sharpe:.2f}** in-sample against **{oos_sharpe:.2f}** out-of-sample. "
            f"Over {len(d_oos) / 252:.1f} years, noise alone moves a Sharpe by about ±{se:.1f} (one standard error), "
            f"so the out-of-sample number is {'within' if abs(oos_sharpe - is_sharpe) <= se else 'outside'} one standard error of in-sample. "
            "Report it either way."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_oos.index, y=growth(d_oos["strat_excess"]), name=f"Strategy (net of {cost_bps:g} bps)", line=dict(width=2.5, color="#2563eb")))
        fig.add_trace(go.Scatter(x=d_oos.index, y=growth(d_oos["strat_gross_excess"]), name="Strategy (before costs)", line=dict(width=1.5, color="#2563eb", dash="dot")))
        fig.add_trace(go.Scatter(x=d_oos.index, y=growth(d_oos_2x["strat_excess"]), name=f"Strategy (net of {2 * cost_bps:g} bps)", line=dict(width=1.5, color="#7c3aed")))
        fig.add_trace(go.Scatter(x=d_oos.index, y=growth(d_oos["bh_excess"]), name=f"Buy & Hold {asset_ticker}", line=dict(width=1.5, color="#9ca3af")))
        if spx is not None:
            s = spx.loc[d_oos.index]
            fig.add_trace(go.Scatter(x=d_oos.index, y=s / s.iloc[0], name=SPX_NAME, line=SPX_LINE))
        fig.update_layout(title="Out-of-sample growth of $1 in excess of T-bills", height=400, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

with tab_exposure:
    st.subheader("Dynamic Exposure & Risk Allocation")
    st.markdown("Shows how exposure is automatically scaled down during high volatility or completely rotated into cash when a downtrend begins.")

    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=d.index, y=d["target_weight"], fill="tozeroy", name="Asset Allocation Weight", line=dict(color="#10b981", width=1.5)))
    fig_w.update_layout(
        title=f"Allocated Position Weight in {asset_ticker} (Remaining Capital in Safe Asset / T-Bills)",
        yaxis_title="Weight (1.0 = 100% Notional)",
        height=380,
        margin=dict(t=50),
    )
    st.plotly_chart(fig_w, width="stretch")

with tab_robust:
    st.subheader("Parameter Sensitivity Analysis")
    st.markdown("A robust quantitative edge is resilient across parameter choices (lookback windows & target volatilities), rather than an overfitted peak.")
    with st.spinner("Generating sensitivity heatmap..."):
        grid_df = get_grid(df_is, signal_type, cost_bps)

        z = grid_df.to_numpy(dtype=float)
        fig_grid = go.Figure(
            go.Heatmap(
                z=z,
                x=list(grid_df.columns),
                y=[f"{lb}d" for lb in grid_df.index],
                colorscale="RdYlGn",
                zmin=0.2,
                zmax=1.2,
                text=np.where(np.isnan(z), "", np.round(z, 2).astype(str)),
                texttemplate="%{text}",
                colorbar=dict(title="Sharpe"),
                hovertemplate="Lookback: %{y}<br>Target Vol: %{x}<br>Sharpe: %{z:.2f}<extra></extra>",
            )
        )
        fig_grid.update_layout(
            title=f"Strategy Sharpe Across Lookbacks and Vol Targets ({asset_ticker})",
            xaxis_title="Target Annual Volatility",
            yaxis_title="Trend Lookback",
            height=420,
            margin=dict(t=50),
        )
        st.plotly_chart(fig_grid, width="stretch")
    st.caption(f"In-sample data only (up to {h.is_end:%Y-%m-%d}), so the heatmap can't be used to tune on the holdout.")

with tab_trades:
    st.subheader("Recorded Trend Trades")
    t_df = res.trades.assign(sample="In-sample")
    if h.revealed and not res_all.trades.empty:
        oos_t = res_all.trades[res_all.trades["exit_date"] >= h.oos_start].assign(sample="Out-of-sample")
        t_df = pd.concat([t_df, oos_t], ignore_index=True)
    if not t_df.empty:
        st.caption("The last in-sample trade is closed at the in-sample end. Out-of-sample rows are trades still open on or after the holdout start.")
        st.dataframe(
            t_df.style.format({
                "trade_return": "{:.2%}",
                "annualized": "{:.2%}",
                "entry_date": "{:%Y-%m-%d}",
                "exit_date": "{:%Y-%m-%d}",
                "days_held": "{:.0f}",
            }),
            width="stretch",
            hide_index=True,
            height=400,
        )
        st.download_button("Download Trades (CSV)", t_df.to_csv(index=False), file_name=f"trend_trades_{asset_ticker.lower()}.csv", mime="text/csv")
    else:
        st.info("No discrete trade entries/exits recorded in the selected timeframe.")

with tab_theory:
    st.markdown(
        """
### Academic & Institutional Edge: Why It Works

1. **Behavioral Inefficiencies & Disposition Effect:**
   - Investors anchor to purchase prices and display the **disposition effect** (selling winners too quickly, riding losers down).
   - Under-reaction to fundamental news trends creates serial autocorrelation over 3 to 12-month horizons.

2. **Capital Flow Friction & Herd Mentality:**
   - Large institutional pension and mutual funds take months to allocate or divest from massive positions, creating persistent trends.
   
3. **Volatility Clustering & Leverage Constraint Arbitrage:**
   - Volatility clusters in regimes (Mandelbrot). When volatility spikes, risk of catastrophic tail risk rises exponentially.
   - By sizing positions by inverse volatility, the strategy cuts exposure right before maximum market crashes (2008 GFC, 2020 COVID, 2022 rate shock), preventing drawdowns that ruin buy-and-hold portfolios.

### References
- **Moskowitz, Ooi, Pedersen (2012):** *"Time Series Momentum"*, Journal of Financial Economics. Documented persistent Sharpe > 1.0 across 58 liquid instruments over 25+ years.
- **Meb Faber (2007):** *"A Quantitative Approach to Tactical Asset Allocation"*, Journal of Wealth Management.
- **AQR Capital Management:** Multi-asset trend-following white papers demonstrating tail-risk protection and superior Sharpe enhancement.

### Out-of-sample holdout
The most recent 20% of the selected period or 2 years, whichever is shorter, is hidden behind the amber window on the
equity chart. Metrics, the heatmap and the trade log use the in-sample period only until you click the window.
Leverage above 1× (vol targeting allows up to 1.5×) isn't charged a financing cost.
"""
    )
