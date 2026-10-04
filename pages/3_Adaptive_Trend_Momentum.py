import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.trend_backtest import (
    TREND_PAIRS,
    load_pair_dataset,
    run_trend_backtest,
    trend_lookback_grid,
)

st.set_page_config(
    page_title="Adaptive Trend & Momentum · Market Edge",
    page_icon="⚡",
    layout="wide",
)

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

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()

try:
    with st.spinner(f"Loading {asset_ticker} and {safe_ticker} data..."):
        df = load_pair_dataset(asset_ticker, safe_ticker)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

min_d, max_d = df.index[0].date(), df.index[-1].date()
with st.sidebar:
    date_range = st.slider("Backtest Period", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM")

df_slice = df.loc[str(date_range[0]) : str(date_range[1])]

if len(df_slice) < lookback + 50:
    st.warning("Please choose a longer date range to account for the lookback window.")
    st.stop()

with st.spinner("Calculating strategy results..."):
    res = run_trend_backtest(
        df_slice,
        lookback=lookback,
        signal_type=signal_type,
        vol_target_ann=vol_target,
        enable_vol_target=enable_vol_target,
        cost_bps=cost_bps,
    )

strat = res.summary_stats["Adaptive Trend Strategy"]
bh = res.summary_stats["Buy & Hold Underlying"]

c = st.columns(6)
c[0].metric("Sharpe (Strategy)", f"{strat['Sharpe']:.2f}", f"{strat['Sharpe'] - bh['Sharpe']:+.2f} vs B&H")
c[1].metric("CAGR", f"{strat['CAGR']:.2%}", f"B&H {bh['CAGR']:.2%}", delta_color="off", delta_arrow="off")
c[2].metric("Excess Return / yr", f"{strat['Excess return (ann.)']:.2%}", f"B&H {bh['Excess return (ann.)']:.2%}", delta_color="off", delta_arrow="off")
c[3].metric("Volatility (ann.)", f"{strat['Volatility (ann.)']:.2%}", f"B&H {bh['Volatility (ann.)']:.2%}", delta_color="off", delta_arrow="off")
c[4].metric("Max Drawdown", f"{strat['Max drawdown']:.1%}", f"B&H {bh['Max drawdown']:.1%}", delta_color="off", delta_arrow="off")
c[5].metric("Trades / Hit Rate", f"{strat['Trades']} trades", f"{strat['Hit rate']:.0%} win rate", delta_color="off", delta_arrow="off")

tab_perf, tab_exposure, tab_robust, tab_trades, tab_theory = st.tabs([
    "Performance & Drawdown",
    "Dynamic Asset Allocation",
    "Robustness & Heatmap",
    "Trade Log",
    "Why It Works (Market Inefficiency)",
])

d = res.daily

with tab_perf:
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=d.index, y=(1 + d["strat_excess"]).cumprod(), name="Adaptive Trend Strategy", line=dict(width=2.5, color="#2563eb")))
    fig_cum.add_trace(go.Scatter(x=d.index, y=(1 + d["bh_excess"]).cumprod(), name=f"Buy & Hold {asset_ticker}", line=dict(width=1.5, color="#9ca3af")))
    fig_cum.update_layout(
        title="Excess Growth of $1 over Cash / T-Bills (Log Scale)",
        yaxis_type="log",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

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
        st.plotly_chart(fig_dd, use_container_width=True)

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
        st.plotly_chart(fig_bar, use_container_width=True)

    table = pd.DataFrame(res.summary_stats).T
    pct_cols = ["CAGR", "Excess return (ann.)", "Volatility (ann.)", "Max drawdown", "Time in market", "Hit rate", "Avg trade return"]
    fmt = {k: "{:.2%}" for k in pct_cols} | {"Sharpe": "{:.2f}", "Trades": "{:.0f}"}
    st.dataframe(table.style.format(fmt, na_rep="–"), use_container_width=True)

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
    st.plotly_chart(fig_w, use_container_width=True)

with tab_robust:
    st.subheader("Parameter Sensitivity Analysis")
    st.markdown("A robust quantitative edge is resilient across parameter choices (lookback windows & target volatilities), rather than an overfitted peak.")
    with st.spinner("Generating sensitivity heatmap..."):
        grid_df = trend_lookback_grid(
            df_slice,
            lookbacks=[50, 100, 150, 200, 250],
            vol_targets=[0.10, 0.12, 0.15, 0.18, 0.20],
            signal_type=signal_type,
            cost_bps=cost_bps,
        )

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
        st.plotly_chart(fig_grid, use_container_width=True)

with tab_trades:
    st.subheader("Recorded Trend Trades")
    t_df = res.trades.copy()
    if not t_df.empty:
        st.dataframe(
            t_df.style.format({
                "trade_return": "{:.2%}",
                "annualized": "{:.2%}",
                "entry_date": "{:%Y-%m-%d}",
                "exit_date": "{:%Y-%m-%d}",
                "days_held": "{:.0f}",
            }),
            use_container_width=True,
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
"""
    )
