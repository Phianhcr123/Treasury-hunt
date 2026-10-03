import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.backtest import (
    Window,
    day_of_month_profile,
    permutation_test,
    run_backtest,
    sensitivity_grid,
    subperiod_table,
    summary,
    yearly_returns,
)
from eom_treasury_rally.data import TREASURY_ETFS, DataError, load_dataset

PRE_REGISTERED = Window(entry=3, exit=0)

st.set_page_config(page_title="Month-End Treasury Rally", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(ticker: str, refresh_token: int) -> pd.DataFrame:
    return load_dataset(ticker, refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def get_grid(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    return sensitivity_grid(df, cost_bps=cost_bps)


@st.cache_data(show_spinner=False)
def get_perm(df: pd.DataFrame, entry: int, exit_: int) -> dict:
    return permutation_test(df, Window(entry, exit_), n_sims=2000)


if "refresh" not in st.session_state:
    st.session_state.refresh = 0

with st.sidebar:
    st.header("Strategy settings")
    ticker = st.selectbox("Treasury ETF", list(TREASURY_ETFS), format_func=lambda t: f"{t} · {TREASURY_ETFS[t]}")
    entry = st.slider("Enter N trading days before month-end", 1, 10, PRE_REGISTERED.entry)
    exit_ = st.slider("Exit (days relative to month-end close)", -2, 3, PRE_REGISTERED.exit, help="0 = last trading day of the month, +1 = first trading day of the next month")
    cost_bps = st.slider("Trading cost per side (bps)", 0.0, 10.0, 2.0, 0.5, help="Spread + commission + slippage. TLT's spread is about 1 bp.")
    if st.button("Re-download latest data", use_container_width=True):
        st.session_state.refresh += 1
        st.cache_data.clear()
    st.caption("Prices: Yahoo Finance, dividend-adjusted. Cash earns the 13-week T-bill rate (^IRX).")

st.title("Month-End Treasury Rally")
st.markdown(
    "Bond index funds rebalance on the **last trading day of each month**. Funds benchmarked to those indices "
    "must buy Treasuries to match the new index duration regardless of price. This strategy buys a Treasury ETF "
    "a few days before month-end, sells at the month-end close, and sits in T-bills the rest of the month."
)

if entry + exit_ <= 0:
    st.error("The exit must come after the entry. Increase the entry days or move the exit later.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker} history…"):
        full = get_data(ticker, st.session_state.refresh)
except DataError as e:
    st.error(f"Couldn't load data: {e}")
    st.stop()
except Exception as e:  # network failures surface as many exception types
    st.error(f"Couldn't reach Yahoo Finance ({type(e).__name__}). Check the connection and press **Re-download latest data**.")
    st.stop()

min_d, max_d = full.index[0].date(), full.index[-1].date()
with st.sidebar:
    date_range = st.slider("Backtest period", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM")
df = full.loc[str(date_range[0]) : str(date_range[1])]
if len(df) < 252:
    st.warning("Pick a period of at least one year.")
    st.stop()

window = Window(entry, exit_)
if window != PRE_REGISTERED:
    st.info(
        f"You're exploring a different window from the pre-registered one (enter 3 days before, exit at month-end). "
        "Picking the best-looking window after seeing results inflates the Sharpe; check the heatmap below to see how robust it is."
    )

with st.spinner("Running backtest…"):
    res = run_backtest(df, window, cost_bps)
    s = summary(res)
    perm = get_perm(df, entry, exit_)
    grid = get_grid(df, cost_bps)
    profile = day_of_month_profile(df)
    subs = subperiod_table(df, window, cost_bps)
    yearly = yearly_returns(res)

strat, bh = s["Month-end strategy"], s["Buy & hold"]
c = st.columns(6)
c[0].metric("Sharpe (strategy)", f"{strat['Sharpe']:.2f}", f"{strat['Sharpe'] - bh['Sharpe']:+.2f} vs buy & hold")
c[1].metric("Excess return / yr", f"{strat['Excess return (ann.)']:.2%}", f"buy & hold {bh['Excess return (ann.)']:.2%}", delta_color="off")
c[2].metric("Max drawdown", f"{strat['Max drawdown']:.1%}", f"buy & hold {bh['Max drawdown']:.1%}", delta_color="off")
c[3].metric("Avg trade (net)", f"{strat.get('Avg trade (net excess)', np.nan):.2%}", f"{int(strat.get('Trades', 0))} trades", delta_color="off")
c[4].metric("Hit rate", f"{strat.get('Hit rate', np.nan):.0%}", f"in market {strat['Time in market']:.0%} of days", delta_color="off")
c[5].metric("Permutation p-value", f"{perm['p_value']:.4f}", "vs random windows", delta_color="off")

tab_overview, tab_robust, tab_trades, tab_about = st.tabs(["Performance", "Robustness", "Trades", "How it works"])

with tab_overview:
    d = res.daily
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=(1 + d["strategy_excess"]).cumprod(), name="Month-end strategy", line=dict(width=2.5, color="#2563eb")))
    fig.add_trace(go.Scatter(x=d.index, y=(1 + d["buy_hold_excess"]).cumprod(), name=f"Buy & hold {ticker}", line=dict(width=1.5, color="#9ca3af")))
    fig.add_trace(go.Scatter(x=d.index, y=(1 + d["rest_of_month_excess"]).cumprod(), name="Rest of month only", line=dict(width=1.5, color="#f97316", dash="dot")))
    fig.update_layout(title="Growth of $1 in excess of T-bills (log scale)", yaxis_type="log", height=430, legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        held = [o for o in profile.index if (-entry < o <= min(exit_, 0)) or (0 < o <= exit_)]
        fig = go.Figure(
            go.Bar(
                x=[f"{o}" if o <= 0 else f"+{o}" for o in profile.index],
                y=profile["mean"] * 1e4,
                error_y=dict(type="data", array=profile["ci95"] * 1e4),
                marker_color=["#2563eb" if o in held else "#cbd5e1" for o in profile.index],
                hovertemplate="Day %{x}<br>%{y:.1f} bps<extra></extra>",
            )
        )
        fig.update_layout(title="Average daily excess return by day of month (blue = held)", xaxis_title="Trading day vs month-end (0 = last day)", yaxis_title="bps per day", height=380, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["Strategy excess"] * 100, name="Strategy", marker_color="#2563eb"))
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["Buy & hold excess"] * 100, name="Buy & hold", marker_color="#cbd5e1"))
        fig.update_layout(title="Calendar-year excess return (%)", barmode="group", height=380, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame(s).T
    pct_cols = ["CAGR", "Excess return (ann.)", "Volatility (ann.)", "Max drawdown", "Time in market", "Avg trade (net excess)", "Hit rate"]
    fmt = {k: "{:.2%}" for k in pct_cols} | {"Sharpe": "{:.2f}", "t-stat (per trade)": "{:.2f}", "Trades": "{:.0f}"}
    st.dataframe(table.style.format(fmt, na_rep="–"), use_container_width=True)

with tab_robust:
    st.subheader("Does it still work? Sub-period results")
    st.dataframe(
        subs.style.format({"Strategy Sharpe": "{:.2f}", "Buy & hold Sharpe": "{:.2f}", "Strategy excess (ann.)": "{:.2%}", "Avg trade (net)": "{:.2%}", "Hit rate": "{:.0%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The original study (Hartley & Schwarz) used data through 2018, so 2019+ is a genuine out-of-sample test.")

    left, right = st.columns([3, 2])
    with left:
        z = grid.to_numpy(dtype=float)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[f"{x:+d}" for x in grid.columns],
                y=[f"-{e}" for e in grid.index],
                colorscale="RdYlGn",
                zmin=-1,
                zmax=1.5,
                text=np.where(np.isnan(z), "", np.round(z, 2).astype(str)),
                texttemplate="%{text}",
                colorbar=dict(title="Sharpe"),
                hovertemplate="Entry %{y}, exit %{x}<br>Sharpe %{z:.2f}<extra></extra>",
            )
        )
        r, col = list(grid.index).index(entry), list(grid.columns).index(exit_)
        fig.add_shape(type="rect", x0=col - 0.5, x1=col + 0.5, y0=r - 0.5, y1=r + 0.5, line=dict(color="black", width=3))
        fig.update_layout(title="Sharpe for every entry/exit window (box = current)", xaxis_title="Exit vs month-end close", yaxis_title="Entry (days before month-end)", yaxis_autorange="reversed", height=470, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Is month-end special?")
        st.markdown(
            f"Each of 2,000 simulations holds a **random {window.holding_days}-day window** inside every month. "
            f"The month-end window averaged **{perm['actual_avg_trade']:.3%}** per trade before costs, against "
            f"**{perm['random_avg_trade']:.3%}** for random windows."
        )
        verdict = "very unlikely to be luck" if perm["p_value"] < 0.01 else "could plausibly be luck" if perm["p_value"] > 0.05 else "borderline"
        st.markdown(f"p-value **{perm['p_value']:.4f}**: the month-end result is {verdict}.")
        st.markdown("#### Reading the heatmap")
        st.markdown(
            "A real effect shows up as a broad green region, not a single bright cell. "
            "Here the windows that **exit at the month-end close (+0)** are consistently the strongest, "
            "which matches the rebalancing explanation: the buying pressure ends when the index resets."
        )

with tab_trades:
    t = res.trades.copy()
    st.markdown(f"{len(t)} trades. Returns are for the holding window; *excess* subtracts T-bill interest, *net* also subtracts costs.")
    st.dataframe(
        t.style.format({"gross": "{:.2%}", "excess": "{:.2%}", "net_excess": "{:.2%}", "entry_close": "{:%Y-%m-%d}", "exit_close": "{:%Y-%m-%d}"}),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.download_button("Download trades (CSV)", t.to_csv(index=False), file_name=f"eom_trades_{ticker.lower()}.csv", mime="text/csv")

with tab_about:
    st.markdown(
        """
### Why the edge exists
- **Index rebalancing.** Treasury indices (e.g. Bloomberg US Treasury) add newly issued bonds and drop short ones
  on the last business day of the month, which extends the index's duration.
- **Price-insensitive buyers.** Index funds, insurers and pensions benchmarked to these indices must buy longer bonds
  that day to avoid tracking error. Volume on the last trading day is roughly 50% higher than other days (NY Fed).
- **Limited arbitrage.** Dealer balance sheets are tight at month-end, so the buying pressure moves prices.

### Evidence
- Hartley & Schwarz (1990–2018): most of the Treasury term premium is earned in the last few days of the month, Sharpe about 1.
- The effect is smaller after 2015 and very weak in the last three years. Present that honestly.

### Assumptions in this backtest
- Trades at the closing price via market-on-close orders. The schedule is fixed by the calendar, so there is no look-ahead.
- Dividend-adjusted ETF prices from Yahoo Finance. Cash earns the 13-week T-bill rate.
- Treasury futures (ZN, ZB) would be cheaper to trade, but Yahoo's continuous futures series isn't roll-adjusted and the
  quarterly rolls fall near month-end, which would contaminate the test. That's why this uses ETFs.
"""
    )
