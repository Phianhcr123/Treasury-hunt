from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.backtest import (
    HOLDOUT_FRACTION,
    HOLDOUT_MAX_YEARS,
    BacktestResult,
    Window,
    day_of_month_profile,
    holdout_start,
    permutation_test,
    run_backtest,
    sensitivity_grid,
    slice_result,
    subperiod_table,
    summary,
    yearly_returns,
)
from eom_treasury_rally.data import TREASURY_ETFS, DataError, load_close, load_dataset

PRE_REGISTERED = Window(entry=3, exit=0)
OOS_LOG = Path(__file__).resolve().parent / "reports" / "oos_evaluations.csv"

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


@st.cache_data(show_spinner=False)
def get_spy(refresh_token: int) -> pd.Series:
    return load_close("SPY", refresh=refresh_token > 0)


def growth(excess: pd.Series) -> pd.Series:
    return (1 + excess).cumprod()


def benchmark_excess(close: pd.Series | None, daily: pd.DataFrame) -> pd.Series | None:
    """Daily total return of a benchmark over T-bills, on the backtest's calendar."""
    if close is None:
        return None
    ret = close.reindex(daily.index).ffill().pct_change().fillna(0.0)
    return ret - daily["rf"]


def comparison_rows(sample: str, net: BacktestResult, net_2x: BacktestResult) -> list[dict]:
    s, s2 = summary(net), summary(net_2x)
    strat, strat2, gross, bh = s["Month-end strategy"], s2["Month-end strategy"], s["Month-end strategy (before costs)"], s["Buy & hold"]
    t = net.trades

    def row(series: str, block: dict, turnover: float, trades: float, hit: float, avg: float) -> dict:
        return {
            "Sample": sample,
            "Series": series,
            "Excess return (ann.)": block["Excess return (ann.)"],
            "Volatility (ann.)": block["Volatility (ann.)"],
            "Sharpe": block["Sharpe"],
            "Max drawdown": block["Max drawdown"],
            "Turnover (×/yr)": turnover,
            "Trades": trades,
            "Hit rate": hit,
            "Avg trade": avg,
        }

    n = float(len(t))
    return [
        row(f"Strategy, net of {net.cost_bps:g} bps/side", strat, strat["Turnover (ann.)"], n, strat.get("Hit rate", np.nan), strat.get("Avg trade (net excess)", np.nan)),
        row(f"Strategy, net of {net_2x.cost_bps:g} bps/side (2× costs)", strat2, strat2["Turnover (ann.)"], n, strat2.get("Hit rate", np.nan), strat2.get("Avg trade (net excess)", np.nan)),
        row("Strategy, before costs", gross, strat["Turnover (ann.)"], n, (t["excess"] > 0).mean() if n else np.nan, gross.get("Avg trade (gross excess)", np.nan)),
        row("Buy & hold", bh, 0.0, np.nan, np.nan, np.nan),
    ]


def log_oos_evaluation(row: dict) -> None:
    OOS_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OOS_LOG, mode="a", header=not OOS_LOG.exists(), index=False)


SPX_NAME = "S&P 500 (SPY, total return)"
SPX_LINE = dict(width=1.3, color="#16a34a")


def equity_chart(
    res_all: BacktestResult, spx: pd.Series | None, is_end: pd.Timestamp, oos_start: pd.Timestamp, revealed: bool, ticker: str, cost_bps: float
) -> tuple[go.Figure, int | None]:
    """Equity curve whose holdout is an amber window at the right; while locked, clicking the window reveals it.

    Returns the figure and, when locked, the index of the invisible click-target trace covering the window.
    """
    full_d = res_all.daily
    end = full_d.index[-1]
    d = full_d if revealed else full_d.loc[:is_end]
    series = {
        f"Month-end strategy (net of {cost_bps:g} bps/side)": (growth(d["strategy_excess"]), dict(width=2.5, color="#2563eb")),
        "Month-end strategy (before costs)": (growth(d["strategy_gross_excess"]), dict(width=1.5, color="#2563eb", dash="dot")),
        f"Buy & hold {ticker}": (growth(d["buy_hold_excess"]), dict(width=1.5, color="#9ca3af")),
        "Rest of month only": (growth(d["rest_of_month_excess"]), dict(width=1.5, color="#f97316", dash="dash")),
    }
    if spx is not None:
        series[SPX_NAME] = (growth(spx.loc[d.index]), SPX_LINE)
    fig = go.Figure()
    for name, (y, line) in series.items():
        fig.add_trace(go.Scatter(x=d.index, y=y, name=name, line=line))

    oos_years = (end - oos_start).days / 365.25
    mid = oos_start + (end - oos_start) / 2
    fig.add_vrect(x0=oos_start, x1=end, fillcolor="rgba(255, 184, 77, 0.10)" if revealed else "rgba(255, 184, 77, 0.22)", line_width=0, layer="below")
    fig.add_vline(x=oos_start, line=dict(color="#f59e0b", width=1.5, dash="dash"))
    amber = dict(color="#b45309", family="monospace")
    fig.add_annotation(x=oos_start, y=1, xref="x", yref="paper", text="IN-SAMPLE ", showarrow=False, xanchor="right", yanchor="top", font=dict(size=10, color="#64748b", family="monospace"))

    target = None
    if revealed:
        fig.add_annotation(x=oos_start, y=1, xref="x", yref="paper", text=" OUT-OF-SAMPLE · EVALUATED", showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, **amber))
    else:
        # Scale the y-axis on in-sample data only, so the axis range leaks nothing about the holdout.
        values = np.concatenate([y.to_numpy() for y, _ in series.values()])
        lo, hi = np.log10(values.min()), np.log10(values.max())
        pad = 0.06 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad])
        fig.update_xaxes(range=[full_d.index[0], end])
        # Short stacked lines, because the holdout can be under a tenth of the chart's width.
        fig.add_annotation(x=mid, y=0.66, xref="x", yref="paper", text="<b>OUT-OF-<br>SAMPLE</b><br>LOCKED", showarrow=False, font=dict(size=10, **amber))
        fig.add_annotation(
            x=mid, y=0.52, xref="x", yref="paper", text=f"LAST {oos_years:.1f} YRS<br>{len(full_d.loc[oos_start:])} BARS",
            showarrow=False, font=dict(size=9, color="#92400e", family="monospace"),
        )
        fig.add_annotation(
            x=mid, y=0.38, xref="x", yref="paper", text="<b>EVALUATE<br>ONCE ▸</b>", showarrow=False,
            font=dict(size=10, color="#78350f", family="monospace"), bgcolor="rgba(255, 184, 77, 0.55)", bordercolor="#f59e0b", borderwidth=1.5, borderpad=5,
        )
        # Plotly can't make shapes clickable, so a grid of transparent points covers the window and catches the click.
        gx, gy = np.meshgrid(pd.date_range(oos_start, end, periods=40), np.logspace(lo - pad, hi + pad, 24))
        fig.add_trace(
            go.Scatter(
                x=gx.ravel(), y=gy.ravel(), mode="markers", name="holdout", showlegend=False,
                marker=dict(size=22, color="rgba(0, 0, 0, 0)"),
                hovertemplate="Click to evaluate the out-of-sample period once<extra></extra>",
            )
        )
        target = len(fig.data) - 1

    title = "Growth of $1 in excess of T-bills (log scale)"
    fig.update_layout(title=title, yaxis_type="log", height=460, legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10), hovermode="closest")
    return fig, target


def reveal(t: str) -> None:
    st.session_state.oos_revealed.add(t)
    st.session_state.chart_nonce += 1


def relock(t: str) -> None:
    st.session_state.oos_revealed.discard(t)
    st.session_state.oos_first_config.pop(t, None)
    st.session_state.chart_nonce += 1


if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "oos_revealed" not in st.session_state:
    st.session_state.oos_revealed = set()
if "oos_logged" not in st.session_state:
    st.session_state.oos_logged = set()
if "oos_first_config" not in st.session_state:
    st.session_state.oos_first_config = {}
if "chart_nonce" not in st.session_state:
    st.session_state.chart_nonce = 0

with st.sidebar:
    st.header("Strategy settings")
    ticker = st.selectbox("Treasury ETF", list(TREASURY_ETFS), format_func=lambda t: f"{t} · {TREASURY_ETFS[t]}")
    entry = st.slider("Enter N trading days before month-end", 1, 10, PRE_REGISTERED.entry)
    exit_ = st.slider("Exit (days relative to month-end close)", -2, 3, PRE_REGISTERED.exit, help="0 = last trading day of the month, +1 = first trading day of the next month")
    cost_bps = st.slider("Trading cost per side (bps)", 0.0, 10.0, 2.0, 0.5, help="Spread + commission + slippage. TLT's spread is about 1 bp.")
    if st.button("Re-download latest data", width="stretch"):
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

with st.sidebar:
    min_d, max_d = full.index[0].date(), full.index[-1].date()
    date_range = st.slider(
        "Sample period", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM",
        help="The holdout is cut from the end of whatever sample you pick.",
    )

d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
span = full.loc[d0:d1]
if len(span) < 2 * 252:
    st.warning("Pick a sample of at least two years.")
    st.stop()
oos_start = holdout_start(span.index)
is_end = span.index[span.index < oos_start][-1]
data_end = span.index[-1]
df = span.loc[:is_end]
if len(df) < 252:
    st.warning("Pick a sample with at least one year of in-sample data.")
    st.stop()

window = Window(entry, exit_)
config = (entry, exit_, cost_bps, str(d0.date()), str(d1.date()))
revealed = ticker in st.session_state.oos_revealed
log_key = (ticker, *config)
peeks = sum(1 for k in st.session_state.oos_logged if k[0] == ticker) + int(revealed and log_key not in st.session_state.oos_logged)

sample_years = (data_end - span.index[0]).days / 365.25
cap_binds = sample_years * HOLDOUT_FRACTION > HOLDOUT_MAX_YEARS
with st.sidebar:
    st.header("Out-of-sample holdout")
    st.markdown(f"**{'2-year cap binds' if cap_binds else '20% binds'}**")
    if cap_binds:
        st.caption(f"20% of {sample_years:.1f} years would be {sample_years * HOLDOUT_FRACTION:.1f} years. The 2-year cap is shorter, so hold out 2 years.")
    else:
        st.caption(f"20% of {sample_years:.1f} years is {sample_years * HOLDOUT_FRACTION:.1f} years, shorter than 2 years, so hold out 20%.")
    st.markdown(
        f"In-sample: **{span.index[0]:%Y-%m-%d} to {is_end:%Y-%m-%d}**  \n"
        f"Holdout: **{oos_start:%Y-%m-%d} to {data_end:%Y-%m-%d}** ({(data_end - oos_start).days / 365.25:.1f} years)"
    )
    st.caption("The cut is moved back to a month start so no trade straddles it.")
    if not revealed:
        st.markdown("Status: **locked**. Click the amber window at the end of the equity chart to evaluate it once.")
    else:
        st.markdown(f"Status: **evaluated** · peeks this session: **{peeks}**")
        st.caption(f"Each evaluation is appended to `reports/{OOS_LOG.name}`. Report every look in the quant note.")
        st.button("Relock", width="stretch", on_click=relock, args=(ticker,))

if revealed:
    first = st.session_state.oos_first_config.setdefault(ticker, config)
    if config != first:
        st.error(
            "**TEST SET LEAKED.** You changed the settings after seeing the out-of-sample period. That is tuning on the test set, "
            "and those results now overstate what the strategy would do on data it hasn't seen. Judges cap the Performance score at 4 for this. "
            f"Peeks this session: **{peeks}**. In real life you can't un-see a result, so report every peek in your note."
        )
        st.button("Relock and start over", on_click=relock, args=(ticker,))
elif window != PRE_REGISTERED:
    st.info(
        f"You're exploring a different window from the pre-registered one (enter 3 days before, exit at month-end). "
        "Picking the best-looking window after seeing results inflates the Sharpe; check the heatmap below to see how robust it is."
    )

with st.spinner("Running backtest…"):
    # The schedule only needs the trading calendar, so run on the whole sample and cut by date afterwards.
    res_all = run_backtest(span, window, cost_bps)
    res_all_2x = run_backtest(span, window, 2 * cost_bps)
    res = slice_result(res_all, end=is_end)
    try:
        spy_close = get_spy(st.session_state.refresh)
    except Exception:  # the benchmark is reference only, so the page works without it
        spy_close = None
    spx = benchmark_excess(spy_close, res_all.daily)
    s = summary(res)
    perm = get_perm(df, entry, exit_)
    grid = get_grid(df, cost_bps)
    profile = day_of_month_profile(df)
    subs = subperiod_table(df, window, cost_bps)
    yearly = yearly_returns(res)

oos_res = oos_res_2x = None
if revealed:
    oos_res = slice_result(res_all, start=oos_start)
    oos_res_2x = slice_result(res_all_2x, start=oos_start)
    oos_sharpe = summary(oos_res)["Month-end strategy"]["Sharpe"]
    if log_key not in st.session_state.oos_logged:
        log_oos_evaluation(
            {
                "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "entry": entry,
                "exit": exit_,
                "cost_bps": cost_bps,
                "pre_registered": window == PRE_REGISTERED,
                "sample_start": str(span.index[0].date()),
                "oos_start": str(oos_start.date()),
                "oos_end": str(data_end.date()),
                "oos_sharpe_net": round(oos_sharpe, 4),
            }
        )
        st.session_state.oos_logged.add(log_key)

st.subheader(f"In-sample: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
strat, bh = s["Month-end strategy"], s["Buy & hold"]
c = st.columns(6)
c[0].metric("Sharpe (strategy)", f"{strat['Sharpe']:.2f}", f"{strat['Sharpe'] - bh['Sharpe']:+.2f} vs buy & hold")
c[1].metric("Excess return / yr", f"{strat['Excess return (ann.)']:.2%}", f"buy & hold {bh['Excess return (ann.)']:.2%}", delta_color="off", delta_arrow="off")
c[2].metric("Max drawdown", f"{strat['Max drawdown']:.1%}", f"buy & hold {bh['Max drawdown']:.1%}", delta_color="off", delta_arrow="off")
c[3].metric("Avg trade (net)", f"{strat.get('Avg trade (net excess)', np.nan):.2%}", f"{int(strat.get('Trades', 0))} trades", delta_color="off", delta_arrow="off")
c[4].metric("Hit rate", f"{strat.get('Hit rate', np.nan):.0%}", f"in market {strat['Time in market']:.0%} of days", delta_color="off", delta_arrow="off")
c[5].metric("Permutation p-value", f"{perm['p_value']:.4f}", "vs random windows", delta_color="off", delta_arrow="off")

tab_overview, tab_oos, tab_robust, tab_trades, tab_about = st.tabs(["Performance", "Out-of-sample", "Robustness", "Trades", "How it works"])

with tab_overview:
    fig, target = equity_chart(res_all, spx, is_end, oos_start, revealed, ticker, cost_bps)
    chart_key = f"equity_{ticker}_{st.session_state.chart_nonce}"
    if target is None:
        st.plotly_chart(fig, width="stretch", key=chart_key)
    else:
        event = st.plotly_chart(fig, width="stretch", key=chart_key, on_select="rerun", selection_mode="points")
        if any(p.get("curve_number") == target for p in event.selection.points):
            reveal(ticker)
            st.rerun()
    st.caption(
        "Dotted blue line: the same trades before transaction costs; the gap to the solid line is the cost drag. "
        "Dashed amber line: where the out-of-sample holdout starts. Green line: the S&P 500 (SPY with dividends) for reference. "
        "Click a legend entry to hide a line."
    )

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
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["Strategy excess"] * 100, name="Strategy", marker_color="#2563eb"))
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["Buy & hold excess"] * 100, name="Buy & hold", marker_color="#cbd5e1"))
        fig.update_layout(title="Calendar-year excess return (%)", barmode="group", height=380, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame(s).T
    pct_cols = ["CAGR", "Excess return (ann.)", "Volatility (ann.)", "Max drawdown", "Time in market", "Avg trade (net excess)", "Avg trade (gross excess)", "Hit rate"]
    fmt = {k: "{:.2%}" for k in pct_cols} | {"Sharpe": "{:.2f}", "t-stat (per trade)": "{:.2f}", "Trades": "{:.0f}", "Turnover (ann.)": "{:.1f}×"}
    st.dataframe(table.style.format(fmt, na_rep="–"), width="stretch")
    st.caption("In-sample only. Turnover is traded value per year as a multiple of capital (each entry and each exit counts once).")

with tab_oos:
    if not revealed:
        st.markdown(
            f"#### Locked: {oos_start:%Y-%m-%d} to {data_end:%Y-%m-%d}\n"
            f"The track holds out the most recent {HOLDOUT_FRACTION:.0%} of history or {HOLDOUT_MAX_YEARS:g} years, whichever is shorter. "
            f"For the selected {ticker} sample that is the last {(data_end - oos_start).days / 365.25:.1f} years. "
            "Develop and tune on the in-sample period, then evaluate the holdout **once**, at the end, by clicking the amber window "
            "on the Performance tab's equity chart. If you change the strategy after looking, it isn't out-of-sample anymore."
        )
    else:
        od = oos_res.daily
        n_oos_trades = len(oos_res.trades)
        st.markdown(f"#### Out-of-sample: {od.index[0]:%Y-%m-%d} to {od.index[-1]:%Y-%m-%d} ({n_oos_trades} trades)")
        rows = comparison_rows("In-sample", res, slice_result(res_all_2x, end=df.index[-1])) + comparison_rows("Out-of-sample", oos_res, oos_res_2x)
        comp = pd.DataFrame(rows)
        comp_fmt = {
            "Excess return (ann.)": "{:.2%}",
            "Volatility (ann.)": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max drawdown": "{:.1%}",
            "Turnover (×/yr)": "{:.1f}",
            "Trades": "{:.0f}",
            "Hit rate": "{:.0%}",
            "Avg trade": "{:.2%}",
        }
        for col, f in comp_fmt.items():
            comp[col] = comp[col].map(lambda v, f=f: "–" if pd.isna(v) else f.format(v))
        st.dataframe(comp, width="stretch", hide_index=True)

        is_sharpe = s["Month-end strategy"]["Sharpe"]
        oos_sharpe_net = summary(oos_res)["Month-end strategy"]["Sharpe"]
        # A 2-year Sharpe has a standard error of roughly 1/sqrt(years).
        se = 1 / np.sqrt(len(od) / 252)
        st.markdown(
            f"Net Sharpe **{is_sharpe:.2f}** in-sample against **{oos_sharpe_net:.2f}** out-of-sample. "
            f"Over {len(od) / 252:.1f} years, noise alone moves a Sharpe by about ±{se:.1f} (one standard error), "
            f"so the out-of-sample number is {'within' if abs(oos_sharpe_net - is_sharpe) <= se else 'outside'} one standard error of in-sample. "
            "Report it either way."
        )

        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=od.index, y=growth(od["strategy_excess"]), name=f"Strategy (net of {cost_bps:g} bps/side)", line=dict(width=2.5, color="#2563eb")))
            fig.add_trace(go.Scatter(x=od.index, y=growth(od["strategy_gross_excess"]), name="Strategy (before costs)", line=dict(width=1.5, color="#2563eb", dash="dot")))
            fig.add_trace(go.Scatter(x=od.index, y=growth(oos_res_2x.daily["strategy_excess"]), name=f"Strategy (net of {2 * cost_bps:g} bps/side)", line=dict(width=1.5, color="#7c3aed")))
            fig.add_trace(go.Scatter(x=od.index, y=growth(od["buy_hold_excess"]), name=f"Buy & hold {ticker}", line=dict(width=1.5, color="#9ca3af")))
            if spx is not None:
                fig.add_trace(go.Scatter(x=od.index, y=growth(spx.loc[od.index]), name=SPX_NAME, line=SPX_LINE))
            fig.update_layout(title="Out-of-sample growth of $1 in excess of T-bills", height=400, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with right:
            ot = oos_res.trades
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ot["exit_close"], y=ot["net_excess"] * 100, name="Net", marker_color=np.where(ot["net_excess"] > 0, "#2563eb", "#ef4444")))
            fig.add_trace(go.Scatter(x=ot["exit_close"], y=ot["excess"] * 100, name="Before costs", mode="markers", marker=dict(symbol="line-ew-open", size=14, color="#111827")))
            fig.update_layout(title="Out-of-sample trades: excess return (%)", height=400, legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

with tab_robust:
    st.subheader("Does it still work? Sub-period results")
    st.dataframe(
        subs.style.format({"Strategy Sharpe": "{:.2f}", "Buy & hold Sharpe": "{:.2f}", "Strategy excess (ann.)": "{:.2%}", "Avg trade (net)": "{:.2%}", "Hit rate": "{:.0%}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The original study (Hartley & Schwarz) used data through 2018, so 2019+ is out-of-sample relative to the paper. "
        f"Everything on this tab, including the heatmap and permutation test, uses in-sample data only (up to {df.index[-1]:%Y-%m-%d})."
    )

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
        st.plotly_chart(fig, width="stretch")
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
    t = res.trades.assign(sample="In-sample")
    if oos_res is not None:
        t = pd.concat([t, oos_res.trades.assign(sample="Out-of-sample")], ignore_index=True)
    st.markdown(f"{len(t)} trades. Returns are for the holding window; *excess* subtracts T-bill interest, *net* also subtracts costs.")
    st.dataframe(
        t.style.format({"gross": "{:.2%}", "excess": "{:.2%}", "net_excess": "{:.2%}", "entry_close": "{:%Y-%m-%d}", "exit_close": "{:%Y-%m-%d}"}),
        width="stretch",
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
- The most recent 20% of the selected sample or 2 years, whichever is shorter, is held out and hidden until you click the
  amber window on the equity chart.
- Costs are charged per side on the entry and exit closes. The dotted lines show the same trades before costs.
- Treasury futures (ZN, ZB) would be cheaper to trade, but Yahoo's continuous futures series isn't roll-adjusted and the
  quarterly rolls fall near month-end, which would contaminate the test. That's why this uses ETFs.
"""
    )
