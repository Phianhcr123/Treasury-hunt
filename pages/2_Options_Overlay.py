import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eom_treasury_rally.backtest import Window
from eom_treasury_rally.data import load_dataset
from eom_treasury_rally.holdout_ui import (
    SPX_LINE,
    SPX_NAME,
    Line,
    ann_stats,
    comparison_table,
    equity_chart,
    locked_message,
    log_evaluation,
    setup_holdout,
    show_equity_chart,
    spx_growth,
)
from eom_treasury_rally.options_backtest import OptionSpec, iv_proxy_check, options_summary, run_options_backtest
from eom_treasury_rally.options_cli import contract_grid
from eom_treasury_rally.pricing import black_scholes, crr_price, drift_tree
from eom_treasury_rally.scanner import scan_calls

st.set_page_config(page_title="Options Overlay · Month-End Treasury Rally", page_icon="🌳", layout="wide")

st.title("Options overlay: pricing the month-end drift with a binomial tree")
st.markdown(
    "Treasuries drift up about **11 bps a day** in the last three trading days of the month, while option prices use a flat "
    "volatility. This page builds a CRR binomial tree that embeds that temporary drift, values TLT calls with it, and "
    "checks whether calls are a better way to hold the month-end trade than the ETF itself."
)
with st.expander("Read this first: why the drift tree makes every call look cheap", expanded=True):
    st.markdown(
        """
- Option prices don't depend on the underlying's expected return. A market maker who sells you a call hedges with
  delta shares of TLT, so the month-end drift **helps their hedge exactly as much as it helps your call**. The fair price uses
  the risk-free rate, not the drift.
- So a tree with an upward drift values *every* call above its market price, and every put below it. The gap is roughly
  delta × drift. That isn't a contract-specific mispricing; it's the same month-end bet with leverage.
- What the options market *could* get wrong is **volatility**: for example, if TLT were more volatile at month-end than implied vol
  assumes. It isn't. Month-end days are slightly *calmer* (13.2% against 14.4% annualized).
- So the useful question is: **is a call a better vehicle for the drift than the ETF, after spreads?** The drift tree answers that
  by giving each contract's expected return and risk over the window.
"""
    )

tab_pricer, tab_hist, tab_scan = st.tabs(["Pricing engine", "Historical test (2002–2026)", "Live TLT call scanner"])

with tab_pricer:
    c1, c2, c3 = st.columns(3)
    S = c1.number_input("TLT price", value=77.5, step=0.5)
    K = c1.number_input("Strike", value=77.5, step=0.5)
    days = c1.slider("Days to expiry", 5, 120, 30)
    sigma = c2.slider("Implied volatility (flat)", 0.05, 0.40, 0.16, 0.01)
    r = c2.slider("Risk-free rate", 0.0, 0.08, 0.04, 0.0025, format="%.4f")
    drift_bps = c2.slider("Month-end excess drift (bps/day)", 0.0, 30.0, 11.0, 0.5)
    hold = c3.slider("Holding window (trading days)", 1, 10, 3)
    div_amt = c3.number_input("Next dividend ($/share)", value=0.31, step=0.01)
    div_days = c3.slider("Days until ex-dividend", 1, 40, 4, help="TLT goes ex-dividend on the first trading day of each month, right after the window.")
    T = days / 365
    divs = ((div_days / 365, div_amt),) if div_amt > 0 and div_days < days else ()

    if hold / 252 >= T:
        st.error("The holding window must end before the option expires.")
    else:
        res = drift_tree(S, K, T, r, sigma, hold / 252, drift_bps / 1e4 * 252, dividends=divs)
        bs = black_scholes(S, K, T, r, sigma, dividends=divs)
        eu = crr_price(S, K, T, r, sigma, american=False, dividends=divs)
        m = st.columns(5)
        m[0].metric("Black-Scholes (European)", f"${bs:.3f}")
        m[1].metric("CRR European", f"${eu:.3f}")
        m[2].metric("CRR American (fair price)", f"${res.price:.3f}", f"early-exercise premium ${res.price - eu:.3f}", delta_color="off", delta_arrow="off")
        m[3].metric(f"Drift tree: expected value after {hold} days", f"${res.expected_exit_value:.3f}")
        m[4].metric("Expected return (before costs)", f"{res.expected_return:+.1%}", f"± {res.return_sd:.0%} (1 sd)", delta_color="off", delta_arrow="off")
        st.caption(
            f"Delta {res.delta:.2f}, elasticity {res.elasticity:.1f}x (a 1% TLT move changes the call {res.elasticity:.1f}%). "
            f"Chance of profit {res.prob_profit:.0%}. Holding TLT itself over the same window: expected "
            f"{drift_bps * hold:.0f} bps ± {sigma * np.sqrt(hold / 252):.2%}."
        )

        strikes = np.round(np.arange(S * 0.94, S * 1.06 + 0.01, 0.5) * 2) / 2
        rows = []
        for k in strikes:
            rk = drift_tree(S, float(k), T, r, sigma, hold / 252, drift_bps / 1e4 * 252, dividends=divs, value_steps=100)
            spread_cost = 0.01 + 0.0065
            net = (rk.expected_exit_value - spread_cost) / (rk.price + spread_cost) - 1
            rows.append({"strike": k, "gross": rk.expected_return, "net": net, "edge_per_risk": net / rk.return_sd})
        sweep = pd.DataFrame(rows)
        etf_epr = drift_bps * hold / 1e4 / (sigma * np.sqrt(hold / 252))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["strike"], y=sweep["edge_per_risk"], name="Call: expected return per unit of risk (after $0.01 half-spread)", mode="lines+markers"))
        fig.add_hline(y=etf_epr, line_dash="dash", annotation_text="Just hold TLT", annotation_position="top left")
        fig.add_vline(x=S, line_dash="dot", line_color="gray")
        fig.update_layout(title="Every call has positive expected return under the drift, but none beats holding TLT per unit of risk", xaxis_title="Strike", yaxis_title="Expected return / standard deviation", height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner=False)
def hist_backtest(dte: int, moneyness: float, half_spread: float, drift: float, cost_mult: float = 1.0) -> pd.DataFrame:
    spec = OptionSpec(dte=dte, moneyness=moneyness, half_spread=half_spread * cost_mult, commission=OptionSpec().commission * cost_mult)
    return run_options_backtest(Window(3, 0), spec, excess_drift_bps=drift, etf_cost_bps=2.0 * cost_mult)


@st.cache_data(show_spinner=False)
def hist_grid(half_spread: float, start: str, end: str) -> pd.DataFrame:
    return contract_grid(Window(3, 0), half_spread, start=start, end=end)


@st.cache_data(show_spinner=False)
def tlt_calendar() -> pd.DataFrame:
    return load_dataset("TLT")[["rf"]]


def growth_at_exits(trades: pd.DataFrame, col: str) -> pd.Series:
    excess = trades[col] if col == "delta_hedged_pnl" else trades[col] - trades["rf_window"]
    return pd.Series((1 + excess).cumprod().to_numpy(), index=pd.to_datetime(trades["exit"]))


def comparison_rows(sample: str, t: pd.DataFrame, t_2x: pd.DataFrame) -> list[dict]:
    def row(series: str, frame: pd.DataFrame, col: str, turnover: float) -> dict:
        excess = frame[col] - frame["rf_window"]
        return {"Sample": sample, "Series": series, **ann_stats(excess, excess, periods_per_year=12), "Turnover (×/yr)": turnover, "Trades": float(len(frame)), "Hit rate": (excess > 0).mean()}

    return [
        row("Call overlay, net", t, "call_overlay_ret", np.nan),
        row("Call overlay, 2× costs", t_2x, "call_overlay_ret", np.nan),
        row("Call overlay, before costs", t, "call_overlay_ret_gross", np.nan),
        row("ETF in window, net of 2 bps/side", t, "etf_ret", 24.0),
        row("ETF in window, 2× costs", t_2x, "etf_ret", 24.0),
        row("ETF in window, before costs", t, "etf_ret_gross", 24.0),
    ]


@st.cache_data(show_spinner=False)
def proxy_check(end: str) -> dict:
    return iv_proxy_check(end=end)


with tab_hist:
    st.markdown(
        "Free historical TLT option quotes don't exist, so each month's call is priced the way the hypothesis assumes the market prices it: "
        "an **American CRR tree at a flat implied volatility**, with real ex-dividend dates. The volatility is the MOVE bond-volatility index "
        "converted to TLT terms (MOVE × duration 16), which matches Cboe's TLT implied-vol index. You buy at the close 3 days before month-end "
        "and sell at the month-end close, paying the spread both ways."
    )
    c1, c2, c3 = st.columns(3)
    dte = c1.select_slider("Days to expiry at entry", [7, 14, 30, 45, 60, 90], 30)
    mny = c2.select_slider("Strike / spot", [0.96, 0.98, 1.0, 1.02, 1.04], 1.0)
    hs = c3.select_slider("Half-spread ($/share)", [0.005, 0.01, 0.015, 0.02, 0.03], 0.01)
    try:
        with st.spinner("Pricing 287 months of options with the binomial tree…"):
            trades_all = hist_backtest(dte, mny, hs, 11.0)
            calendar = tlt_calendar()
    except Exception as e:
        st.error(f"Couldn't run the options backtest ({type(e).__name__}: {e}).")
        st.stop()

    first, last = pd.Timestamp(trades_all["entry"].iloc[0]), pd.Timestamp(trades_all["exit"].iloc[-1])
    date_range = st.slider(
        "Sample period", min_value=first.date(), max_value=last.date(), value=(first.date(), last.date()), format="YYYY-MM",
        help="The holdout is cut from the end of whatever sample you pick.",
    )
    span = calendar.loc[str(date_range[0]) : str(date_range[1])]
    if len(span) < 2 * 252:
        st.warning("Pick a sample of at least two years.")
        st.stop()
    h = setup_holdout("Options overlay · TLT calls", span.index, (dte, mny, hs, str(date_range[0]), str(date_range[1])))
    entry_dt, exit_dt = pd.to_datetime(trades_all["entry"]), pd.to_datetime(trades_all["exit"])
    in_span = (entry_dt >= span.index[0]) & (exit_dt <= span.index[-1])
    trades = trades_all[in_span & (exit_dt <= h.is_end)].reset_index(drop=True)
    trades_span = trades_all[in_span].reset_index(drop=True)
    trades_oos = trades_all[in_span & (entry_dt >= h.oos_start)].reset_index(drop=True)
    if len(trades) < 12:
        st.warning("Pick a sample with at least a year of in-sample trades.")
        st.stop()
    summ = options_summary(trades)
    if h.revealed:
        log_evaluation(
            h,
            options_summary(trades_oos).loc["Delta-matched call overlay (excess)", "Sharpe (ann.)"],
            f"{dte}-day calls, strike {mny:.2f}× spot, ${hs:.3f} half-spread, enter -3, exit +0",
        )
    st.markdown(f"#### In-sample: {trades['entry'].iloc[0]:%Y-%m-%d} to {trades['exit'].iloc[-1]:%Y-%m-%d} ({len(trades)} trades)")

    etf = summ.loc["ETF in window (excess)"]
    call = summ.loc["Call, % of premium (excess)"]
    over = summ.loc["Delta-matched call overlay (excess)"]
    hedged = summ.loc["Delta-hedged call (% of spot)"]
    m = st.columns(4)
    m[0].metric("ETF in window: Sharpe", f"{etf['Sharpe (ann.)']:.2f}", f"{etf['Avg per trade']:.2%} per trade", delta_color="off", delta_arrow="off")
    m[1].metric("Call: avg return on premium", f"{call['Avg per trade']:+.1%}", f"Sharpe {call['Sharpe (ann.)']:.2f}", delta_color="off", delta_arrow="off")
    m[2].metric("Call sized like the ETF: Sharpe", f"{over['Sharpe (ann.)']:.2f}", f"{over['Sharpe (ann.)'] - etf['Sharpe (ann.)']:+.2f} vs ETF")
    m[3].metric("Delta-hedged call: avg P&L", f"{hedged['Avg per trade'] * 1e4:+.1f} bps", "≈ 0: no mispricing left", delta_color="off", delta_arrow="off")

    lines = [
        Line("ETF in window (net)", growth_at_exits(trades_span, "etf_ret"), dict(width=2, color="#2563eb")),
        Line("ETF in window (before costs)", growth_at_exits(trades_span, "etf_ret_gross"), dict(width=1.3, color="#2563eb", dash="dot")),
        Line("Calls sized to the same exposure (net)", growth_at_exits(trades_span, "call_overlay_ret"), dict(width=2, color="#dc2626")),
        Line("Calls sized to the same exposure (before costs)", growth_at_exits(trades_span, "call_overlay_ret_gross"), dict(width=1.3, color="#dc2626", dash="dot")),
        Line("Delta-hedged calls (the 'mispricing')", growth_at_exits(trades_span, "delta_hedged_pnl"), dict(width=1.5, color="#7c3aed")),
    ]
    spx = spx_growth(span.index, span["rf"])
    if spx is not None:
        s = spx.loc[pd.to_datetime(trades_span["exit"])]
        lines.append(Line(f"{SPX_NAME}, held all month", s / spx.loc[: pd.Timestamp(trades_span["entry"].iloc[0])].iloc[-1], SPX_LINE))
    fig, target = equity_chart(lines, h, "Growth of $1 in excess of T-bills, month-end trades only", height=430)
    show_equity_chart(fig, target, h)
    st.caption(
        "Dotted lines: the same trades at model mid prices with no spread, commission or ETF cost. "
        "The delta-hedged line is already before costs. Dashed amber line: where the out-of-sample holdout starts. "
        "Green line: the S&P 500 (SPY with dividends), held every day rather than only in the window, for reference."
    )

    if not h.revealed:
        with st.expander("Out-of-sample holdout"):
            locked_message(h)
    else:
        trades_2x = hist_backtest(dte, mny, hs, 11.0, cost_mult=2.0)
        t2_entry, t2_exit = pd.to_datetime(trades_2x["entry"]), pd.to_datetime(trades_2x["exit"])
        t2_span = (t2_entry >= span.index[0]) & (t2_exit <= span.index[-1])
        st.markdown(f"#### Out-of-sample: {trades_oos['entry'].iloc[0]:%Y-%m-%d} to {trades_oos['exit'].iloc[-1]:%Y-%m-%d} ({len(trades_oos)} trades)")
        rows = comparison_rows("In-sample", trades, trades_2x[t2_span & (t2_exit <= h.is_end)]) + comparison_rows("Out-of-sample", trades_oos, trades_2x[t2_span & (t2_entry >= h.oos_start)])
        st.dataframe(comparison_table(rows), width="stretch", hide_index=True)
        is_sharpe = over["Sharpe (ann.)"]
        oos_sharpe = options_summary(trades_oos).loc["Delta-matched call overlay (excess)", "Sharpe (ann.)"]
        se = 1 / np.sqrt(len(trades_oos) / 12)
        st.markdown(
            f"Call overlay net Sharpe **{is_sharpe:.2f}** in-sample against **{oos_sharpe:.2f}** out-of-sample. "
            f"Over {len(trades_oos) / 12:.1f} years, noise alone moves a Sharpe by about ±{se:.1f} (one standard error). Report it either way. "
            "2× costs doubles the option half-spread, the commission and the ETF cost. ETF turnover is 24× a year (12 round trips)."
        )

    left, right = st.columns([3, 2])
    with left:
        fmt = {c: "{:.2%}" for c in ["Avg per trade", "Std per trade", "Hit rate", "Worst trade"]} | {"Sharpe (ann.)": "{:.2f}", "t-stat": "{:.2f}", "Trades": "{:.0f}"}
        st.dataframe(summ.style.format(fmt), width="stretch")
        chk = proxy_check(str(h.is_end.date()))
        st.caption(
            f"In-sample only. Model check: the drift tree predicted {trades['model_expected_ret_net'].mean():.1%} per call trade after costs; the backtest realized "
            f"{trades['call_ret'].mean():.1%}. Implied-vol proxy averaged {chk['avg_iv_proxy']:.1%} against {chk['avg_next_21d_realized']:.1%} realized."
        )
    with right:
        with st.spinner("Testing 12 expiry/strike combinations…"):
            grid = hist_grid(hs, str(span.index[0].date()), str(h.is_end.date()))
        pivot = grid.pivot(index="Days to expiry", columns="Strike / spot", values="Delta-matched Sharpe")
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], mode="lines+markers", name=f"strike {col:.2f}× spot"))
        fig.add_hline(y=grid["ETF Sharpe"].iloc[0], line_dash="dash", annotation_text="ETF")
        fig.update_layout(title="Sharpe by contract (same exposure as the ETF)", xaxis_title="Days to expiry", yaxis_title="Sharpe", height=360, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    st.info(
        "Takeaway: calls do capture the drift (about +9% of premium per trade at the money), but never with a better Sharpe than the ETF. "
        "The more stock-like the call (in the money, longer-dated), the closer it gets; short-dated out-of-the-money calls lose most of the edge to the spread. "
        "Calls are worth using only for leverage or capped downside, not as a source of extra edge."
    )

with tab_scan:
    st.markdown(
        "Pulls today's TLT call chain, backs out each contract's implied volatility with the American CRR tree, then runs the drift tree over the "
        "**next** month-end window. Spot and implied vol are assumed unchanged until entry, and the quoted spread is paid both ways."
    )
    c1, c2 = st.columns(2)
    drift_live = c1.slider("Assumed drift (bps/day)", 0.0, 20.0, 11.0, 0.5, key="scan_drift", help="11 bps is the full-sample average; 2019+ is about 7.")
    rf_live = c2.slider("Risk-free rate", 0.0, 0.08, 0.04, 0.0025, format="%.4f", key="scan_rf")
    if st.button("Scan TLT option chain", type="primary"):
        try:
            with st.spinner("Downloading the chain and running the tree for each contract…"):
                st.session_state.scan = scan_calls(excess_drift_bps=drift_live, risk_free=rf_live)
        except Exception as e:
            st.session_state.pop("scan", None)
            st.error(f"Couldn't scan the option chain ({type(e).__name__}: {e}). Yahoo may be rate-limiting; try again shortly.")
    if "scan" not in st.session_state:
        st.caption("Press the button to scan. It takes a few seconds.")
    else:
        scan, meta = st.session_state.scan
        if scan.empty:
            st.warning("No contracts passed the filters (expiry after the window, strike within 5% of spot).")
        else:
            m = st.columns(4)
            m[0].metric("TLT", f"${meta['spot']:.2f}", f"as of {meta['today']}", delta_color="off", delta_arrow="off")
            m[1].metric("Next window", f"{meta['window_entry']:%b %d} → {meta['window_exit']:%b %d}", "enter close → exit close", delta_color="off", delta_arrow="off")
            m[2].metric("Best call: return per unit of risk", f"{scan['edge_per_risk'].iloc[0]:.3f}")
            m[3].metric("Holding TLT: return per unit of risk", f"{meta['etf_edge_per_risk']:.3f}")
            if (scan["quote"] != "live").any():
                st.warning("Some quotes are last-trade prices because the market is closed; their spreads use a $0.02 placeholder.")
            show = scan[["contract", "expiry", "strike", "bid", "ask", "market_iv", "delta", "elasticity", "fair_at_entry", "expected_value_at_exit", "expected_ret_net", "ret_sd", "edge_per_risk", "cost_bps_of_spot_per_delta", "volume", "open_interest"]]
            st.dataframe(
                show.style.format(
                    {"strike": "{:.1f}", "bid": "{:.2f}", "ask": "{:.2f}", "market_iv": "{:.1%}", "delta": "{:.2f}", "elasticity": "{:.1f}x", "fair_at_entry": "${:.3f}", "expected_value_at_exit": "${:.3f}", "expected_ret_net": "{:+.1%}", "ret_sd": "{:.0%}", "edge_per_risk": "{:.3f}", "cost_bps_of_spot_per_delta": "{:.1f}"}
                ).background_gradient(subset=["edge_per_risk"], cmap="Greens"),
                width="stretch",
                hide_index=True,
                height=460,
            )
            st.caption(
                "fair_at_entry = model price on the entry date at today's implied vol. expected_value_at_exit = drift-tree expectation at the month-end close. "
                "cost_bps_of_spot_per_delta = round-trip spread and commission per unit of exposure, in bps of TLT; compare with the "
                f"{meta['expected_window_drift_bps']:.0f} bps the window is expected to earn."
            )
