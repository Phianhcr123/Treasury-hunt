"""Streamlit pieces shared by every strategy page.

Each strategy gets the same out-of-sample treatment: the most recent 20% of the selected sample
or 2 years (whichever is shorter) is hidden behind an amber window at the end of the equity
chart, clicking the window evaluates it once, every evaluation is logged, and changing the
settings afterwards raises a leak alert. The S&P 500 is drawn on every chart for reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest import HOLDOUT_FRACTION, HOLDOUT_MAX_YEARS, holdout_start
from .data import load_close

OOS_LOG = Path(__file__).resolve().parents[2] / "reports" / "oos_evaluations.csv"
SPX_NAME = "S&P 500 (SPY, total return)"
SPX_LINE = dict(width=1.3, color="#16a34a")
AMBER = dict(color="#b45309", family="monospace")


@dataclass(frozen=True)
class Holdout:
    key: str
    config: tuple
    sample_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    end: pd.Timestamp
    revealed: bool
    peeks: int

    @property
    def log_key(self) -> tuple:
        return (self.key, *self.config)

    @property
    def oos_years(self) -> float:
        return (self.end - self.oos_start).days / 365.25


@dataclass(frozen=True)
class Line:
    name: str
    growth: pd.Series
    style: dict


def _init_state() -> None:
    for name, default in [("oos_revealed", set()), ("oos_logged", set()), ("oos_first_config", {}), ("chart_nonce", 0)]:
        if name not in st.session_state:
            st.session_state[name] = default


def reveal(key: str) -> None:
    st.session_state.oos_revealed.add(key)
    st.session_state.chart_nonce += 1


def relock(key: str) -> None:
    st.session_state.oos_revealed.discard(key)
    st.session_state.oos_first_config.pop(key, None)
    st.session_state.chart_nonce += 1


def setup_holdout(key: str, index: pd.DatetimeIndex, config: tuple) -> Holdout:
    """Work out the holdout for the selected sample, draw its sidebar panel and any leak alert."""
    _init_state()
    oos_start = holdout_start(index)
    is_end = index[index < oos_start][-1]
    revealed = key in st.session_state.oos_revealed
    log_key = (key, *config)
    peeks = sum(1 for k in st.session_state.oos_logged if k[0] == key) + int(revealed and log_key not in st.session_state.oos_logged)
    h = Holdout(key, config, index[0], is_end, oos_start, index[-1], revealed, peeks)

    sample_years = (h.end - h.sample_start).days / 365.25
    cap_binds = sample_years * HOLDOUT_FRACTION > HOLDOUT_MAX_YEARS
    with st.sidebar:
        st.header("Out-of-sample holdout")
        st.markdown(f"**{'2-year cap binds' if cap_binds else '20% binds'}**")
        if cap_binds:
            st.caption(f"20% of {sample_years:.1f} years would be {sample_years * HOLDOUT_FRACTION:.1f} years. The 2-year cap is shorter, so hold out 2 years.")
        else:
            st.caption(f"20% of {sample_years:.1f} years is {sample_years * HOLDOUT_FRACTION:.1f} years, shorter than 2 years, so hold out 20%.")
        st.markdown(
            f"In-sample: **{h.sample_start:%Y-%m-%d} to {h.is_end:%Y-%m-%d}**  \n"
            f"Holdout: **{h.oos_start:%Y-%m-%d} to {h.end:%Y-%m-%d}** ({h.oos_years:.1f} years)"
        )
        st.caption("The cut is moved back to a month start. Each strategy has its own holdout and its own single evaluation.")
        if not revealed:
            st.markdown("Status: **locked**. Click the amber window at the end of the equity chart to evaluate it once.")
        else:
            st.markdown(f"Status: **evaluated** · peeks this session: **{peeks}**")
            st.caption(f"Each evaluation is appended to `reports/{OOS_LOG.name}`. Report every look in the quant note.")
            st.button("Relock", width="stretch", on_click=relock, args=(key,), key=f"relock_sidebar_{key}")

    if revealed:
        first = st.session_state.oos_first_config.setdefault(key, config)
        if config != first:
            st.error(
                "**TEST SET LEAKED.** You changed the settings after seeing the out-of-sample period. That is tuning on the test set, "
                "and those results now overstate what the strategy would do on data it hasn't seen. Judges cap the Performance score at 4 for this. "
                f"Peeks this session: **{peeks}**. In real life you can't un-see a result, so report every peek in your note."
            )
            st.button("Relock and start over", on_click=relock, args=(key,), key=f"relock_alert_{key}")
    return h


def log_evaluation(h: Holdout, oos_sharpe: float, config_text: str) -> None:
    """Append one row per distinct setting evaluated on the holdout."""
    if not h.revealed or h.log_key in st.session_state.oos_logged:
        return
    row = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": h.key,
        "settings": config_text,
        "sample_start": str(h.sample_start.date()),
        "oos_start": str(h.oos_start.date()),
        "oos_end": str(h.end.date()),
        "oos_sharpe_net": round(float(oos_sharpe), 4),
    }
    OOS_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OOS_LOG, mode="a", header=not OOS_LOG.exists(), index=False)
    st.session_state.oos_logged.add(h.log_key)


@st.cache_data(show_spinner=False)
def _spy_close() -> pd.Series:
    return load_close("SPY")


def spx_growth(index: pd.DatetimeIndex, rf: pd.Series | None = None) -> pd.Series | None:
    """Growth of $1 in the S&P 500 (SPY with dividends) over T-bills, sampled on `index`."""
    try:
        close = _spy_close()
    except Exception:  # the benchmark is reference only, so pages work without it
        return None
    ret = close.reindex(index, method="ffill").pct_change().fillna(0.0)
    if rf is not None:
        ret = ret - rf.reindex(index).fillna(0.0)
    return (1 + ret).cumprod()


def equity_chart(lines: list[Line], h: Holdout, title: str, height: int = 460) -> tuple[go.Figure, int | None]:
    """Growth chart ending in the holdout window: locked and clickable, or evaluated.

    Returns the figure and, while locked, the index of the invisible trace that catches clicks on the window.
    """
    shown = [Line(ln.name, ln.growth if h.revealed else ln.growth.loc[: h.is_end], ln.style) for ln in lines]
    fig = go.Figure()
    for ln in shown:
        fig.add_trace(go.Scatter(x=ln.growth.index, y=ln.growth, name=ln.name, line=ln.style))

    fig.add_vrect(x0=h.oos_start, x1=h.end, fillcolor="rgba(255, 184, 77, 0.10)" if h.revealed else "rgba(255, 184, 77, 0.22)", line_width=0, layer="below")
    fig.add_vline(x=h.oos_start, line=dict(color="#f59e0b", width=1.5, dash="dash"))
    fig.add_annotation(x=h.oos_start, y=1, xref="x", yref="paper", text="IN-SAMPLE ", showarrow=False, xanchor="right", yanchor="top", font=dict(size=10, color="#64748b", family="monospace"))

    target = None
    if h.revealed:
        fig.add_annotation(x=h.oos_start, y=1, xref="x", yref="paper", text=" OUT-OF-SAMPLE · EVALUATED", showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, **AMBER))
    else:
        # Scale the y-axis on in-sample data only, so the axis range leaks nothing about the holdout.
        values = np.concatenate([ln.growth.dropna().to_numpy() for ln in shown if len(ln.growth.dropna())])
        values = values[values > 0]
        lo, hi = np.log10(values.min()), np.log10(values.max())
        pad = max(0.06 * (hi - lo), 0.01)
        fig.update_yaxes(range=[lo - pad, hi + pad])
        fig.update_xaxes(range=[min(ln.growth.index[0] for ln in shown if len(ln.growth)), h.end])
        mid = h.oos_start + (h.end - h.oos_start) / 2
        # Short stacked lines, because the holdout can be under a tenth of the chart's width.
        fig.add_annotation(x=mid, y=0.66, xref="x", yref="paper", text="<b>OUT-OF-<br>SAMPLE</b><br>LOCKED", showarrow=False, font=dict(size=10, **AMBER))
        fig.add_annotation(x=mid, y=0.52, xref="x", yref="paper", text=f"LAST {h.oos_years:.1f} YRS", showarrow=False, font=dict(size=9, color="#92400e", family="monospace"))
        fig.add_annotation(
            x=mid, y=0.38, xref="x", yref="paper", text="<b>EVALUATE<br>ONCE ▸</b>", showarrow=False,
            font=dict(size=10, color="#78350f", family="monospace"), bgcolor="rgba(255, 184, 77, 0.55)", bordercolor="#f59e0b", borderwidth=1.5, borderpad=5,
        )
        # Plotly can't make shapes clickable, so a grid of transparent points covers the window and catches the click.
        gx, gy = np.meshgrid(pd.date_range(h.oos_start, h.end, periods=40), np.logspace(lo - pad, hi + pad, 24))
        fig.add_trace(
            go.Scatter(
                x=gx.ravel(), y=gy.ravel(), mode="markers", name="holdout", showlegend=False,
                marker=dict(size=22, color="rgba(0, 0, 0, 0)"),
                hovertemplate="Click to evaluate the out-of-sample period once<extra></extra>",
            )
        )
        target = len(fig.data) - 1

    fig.update_layout(title=title, yaxis_type="log", height=height, legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10), hovermode="closest")
    return fig, target


def show_equity_chart(fig: go.Figure, target: int | None, h: Holdout) -> None:
    """Render the chart; a click on the locked window evaluates the holdout."""
    key = f"equity_{h.key}_{st.session_state.chart_nonce}"
    if target is None:
        st.plotly_chart(fig, width="stretch", key=key)
        return
    event = st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points")
    if any(p.get("curve_number") == target for p in event.selection.points):
        reveal(h.key)
        st.rerun()


def locked_message(h: Holdout) -> None:
    st.markdown(
        f"#### Locked: {h.oos_start:%Y-%m-%d} to {h.end:%Y-%m-%d}\n"
        f"The track holds out the most recent {HOLDOUT_FRACTION:.0%} of the sample or {HOLDOUT_MAX_YEARS:g} years, whichever is shorter "
        f"({h.oos_years:.1f} years here). Develop and tune on the in-sample period, then evaluate the holdout **once**, at the end, by "
        "clicking the amber window on the equity chart. If you change the strategy after looking, it isn't out-of-sample anymore."
    )


def growth(excess: pd.Series) -> pd.Series:
    return (1 + excess.fillna(0.0)).cumprod()


def ann_stats(total: pd.Series, excess: pd.Series, periods_per_year: float = 252) -> dict[str, float]:
    """Annualized return, volatility, Sharpe and drawdown for one slice of returns."""
    if len(excess) < 2:
        return {"Excess return (ann.)": np.nan, "Volatility (ann.)": np.nan, "Sharpe": np.nan, "Max drawdown": np.nan}
    vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
    eq = (1 + total).cumprod()
    return {
        "Excess return (ann.)": excess.mean() * periods_per_year,
        "Volatility (ann.)": vol,
        "Sharpe": excess.mean() * periods_per_year / vol if vol > 0 else np.nan,
        "Max drawdown": float((eq / eq.cummax() - 1).min()),
    }


COMPARISON_FMT = {
    "Excess return (ann.)": "{:.2%}",
    "Volatility (ann.)": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Max drawdown": "{:.1%}",
    "Turnover (×/yr)": "{:.1f}",
}


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Format rows of {Sample, Series, metrics...} for display, with dashes for missing values."""
    df = pd.DataFrame(rows)
    for col, f in COMPARISON_FMT.items():
        if col in df:
            df[col] = df[col].map(lambda v, f=f: "–" if pd.isna(v) else f.format(v))
    return df
