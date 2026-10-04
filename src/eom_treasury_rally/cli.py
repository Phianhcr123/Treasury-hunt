from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import (  # noqa: E402
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
from .data import load_dataset  # noqa: E402

PCT_KEYS = {"CAGR", "Excess return (ann.)", "Volatility (ann.)", "Max drawdown", "Time in market", "Avg trade (net excess)", "Avg trade (gross excess)", "Hit rate"}


def _fmt(key: str, v: float) -> str:
    if pd.isna(v):
        return "–"
    if key in PCT_KEYS:
        return f"{v:.2%}"
    if key == "Trades":
        return f"{int(v)}"
    return f"{v:.2f}"


def _summary_table(s: dict[str, dict[str, float]]) -> pd.DataFrame:
    keys = list(dict.fromkeys(k for block in s.values() for k in block))
    return pd.DataFrame({name: [_fmt(k, block.get(k, np.nan)) for k in keys] for name, block in s.items()}, index=keys)


def _charts(out: Path, ticker: str, res, profile: pd.DataFrame, grid: pd.DataFrame, yearly: pd.DataFrame, curve=None, oos_start=None) -> list[Path]:
    paths = []
    w = res.window

    fig, ax = plt.subplots(figsize=(10, 5))
    d = (curve or res).daily
    ax.plot((1 + d["strategy_excess"]).cumprod(), label="Month-end window (excess, net of costs)", lw=1.8, color="C0")
    ax.plot((1 + d["strategy_gross_excess"]).cumprod(), label="Month-end window (excess, before costs)", lw=1.2, ls=":", color="C0")
    ax.plot((1 + d["buy_hold_excess"]).cumprod(), label=f"Buy & hold {ticker} (excess)", lw=1.2, alpha=0.8, color="C7")
    ax.plot((1 + d["rest_of_month_excess"]).cumprod(), label="Rest of month only (excess)", lw=1.2, alpha=0.8, color="C1", ls="--")
    if oos_start is not None:
        ax.axvspan(oos_start, d.index[-1], color="#fde68a", alpha=0.4, lw=0, label="Out-of-sample")
    ax.set_yscale("log")
    ax.set_title(f"{ticker}: growth of $1 in excess of T-bills (entry -{w.entry}, exit {w.exit:+d}, {res.cost_bps:g} bps/side)")
    ax.legend()
    ax.grid(alpha=0.3)
    paths.append(out / "equity_curve.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    held = [o for o in profile.index if (-w.entry < o <= min(w.exit, 0)) or (0 < o <= w.exit)]
    colors = ["#2563eb" if o in held else "#9ca3af" for o in profile.index]
    labels = [f"{o}" if o <= 0 else f"+{o}" for o in profile.index]
    ax.bar(labels, profile["mean"] * 1e4, yerr=profile["ci95"] * 1e4, color=colors, capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trading day relative to month-end (0 = last trading day, +1 = first day of next month)")
    ax.set_ylabel("Mean daily excess return (bps)")
    ax.set_title(f"{ticker}: average excess return by day of month (blue = held, bars = 95% CI)")
    ax.grid(alpha=0.3, axis="y")
    paths.append(out / "day_of_month_profile.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="RdYlGn", vmin=-1, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [f"{c:+d}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"-{i}" for i in grid.index])
    ax.set_xlabel("Exit (trading days relative to month-end close)")
    ax.set_ylabel("Entry (trading days before month-end close)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    r, c = list(grid.index).index(w.entry), list(grid.columns).index(w.exit)
    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="black", lw=2.5))
    fig.colorbar(im, ax=ax, label="Sharpe ratio")
    ax.set_title(f"{ticker}: Sharpe by window (box = chosen window)")
    paths.append(out / "sensitivity_heatmap.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(yearly))
    ax.bar(x - 0.2, yearly["Strategy excess"] * 100, width=0.4, label="Month-end strategy")
    ax.bar(x + 0.2, yearly["Buy & hold excess"] * 100, width=0.4, label=f"Buy & hold {ticker}", alpha=0.6)
    ax.set_xticks(x, yearly.index, rotation=60)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Excess return over T-bills (%)")
    ax.set_title("Calendar-year excess returns")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    paths.append(out / "yearly_returns.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=140)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Backtest the month-end Treasury rally on a Treasury ETF.")
    p.add_argument("--ticker", default="TLT")
    p.add_argument("--entry", type=int, default=3, help="enter at the close N trading days before the month's last close")
    p.add_argument("--exit", type=int, default=0, help="exit offset vs the month's last close (0 = last day, +1 = next month's first day)")
    p.add_argument("--cost-bps", type=float, default=2.0, help="cost per side in basis points")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--sims", type=int, default=2000, help="permutation-test simulations")
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    p.add_argument("--out", default="reports")
    p.add_argument(
        "--evaluate-oos",
        action="store_true",
        help="also report the out-of-sample holdout (most recent 20%% or 2 years, whichever is shorter). Run once, at the end.",
    )
    a = p.parse_args(argv)

    full = load_dataset(a.ticker, refresh=a.refresh)
    oos_start = holdout_start(full.index)
    is_end = full.index[full.index < oos_start][-1]
    span = full.loc[a.start : a.end]
    df = span.loc[:is_end]
    window = Window(a.entry, a.exit)
    res_all = run_backtest(span, window, a.cost_bps)
    res = slice_result(res_all, end=is_end)
    s = summary(res)
    perm = permutation_test(df, window, n_sims=a.sims)
    sub = subperiod_table(df, window, a.cost_bps)
    profile = day_of_month_profile(df)
    grid = sensitivity_grid(df, cost_bps=a.cost_bps)
    yearly = yearly_returns(res)

    oos_md = f"Out-of-sample holdout {oos_start.date()} to {full.index[-1].date()} is locked. Re-run with `--evaluate-oos` once, at the end."
    show_oos = a.evaluate_oos and span.index[-1] >= oos_start
    if show_oos:
        res_all_2x = run_backtest(span, window, 2 * a.cost_bps)
        rows = {}
        for label, r, r2 in [
            ("In-sample", res, slice_result(res_all_2x, end=is_end)),
            ("Out-of-sample", slice_result(res_all, start=oos_start), slice_result(res_all_2x, start=oos_start)),
        ]:
            sr, sr2 = summary(r), summary(r2)
            rows[f"{label}: net"] = sr["Month-end strategy"]
            rows[f"{label}: net, 2× costs"] = sr2["Month-end strategy"]
            rows[f"{label}: before costs"] = sr["Month-end strategy (before costs)"]
            rows[f"{label}: buy & hold"] = sr["Buy & hold"]
        oos_table = _summary_table(rows)
        oos_md = (
            f"Holdout {oos_start.date()} to {span.index[-1].date()}: the most recent 20% of history or 2 years, whichever is shorter.\n\n"
            + oos_table.to_markdown()
        )

    out = Path(a.out) / a.ticker.lower()
    out.mkdir(parents=True, exist_ok=True)
    charts = _charts(out, a.ticker, res, profile, grid, yearly, curve=res_all if show_oos else None, oos_start=oos_start if show_oos else None)
    trades = res.trades.assign(sample="in-sample")
    if show_oos:
        trades = pd.concat([trades, slice_result(res_all, start=oos_start).trades.assign(sample="out-of-sample")], ignore_index=True)
    trades.to_csv(out / "trades.csv", index=False)

    table = _summary_table(s)
    sub_fmt = sub.copy()
    for col in ["Strategy Sharpe", "Buy & hold Sharpe"]:
        sub_fmt[col] = sub_fmt[col].map("{:.2f}".format)
    for col in ["Strategy excess (ann.)", "Avg trade (net)", "Hit rate"]:
        sub_fmt[col] = sub_fmt[col].map("{:.2%}".format)

    header = (
        f"# Month-end Treasury rally: {a.ticker}\n\n"
        f"In-sample data {df.index[0].date()} to {df.index[-1].date()}. Long from the close {window.entry} trading days "
        f"before month-end to the close {window.exit:+d} days relative to month-end; T-bills otherwise. "
        f"Costs {a.cost_bps:g} bps per side.\n"
    )
    perm_txt = (
        f"Average trade excess over T-bills, before costs: {perm['actual_avg_trade']:.3%} vs {perm['random_avg_trade']:.3%} for a random "
        f"same-length window each month; p-value {perm['p_value']:.4f} ({int(perm['n_sims'])} simulations)."
    )
    md = "\n".join(
        [
            header,
            "## Summary (in-sample)\n",
            table.to_markdown(),
            "\n## Out-of-sample\n",
            oos_md,
            "\n## Is month-end special? (permutation test)\n",
            perm_txt,
            "\n## Sub-periods\n",
            sub_fmt.to_markdown(index=False),
            "\n## Charts\n",
            *[f"![{c.stem}]({c.name})" for c in charts],
        ]
    )
    (out / "report.md").write_text(md + "\n")

    print(header)
    print(table.to_string())
    print("\nOut-of-sample:\n" + (oos_table.to_string() if show_oos else oos_md))
    print("\nPermutation test:", perm_txt)
    print("\nSub-periods:\n" + sub_fmt.to_string(index=False))
    print(f"\nWrote {out / 'report.md'} and {len(charts)} charts.")


if __name__ == "__main__":
    main()
