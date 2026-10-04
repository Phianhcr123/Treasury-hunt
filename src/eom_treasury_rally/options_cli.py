from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import Window  # noqa: E402
from .options_backtest import OptionSpec, iv_proxy_check, options_summary, run_options_backtest  # noqa: E402
from .scanner import scan_calls  # noqa: E402

GRID_DTE = (7, 14, 30, 60)
GRID_MONEYNESS = (0.98, 1.0, 1.02)


def contract_grid(window: Window, half_spread: float, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Sharpe by expiry and strike, using only trades entered on or after `start` and closed by `end`."""
    rows = []
    for dte in GRID_DTE:
        for m in GRID_MONEYNESS:
            t = run_options_backtest(window, OptionSpec(dte=dte, moneyness=m, half_spread=half_spread), with_model=False)
            # Cut trades rather than prices, so options near the cut are still priced with their known dividends.
            t = t[(pd.to_datetime(t["entry"]) >= pd.Timestamp(start or "1900-01-01")) & (pd.to_datetime(t["exit"]) <= pd.Timestamp(end or "2260-01-01"))]
            s = options_summary(t)
            rows.append(
                {
                    "Days to expiry": dte,
                    "Strike / spot": m,
                    "Avg call return": s.loc["Call, % of premium (excess)", "Avg per trade"],
                    "Call Sharpe": s.loc["Call, % of premium (excess)", "Sharpe (ann.)"],
                    "Delta-matched Sharpe": s.loc["Delta-matched call overlay (excess)", "Sharpe (ann.)"],
                    "ETF Sharpe": s.loc["ETF in window (excess)", "Sharpe (ann.)"],
                    "Costs / premium": t["cost_share_of_premium"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Month-end call overlay: synthetic options backtest and live TLT call scan.")
    p.add_argument("--entry", type=int, default=3)
    p.add_argument("--exit", type=int, default=0)
    p.add_argument("--dte", type=int, default=30)
    p.add_argument("--moneyness", type=float, default=1.0)
    p.add_argument("--half-spread", type=float, default=0.01, help="dollars per share paid each side")
    p.add_argument("--drift-bps", type=float, default=11.0, help="assumed excess drift per day inside the window")
    p.add_argument("--no-scan", action="store_true", help="skip the live option-chain scan")
    p.add_argument("--out", default="reports/options")
    a = p.parse_args(argv)

    window = Window(a.entry, a.exit)
    spec = OptionSpec(dte=a.dte, moneyness=a.moneyness, half_spread=a.half_spread)
    trades = run_options_backtest(window, spec, excess_drift_bps=a.drift_bps)
    summ = options_summary(trades)
    grid = contract_grid(window, a.half_spread)
    check = iv_proxy_check()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "option_trades.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot = grid.pivot(index="Days to expiry", columns="Strike / spot", values="Delta-matched Sharpe")
    for m in pivot.columns:
        ax.plot(pivot.index, pivot[m], marker="o", label=f"Calls, strike = {m:.2f} x spot")
    ax.axhline(grid["ETF Sharpe"].iloc[0], color="black", ls="--", label="Just hold the ETF in the window")
    ax.set_xlabel("Days to expiry at entry")
    ax.set_ylabel("Sharpe (same dollar exposure as the ETF)")
    ax.set_title("Calls never beat holding the ETF; the most stock-like calls come closest")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "calls_vs_etf.png", dpi=140)
    plt.close(fig)

    pct = lambda x: f"{x:.2%}"  # noqa: E731
    summ_fmt = summ.copy()
    for c in ["Avg per trade", "Std per trade", "Hit rate", "Worst trade"]:
        summ_fmt[c] = summ_fmt[c].map(pct)
    for c in ["Sharpe (ann.)", "t-stat"]:
        summ_fmt[c] = summ_fmt[c].map("{:.2f}".format)
    grid_fmt = grid.copy()
    for c in ["Avg call return", "Costs / premium"]:
        grid_fmt[c] = grid_fmt[c].map(pct)
    for c in ["Call Sharpe", "Delta-matched Sharpe", "ETF Sharpe"]:
        grid_fmt[c] = grid_fmt[c].map("{:.2f}".format)

    model_line = (
        f"Drift-tree model expected {trades['model_expected_ret_net'].mean():.2%} per call trade after costs; "
        f"realized {trades['call_ret'].mean():.2%}."
    )
    proxy_line = (
        f"Implied-vol proxy (MOVE x duration 16) averaged {check['avg_iv_proxy']:.1%} vs {check['avg_next_21d_realized']:.1%} "
        f"realized over the next 21 days (correlation {check['correlation']:.2f})."
    )
    parts = [
        "# Month-end call overlay (TLT)\n",
        f"Buy a {a.dte}-day call (strike {a.moneyness:.2f} x spot) at the close {a.entry} days before month-end, "
        f"sell at the close {a.exit:+d} days relative to month-end. Synthetic American CRR prices at a flat implied vol; "
        f"${a.half_spread:.2f}/share half-spread plus $0.65/contract each side.\n",
        summ_fmt.to_markdown(),
        "\n" + model_line,
        "\n" + proxy_line,
        "\n## Contract choice\n",
        grid_fmt.to_markdown(index=False),
        "\n![calls vs ETF](calls_vs_etf.png)",
    ]
    print(parts[1])
    print(summ_fmt.to_string())
    print("\n" + model_line + "\n" + proxy_line)
    print("\nContract choice:\n" + grid_fmt.to_string(index=False))

    if not a.no_scan:
        scan, meta = scan_calls(window=window, excess_drift_bps=a.drift_bps)
        head = (
            f"Spot {meta['spot']:.2f} on {meta['today']}. Next window: enter {meta['window_entry']} close, exit "
            f"{meta['window_exit']} close. Holding TLT itself scores {meta['etf_edge_per_risk']:.3f} expected return per unit of risk."
        )
        cols = ["contract", "strike", "bid", "ask", "market_iv", "elasticity", "expected_ret_net", "edge_per_risk", "cost_bps_of_spot_per_delta"]
        print("\nLive scan: " + head)
        print(scan[cols].head(10).round(3).to_string(index=False) if not scan.empty else "No contracts passed the filters.")
        parts += ["\n## Live scan\n", head + "\n", scan[cols].head(15).round(3).to_markdown(index=False) if not scan.empty else "No contracts."]
        scan.to_csv(out / "live_scan.csv", index=False)

    (out / "report.md").write_text("\n".join(parts) + "\n")
    print(f"\nWrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
