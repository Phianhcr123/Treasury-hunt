# Month-End Treasury Rally

**Submitted strategy.** Buy TLT at the close three trading days before month-end, sell at the month-end close, and hold T-bills otherwise. The window was fixed before the backtest, following Hartley and Schwarz, rather than chosen from the results. The quant note is [`reports/quant_note/month_end_tlt_note.pdf`](reports/quant_note/month_end_tlt_note.pdf). The same text is in [`reports/quant_note/note.html`](reports/quant_note/note.html).

Dependencies are `pyproject.toml` and `uv.lock`. Data is downloaded from Yahoo Finance by the code and cached in `data/`, which is not committed.

## Headline results

Net of 2 bps per side. In-sample and the holdout are separate. Do not quote one Sharpe over the whole history: that blend includes the test set.

| | In-sample strategy | In-sample buy & hold TLT | Holdout strategy | Holdout buy & hold TLT |
|---|---|---|---|---|
| Period | 31 Jul 2002 – 30 Sep 2024 | same | 1 Oct 2024 – 2 Oct 2026 | same |
| Excess return / year | 4.0% | 3.7% | −1.0% | −10.5% |
| Volatility | 5.1% | 14.5% | 4.0% | 11.6% |
| Sharpe | **0.78** | 0.26 | **−0.26** | −0.91 |
| Max drawdown | −12.4% | −48.4% | −5.6% | −14.1% |
| Turnover | 24× | 0 | 24× | 0 |

The holdout lost to T-bills. Its total return is still positive only because cash yields were high. A permutation test on the in-sample months, 2,000 random windows of the same length, has p-value 0.0005. Windows that exit on the month-end close have in-sample Sharpes from 0.58 to 0.79. Entry −3 is 0.78. IEF with the same rule, in-sample only, has a Sharpe of 0.83. One extra holdout look, entry −10, scored +0.07 and was discarded. Those looks are in `reports/oos_evaluations.csv`.

```bash
uv run eom-treasury-rally --ticker TLT --entry 3 --exit 0 --cost-bps 2 --evaluate-oos
```

## Other strategies left in the repo

Three other books are in the code. They are not the submitted strategy, and their numbers are not in the quant note. They were not selected because their return per unit of risk does not outweigh the month-end TLT book. That book is about 5% volatility, with a −12% in-sample drawdown, in a liquid ETF.

- **Options overlay** (`pages/2_Options_Overlay.py`). Calls on the same month-end window. They capture the drift and do not beat holding the ETF per unit of risk, after spreads.
- **Adaptive trend** (`pages/3_Adaptive_Trend_Momentum.py`). A 200-day trend filter with a 15% volatility target. On QQQ it earns more dollars at about 14% volatility, so the Sharpe is lower, and the page has several free parameters.
- **Diversified trend** (`pages/4_Diversified_Trend_Research.py`). Monthly time-series momentum across ETF proxies, with volatility-scaled long/short weights. It is a different hypothesis, on proxies rather than futures.

Charts from the submitted backtest: [`reports/tlt/report.md`](reports/tlt/report.md).

## Options overlay

Not the submitted strategy. The numbers below are the research check described above.

**Why a drift tree isn't a mispricing detector.** Option prices are set under the risk-neutral measure, so the underlying's expected return doesn't enter them. A dealer who sells you a call hedges with delta shares, and the drift helps that hedge exactly as much as it helps your call. A tree with an upward drift therefore values *every* call above market and every put below it, by roughly delta × drift. The month-end volatility isn't mispriced either: TLT is slightly *calmer* in the window (13.2% against 14.4% annualized).

**What the overlay tests instead:** whether calls hold the drift more efficiently than the ETF, after spreads.

Historical test, 287 months (Sharpe annualized from monthly trades):

| | Avg per trade | Sharpe |
|---|---|---|
| ETF in window | +0.28% | **0.74** |
| ATM 30-day call (as % of premium) | **+8.8%** | 0.65 |
| Call sized to the same exposure as the ETF | +0.25% | 0.61 |
| Delta-hedged call (the "mispricing") | −0.01% of spot | ≈ 0 |

- **Calls capture the drift but never beat the ETF.** In-the-money, longer-dated calls come closest (Sharpe 0.71–0.76).
- **Out-of-the-money calls suffer:** short-dated ones lose most of the edge to spreads, which can be 16–64% of the premium.
- **The model is calibrated:** the drift tree predicted +10.3% per call trade after costs; the backtest realized +8.8%.
- **What calls are good for:** leverage (about 30× for at-the-money calls) and capped downside, not extra edge.

How the historical prices are built:

- **Pricing model:** synthetic American CRR prices at a flat implied volatility, using real TLT ex-dividend dates.
- **Volatility:** the MOVE bond-volatility index × duration 16, which matches Cboe's TLT implied-vol index (VXTLT).
- **Costs:** a $0.01/share half-spread plus $0.65 per contract on each side.

The **live scanner** pulls today's TLT call chain and backs out each contract's implied volatility. It then runs the drift tree over the next month-end window and ranks contracts by expected return per unit of risk, against the same figure for holding TLT.

Report: [`reports/options/report.md`](reports/options/report.md).

## Run it

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Interactive dashboard on http://localhost:8631
uv run streamlit run 1_Month_End_Treasury_Rally.py --server.port 8631

# Headline result: in-sample and holdout, net of 2 bps. Matches the quant note.
uv run eom-treasury-rally --ticker TLT --entry 3 --exit 0 --cost-bps 2 --evaluate-oos

# Options report: historical call backtest, contract grid and live chain scan
uv run eom-options --dte 30 --moneyness 1.0 --half-spread 0.01 --drift-bps 11

# Tests
uv run pytest
```

`eom-treasury-rally` options:

- `--ticker`: TLT, IEF, EDV, GOVT or SHY.
- `--entry N`: enter at the close N trading days before month-end.
- `--exit K`: exit at the close K days relative to month-end. 0 is the last trading day; +1 is the first trading day of the next month.
- `--cost-bps`: trading cost per side, in basis points.
- `--start` / `--end`: restrict the date range.
- `--refresh`: re-download data instead of using the cache.
- `--evaluate-oos`: also report the out-of-sample holdout. Run it once, at the end.

### Out-of-sample holdout

Following the track rule, the most recent 20% of the selected sample or the most recent 2 years, whichever is shorter, is held out. The cut is moved back to the start of its month, so no monthly trade straddles it. For the full TLT history the holdout is 2024-10-01 onward. Every strategy page (month-end rally, options overlay, adaptive trend, diversified trend) uses the same rule, and each strategy has its own holdout. Shrinking the sample slider shortens the holdout to 20% once 20% is under 2 years.

- **Hidden by default.** The CLI report and every dashboard number use in-sample data only, including the heatmaps, permutation test, sub-periods, sensitivity tables, contract grid and trade logs. While locked, the equity chart's axis is scaled on in-sample data only.
- **Revealing it.** The equity chart ends in an amber "out-of-sample locked" window, with a dashed line where the holdout starts. Click the window to evaluate it once. The out-of-sample section then shows in-sample and out-of-sample side by side: net of costs, net of 2× costs, before costs, and the benchmark, with Sharpe, drawdown, turnover and an equity curve. The sidebar's **Relock** button hides it again.
- **Leak alert.** Changing any setting after revealing shows a "TEST SET LEAKED" alert with the number of peeks this session.
- **Audit log.** Every strategy and setting evaluated on the holdout is appended to `reports/oos_evaluations.csv`. Report every look in the quant note.
- **Gross vs net.** Equity curves draw the strategy before costs as a dotted line next to the net line, so the cost drag is visible. Every reported number is net of costs unless labelled "before costs".
- **S&P 500 reference.** Each equity chart also shows SPY with dividends, in excess of T-bills, over the same period.

`eom-options` options:

- `--dte`: days to expiry at entry.
- `--moneyness`: strike divided by spot (1.0 is at the money).
- `--half-spread`: dollars per share paid on each side.
- `--drift-bps`: assumed excess drift per day inside the window.
- `--no-scan`: skip the live option-chain scan.

Data comes from Yahoo Finance and is cached in `data/`:

- dividend-adjusted and unadjusted ETF closes, plus dividends;
- the 13-week T-bill rate (`^IRX`);
- the MOVE index;
- the live TLT option chain.

## Project layout

- `src/eom_treasury_rally/data.py`: downloads and caches the data, and aligns the risk-free rate.
- `src/eom_treasury_rally/backtest.py`:
  - month-end calendar and position building;
  - backtest including costs and summary stats;
  - day-of-month profile, permutation test, window sensitivity grid, sub-period results.
- `src/eom_treasury_rally/pricing.py`:
  - Black-Scholes;
  - CRR binomial pricing (American, discrete dividends);
  - implied volatility;
  - the drift tree.
- `src/eom_treasury_rally/options_backtest.py`: the synthetic historical call overlay.
- `src/eom_treasury_rally/scanner.py`: the live chain scan and the NYSE month-end calendar.
- `src/eom_treasury_rally/diversified_trend.py`: monthly, volatility-scaled trend research using adjusted ETF proxies.
- `src/eom_treasury_rally/cli.py`, `options_cli.py`: the reports and static charts.
- `1_Month_End_Treasury_Rally.py`, `pages/`: the Streamlit dashboard and research pages.
- `tests/`: unit tests for calendar logic, costs, the planted-effect and pure-noise checks, pricing convergence, early exercise, implied-volatility round trips, and the drift tree.

## Assumptions and caveats

- **Trade timing:** trades fill at the closing price, as with market-on-close orders. The trade dates come from the calendar, so the backtest doesn't use any information from the future.
- **Partial months:** partial windows at the start or end of the data are skipped.
- **ETFs, not futures:** Treasury futures would be cheaper to trade. But Yahoo's continuous futures series isn't roll-adjusted, and the quarterly rolls fall near month-end, which would contaminate exactly the window being tested.
- **Synthetic option prices:** these test the stated hypothesis that the market prices options at a flat volatility. They can't show whether real quotes already lean toward the drift. Use the live scanner, or saved snapshots across several month-ends, to check that.
- **Small, noisy returns:** about 0.33% per trade in-sample, with long flat periods. The holdout Sharpe is −0.26. Report that next to the in-sample 0.78.
