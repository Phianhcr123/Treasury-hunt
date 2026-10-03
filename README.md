# Month-End Treasury Rally

A backtest of the **month-end Treasury rally**. Bond index funds rebalance on the last trading day of each month and must buy longer Treasuries whatever the price, so Treasuries tend to rise in the last few trading days of the month.

The strategy holds a Treasury ETF (TLT by default) from the close **3 trading days before month-end** to the **month-end close**. The rest of the time it sits in T-bills. The 3-day window was fixed up front, following Hartley & Schwarz, rather than chosen after looking at the results.

## Results (TLT, Jul 2002 – Oct 2026, 2 bps cost per side)

| | Month-end strategy | Buy & hold TLT |
|---|---|---|
| Sharpe | **0.72** | 0.18 |
| Excess return over T-bills / yr | 3.6% | 2.6% |
| Max drawdown | −12% | −48% |
| Time in market | 14% | 100% |
| Avg trade (net) / hit rate | 0.30% / 62% (291 trades) | – |

- **Not luck:** a permutation test compares the month-end window with a random 3-day window in every month. The p-value is 0.0005.
- **Robust to the window:** the heatmap shows every window that exits at the month-end close has a Sharpe of 0.53–0.75.
- **Decaying:** the Sharpe was 0.87 up to 2018 (the paper's sample period), 0.41 from 2019 on, and about 0 over the last 3 years.

Full report and charts: [`reports/tlt/report.md`](reports/tlt/report.md).

## Run it

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Interactive dashboard on http://localhost:8631
uv run streamlit run app.py --server.port 8631

# Command-line report: prints stats, writes reports/<ticker>/report.md and PNG charts
uv run eom-treasury-rally --ticker TLT --entry 3 --exit 0 --cost-bps 2

# Tests
uv run pytest
```

CLI options:

- `--ticker`: TLT, IEF, EDV, GOVT or SHY.
- `--entry N`: enter at the close N trading days before month-end.
- `--exit K`: exit at the close K days relative to month-end. 0 is the last trading day; +1 is the first trading day of the next month.
- `--cost-bps`: trading cost per side, in basis points.
- `--start` / `--end`: restrict the date range.
- `--refresh`: re-download data instead of using the cache.

Data comes from Yahoo Finance and is cached in `data/`:

- dividend-adjusted ETF closes;
- the 13-week T-bill rate (`^IRX`), which is what cash earns.

## Project layout

- `src/eom_treasury_rally/data.py`: downloads and caches the data, and aligns the risk-free rate.
- `src/eom_treasury_rally/backtest.py`:
  - month-end calendar and position building;
  - backtest including costs and summary stats;
  - day-of-month profile, permutation test, window sensitivity grid, sub-period results.
- `src/eom_treasury_rally/cli.py`: the report and static charts.
- `app.py`: the Streamlit dashboard.
- `tests/`: unit tests, including detecting a planted month-end effect and giving no false positives on pure noise.

## Assumptions and caveats

- **Trade timing:** trades fill at the closing price, as with market-on-close orders. The trade dates come from the calendar, so the backtest doesn't use any information from the future.
- **Incomplete final month:** if the data ends partway through a month, that month is skipped, because its last trading day isn't known yet.
- **ETFs, not futures:** Treasury futures would be cheaper to trade. But Yahoo's continuous futures series isn't roll-adjusted, and the quarterly rolls fall near month-end, which would contaminate exactly the window being tested.
- **Small, noisy returns:** about 0.3% per trade, with long flat periods. Report the recent decay alongside the headline Sharpe.
