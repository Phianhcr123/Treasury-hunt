# Month-end Treasury rally: IEF

Data 2002-07-31 to 2026-10-02. Long from the close 3 trading days before month-end to the close +0 days relative to month-end; T-bills otherwise. Costs 2 bps per side.

## Summary

|                        | Month-end strategy   | Buy & hold   | Rest of month only   |
|:-----------------------|:---------------------|:-------------|:---------------------|
| CAGR                   | 3.72%                | 3.37%        | 0.90%                |
| Excess return (ann.)   | 1.97%                | 1.84%        | -0.61%               |
| Volatility (ann.)      | 2.45%                | 6.79%        | 6.33%                |
| Sharpe                 | 0.81                 | 0.27         | -0.10                |
| Max drawdown           | -6.10%               | -23.92%      | -27.65%              |
| Time in market         | 14.32%               | –            | –                    |
| Trades                 | 291                  | –            | –                    |
| Avg trade (net excess) | 0.16%                | –            | –                    |
| Hit rate               | 60.48%               | –            | –                    |
| t-stat (per trade)     | 4.21                 | –            | –                    |

## Is month-end special? (permutation test)

Average trade excess over T-bills, before costs: 0.203% vs 0.013% for a random same-length window each month; p-value 0.0005 (2000 simulations).

## Sub-periods

| Period                       | Start      | End        |   Strategy Sharpe |   Buy & hold Sharpe | Strategy excess (ann.)   | Avg trade (net)   | Hit rate   |   Trades |
|:-----------------------------|:-----------|:-----------|------------------:|--------------------:|:-------------------------|:------------------|:-----------|---------:|
| Full sample                  | 2002-07-31 | 2026-10-02 |              0.81 |                0.27 | 1.97%                    | 0.16%             | 60.48%     |      291 |
| 2002–2014                    | 2002-07-31 | 2014-12-31 |              0.82 |                0.65 | 2.13%                    | 0.18%             | 60.67%     |      150 |
| 2015–present                 | 2015-01-02 | 2026-10-02 |              0.79 |               -0.16 | 1.81%                    | 0.15%             | 60.28%     |      141 |
| In paper's sample (≤2018)    | 2002-07-31 | 2018-12-31 |              0.85 |                0.55 | 2.08%                    | 0.17%             | 61.62%     |      198 |
| After paper's sample (2019+) | 2019-01-02 | 2026-10-02 |              0.71 |               -0.28 | 1.74%                    | 0.14%             | 58.06%     |       93 |
| Last 3 years                 | 2023-10-02 | 2026-10-02 |              0.48 |               -0.2  | 0.99%                    | 0.08%             | 61.11%     |       36 |

## Charts

![equity_curve](equity_curve.png)
![day_of_month_profile](day_of_month_profile.png)
![sensitivity_heatmap](sensitivity_heatmap.png)
![yearly_returns](yearly_returns.png)
