# Agent Instructions: A-share Medium-Term Candlestick Model Research

## Project Goal

This project aims to find a medium-term A-share candlestick pattern model from roughly the past 10 years of A-share K-line data.

The model should satisfy:

- Low trading frequency, no ultra-short-term trading.
- Target holding period after entry: 1 month to 6 months, roughly 4 to 26 weeks.
- Core return should come from capturing a trend leg, not high-frequency arbitrage.
- Backtest win-rate target above 50%.
- Strict stop-loss rules are mandatory.
- Winners must have room to run so average profit can exceed average loss.
- Final output should be reproducible research evidence, not subjective chart-reading experience.

This project is for research and backtesting only. It is not investment advice and does not place orders automatically.

## Current Project Background

The current workspace already has:

- `daily_sector_report.py`: A-share sector and leading-stock daily report script.
- `darvas_weekly_backtest.py`: Existing Darvas-style weekly breakout backtest draft.
- `tests/test_darvas_weekly_backtest.py`: Existing tests for part of the backtest core logic.
- `requirements.txt`: Existing base dependencies including `akshare`, `pandas`, `numpy`, `matplotlib`, `mplfinance`, and `requests`.

Future research should prefer reusing existing code and directory structure, and avoid unrelated refactoring.

## Research Priority

Prioritize these low-frequency K-line pattern models:

1. Weekly box breakout model  
   Examples: Darvas Box, long platform breakout, volume-confirmed breakout.

2. Trend pullback and restart model  
   Examples: medium/long-term moving averages in bullish alignment, then pullback to 20-day/60-day/10-week average and restart upward.

3. Bottom structure reversal model  
   Examples: after a long decline, volume contraction base-building and volume breakout above neckline. Recognition must be rule-based, not subjective.

4. Strong consolidation breakout model  
   Examples: sideways consolidation after an uptrend, volatility contraction, then volume breakout above prior high.

Do not start with complex machine learning models. Start with clear, interpretable, backtestable rule models. Use Qlib or RQAlpha only if there is a clear later need.

## Data Scope

Default research scope:

- Market: A-shares.
- Period: past 10 years, split into segments if needed.
- Frequency: daily and weekly bars.
- Holding period: 4 to 26 weeks.
- Initial preferred universe: all A-shares or reproducible universes such as CSI 800, CSI 300, or CSI 500.
- Exclude:
  - ST and *ST stocks.
  - Stocks with long suspensions or severe data gaps.
  - Stocks listed for less than 2 years.
  - Stocks with obviously insufficient liquidity.
  - Delisting and survivorship-bias issues must be stated clearly.

Data-source priority:

1. Existing AKShare interfaces in this project.
2. Tushare if more stable data is needed, but token, permission, and dependency checks must happen first.
3. If online data is unstable, request user approval before installing or using a Tongdaxin local data reader.
4. Do not install new plugins or dependencies before user approval.

## Backtest Principles

Avoid these issues:

- Do not use future data.
- Do not assume entry with intraday information unavailable when the signal is generated.
- Default entry price should be next trading day's open after signal confirmation, or clearly disclose the bias if signal close is used.
- Stop-loss, take-profit, and trailing-stop rules must be fixed in the backtest rules, not adjusted after seeing results.
- Parameter optimization must split train, validation, and test periods to avoid overfitting.
- At least annual statistics are required. Do not rely only on aggregate results.
- Include fees and slippage, or at least run transaction-cost sensitivity tests.
- Record failed samples and extreme drawdowns.

## Signal And Trading Rule Requirements

Every candidate model must specify:

- Entry conditions.
- Initial stop-loss point.
- Trailing stop point.
- Take-profit or exit conditions.
- Maximum holding period.
- Single-trade risk.
- Whether adding to a position is allowed.
- Whether multiple stocks can be held simultaneously.
- Whether same-industry or same-theme concentration is limited.

Default risk constraints:

- Initial single-trade loss should generally not exceed 8% to 12% of entry price unless backtest evidence supports otherwise.
- If price rises, the stop should be raised gradually.
- Prefer trend-following exits, such as:
  - Breaking below a key weekly moving average.
  - Breaking below a recent swing low.
  - Breaking below an ATR trailing stop.
  - Fixed-percentage drawdown from highest high.
- Take-profit should not be fixed too early unless backtests show fixed take-profit clearly outperforms trailing exits.

## Acceptance Metrics

A model enters candidate conclusion only if it satisfies all of these:

- Enough sample trades; do not accept wins from a tiny sample.
- Total win rate above 50%.
- Average profit exceeds average loss.
- Profit/loss ratio and expectancy are positive.
- Maximum drawdown is acceptable.
- Results are not extremely dependent on a single year.
- Performance is separately explained in bull, sideways, and bear market conditions.
- Positive expectancy remains after transaction costs.
- Results do not collapse after small parameter changes.
- At least one out-of-sample test segment is preserved.

Key metrics:

- Win rate.
- Average return.
- Median return.
- Average profit / average loss.
- Single-trade expectancy.
- Maximum drawdown.
- Annual return.
- Annual win rate.
- Consecutive loss count.
- Holding-period distribution.
- Exit-reason distribution.

## Output Requirements

Each research round should include:

- Data source and date range.
- Stock universe definition.
- Model rules.
- Parameter settings.
- Backtest assumptions.
- Transaction-cost assumptions.
- Core statistics.
- Annual results.
- Failure reasons.
- Whether the model passed acceptance.
- Next improvement suggestion with only one major variable changed.

Final output should include:

- Recommended K-line pattern model.
- Clear entry rules.
- Clear stop-loss rules.
- Clear take-profit or trailing-exit rules.
- Suitable market environment.
- Unsuitable market environment.
- Backtest evidence.
- Execution notes for a human user.

## Engineering Constraints

- Prefer Python, pandas, and numpy.
- Prefer reusable scripts over one-off notebooks.
- Data, results, and reports should be output under `reports/`.
- Tests should be placed under `tests/`.
- Critical logic must be unit-testable.
- Run a small-sample smoke test before large-scale backtests.
- Long tasks should save intermediate results to avoid repeated data fetching.
- Do not delete existing user files.
- Do not overwrite user-unconfirmed result files.

## Execution Gates

Execution order must follow:

1. Prepare and confirm this `agent.md` first.
2. After user confirms saving it, submit a detailed execution plan.
3. Only after user confirms the plan, start dependency installation, data fetching, code writing, or backtest runs.
4. If new plugins, skills, Python packages, or Tongdaxin readers are needed, explain purpose, command, and risk first, then wait for user approval.
5. If PowerShell permissions prevent installation, provide commands for the user to run manually, then continue after the user completes them.

## Default Research Attitude

Do not overfit just to satisfy "win rate above 50%".

If no model satisfying the criteria can be found, state that clearly with failure evidence. The model must survive out-of-sample validation, not merely look good on historical data.
