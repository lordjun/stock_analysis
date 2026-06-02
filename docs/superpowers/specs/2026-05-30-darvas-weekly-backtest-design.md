# Darvas Weekly Backtest Design

## Goal

Scan A-share stocks over the past five years with a quantified Darvas-style weekly trend model and report trade win rate.

## Model

The strategy uses weekly bars, not daily bars. Daily A-share data is fetched with forward-adjusted prices and aggregated to weekly OHLCV bars. A signal is evaluated only when at least 40 prior weekly bars exist.

## Entry Rules

- Box window: the 20 complete weeks before the signal week.
- Box top: highest high in the 20-week box window.
- Breakout: signal week close is greater than the prior 20-week box top.
- Volume confirmation: signal week volume is greater than 1.5 times the prior 20-week average volume.
- Trend filter: signal week close is greater than the 20-week moving average, and the 20-week moving average is greater than the 40-week moving average.
- Entry price: signal week close.
- De-duplication: while a stock is in a five-week observation window, later signals for the same stock are ignored.

## Exit Rules

After entry, observe the next five weekly bars:

- Fixed stop: if any weekly low is at or below 90% of entry price, exit at `entry * 0.90`.
- Trailing drawdown stop: track the highest high since entry. If any weekly low is at or below 80% of that tracked high, exit at `tracked_high * 0.80`.
- Same-week conflict: if both stops trigger in the same week, use the lower exit price.
- Time exit: if neither stop triggers within five weeks, exit at the fifth week close.

## Statistics

The main win rate is the count of trades with realized return greater than 0 divided by total trades. The report also includes total trades, average return, median return, exit reason counts, and yearly win rate based on entry year.

For trades held for the full five weeks, the report additionally calculates the best possible five-week high return, defined as `max_high_5w / entry - 1`. This is reported separately from the realized win rate.

## Outputs

- Markdown summary report under `reports/darvas/`.
- CSV trade details under `reports/darvas/`.
- Console summary for quick review.

## Data Limits

AKShare upstream availability can vary. Stocks with insufficient data, failed downloads, or unusable columns are skipped and listed in the run summary.
