from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
import time

import pandas as pd


MODEL_NAMES = [
    "weekly_box_breakout",
    "trend_pullback_restart",
    "consolidation_breakout",
    "breakout_pullback_restart",
    "stage_high_breakout_retest",
    "jianghua_acceleration_retest",
    "uptrend_bullish_engulfing",
]


class ExitRule(str, Enum):
    TRAILING_HIGH_DRAWDOWN = "trailing_high_drawdown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research medium-term A-share K-line pattern models.")
    parser.add_argument("--start-date", default="20160101", help="YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD, defaults to latest available upstream data")
    parser.add_argument("--universe", default="all", help="Research universe label; currently supports all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks for smoke tests")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many stocks after universe filtering")
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default="all")
    parser.add_argument("--mode", choices=["research", "sweep"], default="research")
    parser.add_argument("--output-dir", default="reports/kline_models")
    parser.add_argument("--cache-dir", default="data/cache/kline_models")
    parser.add_argument("--fee-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--sweep-top-n", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run_sweep(args) if args.mode == "sweep" else run_research(args)
    for path in output_paths:
        print(path)


@dataclass(frozen=True)
class TradeSetup:
    code: str
    name: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    initial_stop: float
    max_holding_bars: int
    exit_rule: ExitRule
    trailing_drawdown: float


@dataclass(frozen=True)
class CompletedTrade:
    code: str
    name: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    realized_return: float
    holding_bars: int
    max_high: float


@dataclass(frozen=True)
class PatternSignal:
    model: str
    signal_index: int
    signal_date: pd.Timestamp
    trigger_price: float
    initial_stop: float
    metadata: dict[str, float]


@dataclass(frozen=True)
class SweepResult:
    model: str
    params: dict[str, object]
    summary: dict[str, object]


def simulate_position(setup: TradeSetup, future_bars: pd.DataFrame) -> CompletedTrade:
    bars = future_bars.reset_index(drop=True).copy()
    if bars.empty:
        raise ValueError("future_bars must not be empty")

    max_bars = min(int(setup.max_holding_bars), len(bars))
    tracked_high = float(setup.entry_price)

    for offset in range(max_bars):
        row = bars.iloc[offset]
        low = float(row["low"])
        exit_candidates: list[tuple[str, float]] = []

        if low <= setup.initial_stop:
            exit_candidates.append(("initial_stop", float(setup.initial_stop)))

        if setup.exit_rule == ExitRule.TRAILING_HIGH_DRAWDOWN:
            trailing_stop = tracked_high * (1 - float(setup.trailing_drawdown))
            if low <= trailing_stop:
                exit_candidates.append(("trailing_stop", float(trailing_stop)))

        if exit_candidates:
            exit_reason, exit_price = min(exit_candidates, key=lambda item: item[1])
            return _completed_trade(setup, row, exit_price, exit_reason, offset + 1, tracked_high)

        tracked_high = max(tracked_high, float(row["high"]))

    exit_row = bars.iloc[max_bars - 1]
    exit_price = float(exit_row["close"])
    return _completed_trade(setup, exit_row, exit_price, "time_exit", max_bars, tracked_high)


def _completed_trade(
    setup: TradeSetup,
    row: pd.Series,
    exit_price: float,
    exit_reason: str,
    holding_bars: int,
    max_high: float,
) -> CompletedTrade:
    return CompletedTrade(
        code=setup.code,
        name=setup.name,
        signal_date=pd.Timestamp(setup.signal_date),
        entry_date=pd.Timestamp(setup.entry_date),
        entry_price=float(setup.entry_price),
        exit_date=pd.Timestamp(row["date"]),
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        realized_return=float(exit_price) / float(setup.entry_price) - 1,
        holding_bars=int(holding_bars),
        max_high=float(max_high),
    )


def find_weekly_box_breakouts(
    bars: pd.DataFrame,
    box_bars: int = 20,
    volume_multiplier: float = 1.5,
    ma_fast: int = 20,
    ma_slow: int = 40,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    for index in range(max(box_bars, ma_slow), len(df)):
        prior_box = df.iloc[index - box_bars : index]
        box_high = float(prior_box["high"].max())
        box_low = float(prior_box["low"].min())
        avg_volume = float(prior_box["volume"].mean())
        close = float(df.at[index, "close"])
        volume = float(df.at[index, "volume"])
        fast = float(df.at[index, "ma_fast"])
        slow = float(df.at[index, "ma_slow"])

        if close > box_high and volume > avg_volume * volume_multiplier and close > fast > slow:
            signals.append(
                PatternSignal(
                    model="weekly_box_breakout",
                    signal_index=index,
                    signal_date=pd.Timestamp(df.at[index, "date"]),
                    trigger_price=close,
                    initial_stop=box_low,
                    metadata={"box_high": box_high, "volume_ratio": volume / avg_volume},
                )
            )

    return signals


def summarize_trades(trades: list[CompletedTrade]) -> dict[str, object]:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "median_return": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "profit_loss_ratio": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "exit_reason_counts": {},
            "holding_period_distribution": {"1-4": 0, "5-8": 0, "9-13": 0, "14+": 0},
            "annual": {},
        }

    returns = pd.Series([trade.realized_return for trade in trades], dtype=float)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    win_rate = float(len(winners) / len(returns))
    average_winner = float(winners.mean()) if not winners.empty else 0.0
    average_loser = float(losers.mean()) if not losers.empty else 0.0
    profit_loss_ratio = abs(average_winner / average_loser) if average_loser else 0.0
    expectancy = win_rate * average_winner + (1 - win_rate) * average_loser

    return {
        "trade_count": len(trades),
        "win_rate": win_rate,
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "average_winner": average_winner,
        "average_loser": average_loser,
        "profit_loss_ratio": float(profit_loss_ratio),
        "expectancy": float(expectancy),
        "max_drawdown": _max_drawdown(returns.tolist()),
        "exit_reason_counts": dict(Counter(trade.exit_reason for trade in trades)),
        "holding_period_distribution": _holding_period_distribution(trades),
        "annual": _annual_summary(trades),
    }


def rank_sweep_results(results: list[SweepResult], min_trades: int = 30) -> list[SweepResult]:
    eligible = [result for result in results if int(result.summary.get("trade_count", 0)) >= min_trades]
    return sorted(
        eligible,
        key=lambda result: (
            float(result.summary.get("win_rate", 0.0)),
            float(result.summary.get("expectancy", 0.0)),
            float(result.summary.get("profit_loss_ratio", 0.0)),
        ),
        reverse=True,
    )


def backtest_signals(
    code: str,
    name: str,
    bars: pd.DataFrame,
    signals: list[PatternSignal],
    max_holding_bars: int = 26,
    max_initial_loss: float = 0.10,
    trailing_drawdown: float = 0.20,
) -> list[CompletedTrade]:
    df = _prepare_bars(bars)
    trades: list[CompletedTrade] = []
    next_allowed_signal_index = 0

    for signal in sorted(signals, key=lambda item: item.signal_index):
        if signal.signal_index < next_allowed_signal_index:
            continue
        entry_index = signal.signal_index + 1
        if entry_index >= len(df):
            continue

        entry_row = df.iloc[entry_index]
        entry_price = float(entry_row["open"])
        capped_stop = entry_price * (1 - max_initial_loss)
        initial_stop = max(float(signal.initial_stop), capped_stop)
        future = df.iloc[entry_index : entry_index + max_holding_bars]
        setup = TradeSetup(
            code=code,
            name=name,
            signal_date=pd.Timestamp(signal.signal_date),
            entry_date=pd.Timestamp(entry_row["date"]),
            entry_price=entry_price,
            initial_stop=initial_stop,
            max_holding_bars=max_holding_bars,
            exit_rule=ExitRule.TRAILING_HIGH_DRAWDOWN,
            trailing_drawdown=trailing_drawdown,
        )
        trade = simulate_position(setup, future)
        trades.append(trade)
        next_allowed_signal_index = entry_index + trade.holding_bars

    return trades


def normalize_price_bars(bars: pd.DataFrame, source: str = "normalized") -> pd.DataFrame:
    if source == "akshare_a_hist":
        if len(bars.columns) < 7:
            raise ValueError("akshare_a_hist bars must have at least 7 columns")
        mapped = pd.DataFrame(
            {
                "date": bars.iloc[:, 0],
                "open": bars.iloc[:, 2],
                "high": bars.iloc[:, 4],
                "low": bars.iloc[:, 5],
                "close": bars.iloc[:, 3],
                "volume": bars.iloc[:, 6],
            }
        )
        return _prepare_bars(mapped)
    return _prepare_bars(bars)


def aggregate_weekly_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    daily = normalize_price_bars(daily_bars)
    daily = daily.sort_values("date")
    daily["_actual_date"] = daily["date"]
    weekly = (
        daily.set_index("date")
        .resample("W-FRI")
        .agg(
            {
                "_actual_date": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .rename(columns={"_actual_date": "date"})
        .reset_index(drop=True)
    )
    return weekly[["date", "open", "high", "low", "close", "volume"]]


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for realized_return in returns:
        equity *= 1 + realized_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        max_drawdown = min(max_drawdown, drawdown)
    return float(max_drawdown)


def _holding_period_distribution(trades: list[CompletedTrade]) -> dict[str, int]:
    distribution = {"1-4": 0, "5-8": 0, "9-13": 0, "14+": 0}
    for trade in trades:
        if trade.holding_bars <= 4:
            distribution["1-4"] += 1
        elif trade.holding_bars <= 8:
            distribution["5-8"] += 1
        elif trade.holding_bars <= 13:
            distribution["9-13"] += 1
        else:
            distribution["14+"] += 1
    return distribution


def _annual_summary(trades: list[CompletedTrade]) -> dict[str, dict[str, float]]:
    annual: dict[str, dict[str, float]] = {}
    for trade in trades:
        year = str(pd.Timestamp(trade.entry_date).year)
        bucket = annual.setdefault(year, {"trades": 0, "wins": 0, "losses": 0, "return_sum": 0.0, "win_rate": 0.0})
        bucket["trades"] += 1
        bucket["return_sum"] += float(trade.realized_return)
        if trade.realized_return > 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1

    for bucket in annual.values():
        bucket["win_rate"] = bucket["wins"] / bucket["trades"] if bucket["trades"] else 0.0
        bucket["average_return"] = bucket["return_sum"] / bucket["trades"] if bucket["trades"] else 0.0
        del bucket["return_sum"]
    return annual


def find_trend_pullback_restarts(
    bars: pd.DataFrame,
    ma_fast: int = 10,
    ma_slow: int = 30,
    pullback_tolerance: float = 0.03,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    for index in range(max(ma_fast, ma_slow) + 1, len(df)):
        prior_index = index - 1
        close = float(df.at[index, "close"])
        fast = float(df.at[index, "ma_fast"])
        slow = float(df.at[index, "ma_slow"])
        prior_fast = float(df.at[prior_index, "ma_fast"])
        prior_slow = float(df.at[prior_index, "ma_slow"])
        prior_low = float(df.at[prior_index, "low"])
        prior_high = float(df.at[prior_index, "high"])
        prior_close = float(df.at[prior_index, "close"])

        had_trend = prior_close >= prior_slow and prior_fast > prior_slow
        pulled_back = prior_low <= prior_fast * (1 + pullback_tolerance)
        recovered = close > prior_high and close > fast > slow
        if had_trend and pulled_back and recovered:
            signals.append(
                PatternSignal(
                    model="trend_pullback_restart",
                    signal_index=index,
                    signal_date=pd.Timestamp(df.at[index, "date"]),
                    trigger_price=close,
                    initial_stop=prior_low,
                    metadata={"ma_fast": fast, "ma_slow": slow},
                )
            )

    return signals


def find_uptrend_bullish_engulfing(
    bars: pd.DataFrame,
    ma_fast: int = 10,
    ma_slow: int = 30,
    pullback_tolerance: float = 0.03,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    for index in range(max(ma_fast, ma_slow) + 1, len(df)):
        prior_index = index - 1
        prior_open = float(df.at[prior_index, "open"])
        prior_close = float(df.at[prior_index, "close"])
        prior_low = float(df.at[prior_index, "low"])
        open_ = float(df.at[index, "open"])
        close = float(df.at[index, "close"])
        low = float(df.at[index, "low"])
        fast = float(df.at[index, "ma_fast"])
        slow = float(df.at[index, "ma_slow"])
        prior_fast = float(df.at[prior_index, "ma_fast"])
        prior_slow = float(df.at[prior_index, "ma_slow"])

        had_trend = prior_fast > prior_slow and prior_close >= prior_slow
        pulled_back = prior_low <= prior_fast * (1 + pullback_tolerance)
        prior_bearish = prior_close < prior_open
        bullish_engulfing = close > open_ and open_ <= prior_close and close >= prior_open
        restored_trend = close > fast > slow
        if had_trend and pulled_back and prior_bearish and bullish_engulfing and restored_trend:
            signals.append(
                PatternSignal(
                    model="uptrend_bullish_engulfing",
                    signal_index=index,
                    signal_date=pd.Timestamp(df.at[index, "date"]),
                    trigger_price=close,
                    initial_stop=min(prior_low, low),
                    metadata={"ma_fast": fast, "ma_slow": slow},
                )
            )

    return signals


def find_consolidation_breakouts(
    bars: pd.DataFrame,
    uptrend_lookback: int = 12,
    consolidation_bars: int = 5,
    min_uptrend_return: float = 0.20,
    max_range_pct: float = 0.10,
    volume_multiplier: float = 1.3,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    signals: list[PatternSignal] = []
    first_index = uptrend_lookback + consolidation_bars

    for index in range(first_index, len(df)):
        trend_window = df.iloc[index - consolidation_bars - uptrend_lookback : index - consolidation_bars]
        consolidation = df.iloc[index - consolidation_bars : index]
        trend_return = float(trend_window.iloc[-1]["close"]) / float(trend_window.iloc[0]["close"]) - 1
        consolidation_high = float(consolidation["high"].max())
        consolidation_low = float(consolidation["low"].min())
        consolidation_mid = float(consolidation["close"].mean())
        range_pct = (consolidation_high - consolidation_low) / consolidation_mid
        avg_volume = float(consolidation["volume"].mean())
        close = float(df.at[index, "close"])
        volume = float(df.at[index, "volume"])

        if (
            trend_return >= min_uptrend_return
            and range_pct <= max_range_pct
            and close > consolidation_high
            and volume > avg_volume * volume_multiplier
        ):
            signals.append(
                PatternSignal(
                    model="consolidation_breakout",
                    signal_index=index,
                    signal_date=pd.Timestamp(df.at[index, "date"]),
                    trigger_price=close,
                    initial_stop=consolidation_low,
                    metadata={"trend_return": trend_return, "range_pct": range_pct},
                )
            )

    return signals


def find_breakout_pullback_restarts(
    bars: pd.DataFrame,
    box_bars: int = 20,
    breakout_volume_multiplier: float = 1.5,
    pullback_bars: int = 4,
    max_pullback_pct: float = 0.08,
    ma_fast: int = 20,
    ma_slow: int = 40,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    first_breakout = max(box_bars, ma_slow)
    for breakout_index in range(first_breakout, len(df) - 1):
        prior_box = df.iloc[breakout_index - box_bars : breakout_index]
        box_high = float(prior_box["high"].max())
        box_low = float(prior_box["low"].min())
        avg_volume = float(prior_box["volume"].mean())
        breakout_close = float(df.at[breakout_index, "close"])
        breakout_volume = float(df.at[breakout_index, "volume"])
        fast = float(df.at[breakout_index, "ma_fast"])
        slow = float(df.at[breakout_index, "ma_slow"])

        is_breakout = breakout_close > box_high
        has_volume = breakout_volume > avg_volume * breakout_volume_multiplier
        has_trend = breakout_close > fast > slow
        if not (is_breakout and has_volume and has_trend):
            continue

        pullback_limit = box_high * (1 - max_pullback_pct)
        latest_index = min(len(df), breakout_index + pullback_bars + 1)
        for restart_index in range(breakout_index + 1, latest_index):
            pullback = df.iloc[breakout_index + 1 : restart_index + 1]
            pullback_low = float(pullback["low"].min())
            restart_close = float(df.at[restart_index, "close"])
            restart_high = float(df.at[restart_index, "high"])
            held_breakout_zone = pullback_low >= max(box_low, pullback_limit)
            recovered = restart_close > breakout_close and restart_close >= float(pullback["close"].max())
            if held_breakout_zone and recovered and restart_high > box_high:
                signals.append(
                    PatternSignal(
                        model="breakout_pullback_restart",
                        signal_index=restart_index,
                        signal_date=pd.Timestamp(df.at[restart_index, "date"]),
                        trigger_price=restart_close,
                        initial_stop=pullback_low,
                        metadata={
                            "box_high": box_high,
                            "breakout_index": float(breakout_index),
                            "pullback_low": pullback_low,
                        },
                    )
                )
                break

    return signals


def find_stage_high_breakout_retests(
    bars: pd.DataFrame,
    lookback_bars: int = 60,
    min_base_bars: int = 30,
    max_retest_bars: int = 10,
    breakout_buffer: float = 0.005,
    support_tolerance: float = 0.015,
    min_pullback_pct: float = 0.03,
    max_extension_pct: float = 0.08,
    ma_fast: int = 20,
    ma_slow: int = 60,
    first_retest_only: bool = True,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    first_breakout = max(lookback_bars, min_base_bars, ma_slow)
    for breakout_index in range(first_breakout, len(df) - 1):
        prior_window = df.iloc[breakout_index - lookback_bars : breakout_index]
        base_window = df.iloc[breakout_index - min_base_bars : breakout_index]
        prior_high = float(prior_window["high"].max())
        base_low = float(base_window["low"].min())
        breakout_close = float(df.at[breakout_index, "close"])
        breakout_high = float(df.at[breakout_index, "high"])
        breakout_low = float(df.at[breakout_index, "low"])
        fast = float(df.at[breakout_index, "ma_fast"])
        slow = float(df.at[breakout_index, "ma_slow"])

        broke_stage_high = breakout_close >= prior_high * (1 + breakout_buffer)
        had_trend = breakout_close > fast > slow
        if not (broke_stage_high and had_trend):
            continue

        latest_index = min(len(df), breakout_index + max_retest_bars + 1)
        for retest_index in range(breakout_index + 1, latest_index):
            retest = df.iloc[breakout_index + 1 : retest_index + 1]
            pullback_low = float(retest["low"].min())
            latest_close = float(df.at[retest_index, "close"])
            latest_low = float(df.at[retest_index, "low"])
            highest_after_breakout = float(df.iloc[breakout_index : retest_index + 1]["high"].max())

            held_prior_high = pullback_low >= prior_high * (1 - support_tolerance)
            did_pullback = latest_low <= highest_after_breakout * (1 - min_pullback_pct)
            not_too_extended = latest_close <= prior_high * (1 + max_extension_pct)
            close_above_support = latest_close >= prior_high * (1 - support_tolerance)
            if held_prior_high and did_pullback and not_too_extended and close_above_support:
                signals.append(
                    PatternSignal(
                        model="stage_high_breakout_retest",
                        signal_index=retest_index,
                        signal_date=pd.Timestamp(df.at[retest_index, "date"]),
                        trigger_price=latest_close,
                        initial_stop=prior_high * (1 - support_tolerance),
                        metadata={
                            "breakout_index": float(breakout_index),
                            "breakout_date_ordinal": float(pd.Timestamp(df.at[breakout_index, "date"]).toordinal()),
                            "prior_high": prior_high,
                            "breakout_close": breakout_close,
                            "breakout_high": breakout_high,
                            "breakout_low": breakout_low,
                            "pullback_low": pullback_low,
                            "days_since_breakout": float(retest_index - breakout_index),
                            "distance_to_prior_high": latest_close / prior_high - 1,
                            "drawdown_from_breakout_high": pullback_low / highest_after_breakout - 1,
                            "base_low": base_low,
                            "ma_fast": fast,
                            "ma_slow": slow,
                        },
                    )
                )
                if first_retest_only:
                    break

    return signals


def find_jianghua_acceleration_retests(
    bars: pd.DataFrame,
    structure_lookback_bars: int = 60,
    min_base_bars: int = 30,
    max_peak_bars: int = 5,
    max_retest_bars: int = 10,
    breakout_buffer: float = 0.005,
    support_tolerance: float = 0.02,
    min_flagpole_pct: float = 0.15,
    max_flagpole_pct: float = 0.45,
    min_peak_drawdown_pct: float = 0.10,
    max_peak_drawdown_pct: float = 0.28,
    min_close_above_support_pct: float = 0.02,
    max_close_above_support_pct: float = 0.09,
    min_breakout_volume_ratio: float = 1.5,
    max_pullback_volume_ratio: float = 0.8,
    ma_fast: int = 20,
    ma_slow: int = 60,
    first_retest_only: bool = True,
) -> list[PatternSignal]:
    df = _prepare_bars(bars)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    signals: list[PatternSignal] = []

    first_breakout = max(structure_lookback_bars, min_base_bars, ma_slow)
    for breakout_index in range(first_breakout, len(df) - 2):
        prior_window = df.iloc[breakout_index - structure_lookback_bars : breakout_index]
        base_window = df.iloc[breakout_index - min_base_bars : breakout_index]
        structure_high = float(prior_window["close"].max())
        base_low = float(base_window["low"].min())
        breakout_close = float(df.at[breakout_index, "close"])
        fast = float(df.at[breakout_index, "ma_fast"])
        slow = float(df.at[breakout_index, "ma_slow"])
        prior_avg_volume = float(prior_window.tail(20)["volume"].mean())

        broke_structure_high = breakout_close >= structure_high * (1 + breakout_buffer)
        had_trend = breakout_close > fast > slow
        if not (broke_structure_high and had_trend and prior_avg_volume > 0):
            continue

        latest_peak_search_index = min(len(df), breakout_index + max_peak_bars + 1)
        latest_retest_index = min(len(df), breakout_index + max_retest_bars + 1)
        for retest_index in range(breakout_index + 2, latest_retest_index):
            impulse = df.iloc[breakout_index: min(retest_index, latest_peak_search_index)]
            if impulse.empty:
                continue
            peak_index = int(impulse["high"].idxmax())
            if peak_index <= breakout_index or peak_index >= retest_index:
                continue

            peak_high = float(df.at[peak_index, "high"])
            flagpole_pct = peak_high / structure_high - 1
            if not (min_flagpole_pct <= flagpole_pct <= max_flagpole_pct):
                continue

            retest = df.iloc[peak_index + 1 : retest_index + 1]
            if retest.empty:
                continue
            all_after_breakout = df.iloc[breakout_index + 1 : retest_index + 1]
            pullback_low = float(all_after_breakout["low"].min())
            latest_close = float(df.at[retest_index, "close"])
            peak_drawdown_pct = 1 - pullback_low / peak_high
            close_above_support_pct = latest_close / structure_high - 1
            impulse_volume = float(df.iloc[breakout_index : peak_index + 1]["volume"].mean())
            pullback_volume = float(retest["volume"].mean())

            held_structure_high = pullback_low >= structure_high * (1 - support_tolerance)
            drawdown_ok = min_peak_drawdown_pct <= peak_drawdown_pct <= max_peak_drawdown_pct
            location_ok = min_close_above_support_pct <= close_above_support_pct <= max_close_above_support_pct
            volume_ok = (
                impulse_volume >= prior_avg_volume * min_breakout_volume_ratio
                and pullback_volume <= impulse_volume * max_pullback_volume_ratio
            )
            if not (held_structure_high and drawdown_ok and location_ok and volume_ok):
                continue

            similarity = _jianghua_similarity_score(
                flagpole_pct=flagpole_pct,
                peak_drawdown_pct=peak_drawdown_pct,
                close_above_support_pct=close_above_support_pct,
                days_since_breakout=retest_index - breakout_index,
                pullback_volume_ratio=pullback_volume / impulse_volume if impulse_volume else 9.99,
            )
            signals.append(
                PatternSignal(
                    model="jianghua_acceleration_retest",
                    signal_index=retest_index,
                    signal_date=pd.Timestamp(df.at[retest_index, "date"]),
                    trigger_price=latest_close,
                    initial_stop=structure_high * (1 - support_tolerance),
                    metadata={
                        "breakout_index": float(breakout_index),
                        "peak_index": float(peak_index),
                        "structure_high": structure_high,
                        "base_low": base_low,
                        "breakout_close": breakout_close,
                        "breakout_high": float(df.at[breakout_index, "high"]),
                        "peak_high": peak_high,
                        "pullback_low": pullback_low,
                        "days_since_breakout": float(retest_index - breakout_index),
                        "days_since_peak": float(retest_index - peak_index),
                        "flagpole_pct": flagpole_pct,
                        "peak_drawdown_pct": peak_drawdown_pct,
                        "close_above_support_pct": close_above_support_pct,
                        "prior_avg_volume": prior_avg_volume,
                        "impulse_volume": impulse_volume,
                        "pullback_volume": pullback_volume,
                        "pullback_volume_ratio": pullback_volume / impulse_volume if impulse_volume else 0.0,
                        "similarity_score": similarity,
                        "ma_fast": fast,
                        "ma_slow": slow,
                    },
                )
            )
            if first_retest_only:
                break

    return signals


def _jianghua_similarity_score(
    flagpole_pct: float,
    peak_drawdown_pct: float,
    close_above_support_pct: float,
    days_since_breakout: int,
    pullback_volume_ratio: float,
) -> float:
    targets = {
        "flagpole_pct": 0.27,
        "peak_drawdown_pct": 0.19,
        "close_above_support_pct": 0.06,
        "days_since_breakout": 7.0,
        "pullback_volume_ratio": 0.75,
    }
    tolerances = {
        "flagpole_pct": 0.14,
        "peak_drawdown_pct": 0.10,
        "close_above_support_pct": 0.04,
        "days_since_breakout": 5.0,
        "pullback_volume_ratio": 0.35,
    }
    values = {
        "flagpole_pct": flagpole_pct,
        "peak_drawdown_pct": peak_drawdown_pct,
        "close_above_support_pct": close_above_support_pct,
        "days_since_breakout": float(days_since_breakout),
        "pullback_volume_ratio": pullback_volume_ratio,
    }
    scores = [
        max(0.0, 1 - abs(values[key] - targets[key]) / tolerances[key])
        for key in targets
    ]
    return float(sum(scores) / len(scores) * 100)


def find_signals_for_model(model_name: str, bars: pd.DataFrame, params: dict[str, object]) -> list[PatternSignal]:
    if model_name == "weekly_box_breakout":
        return find_weekly_box_breakouts(bars, **params)
    if model_name == "trend_pullback_restart":
        return find_trend_pullback_restarts(bars, **params)
    if model_name == "uptrend_bullish_engulfing":
        return find_uptrend_bullish_engulfing(bars, **params)
    if model_name == "consolidation_breakout":
        return find_consolidation_breakouts(bars, **params)
    if model_name == "breakout_pullback_restart":
        return find_breakout_pullback_restarts(bars, **params)
    if model_name == "stage_high_breakout_retest":
        return find_stage_high_breakout_retests(bars, **params)
    if model_name == "jianghua_acceleration_retest":
        return find_jianghua_acceleration_retests(bars, **params)
    raise ValueError(f"unknown model: {model_name}")


def filter_signals_by_market_trend(
    signals: list[PatternSignal],
    market_bars: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 40,
) -> list[PatternSignal]:
    market = _prepare_bars(market_bars)
    market["ma_fast"] = market["close"].rolling(ma_fast).mean()
    market["ma_slow"] = market["close"].rolling(ma_slow).mean()
    filtered: list[PatternSignal] = []

    for signal in signals:
        history = market[market["date"] <= pd.Timestamp(signal.signal_date)]
        if history.empty:
            continue
        row = history.iloc[-1]
        close = float(row["close"])
        fast = float(row["ma_fast"])
        slow = float(row["ma_slow"])
        if close > fast > slow:
            filtered.append(signal)

    return filtered


def run_research(args: argparse.Namespace) -> list[Path]:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - covered by runtime environment check
        raise RuntimeError("akshare is required for data fetching") from exc

    end_date = args.end_date or date.today().strftime("%Y%m%d")
    universe = ak.stock_info_a_code_name()
    universe = universe[~universe["name"].astype(str).str.upper().str.contains("ST", na=False)].reset_index(drop=True)
    if args.offset:
        universe = universe.iloc[args.offset :].reset_index(drop=True)
    if args.limit:
        universe = universe.head(args.limit)

    model_names = MODEL_NAMES if args.model == "all" else [args.model]
    trades_by_model: dict[str, list[CompletedTrade]] = {model: [] for model in model_names}
    failures: list[dict[str, str]] = []

    total_stocks = len(universe)
    for position, (_, stock) in enumerate(universe.iterrows(), start=1):
        code = str(stock.get("code", "")).zfill(6)
        name = str(stock.get("name", code))
        if not code:
            continue
        print(f"research {position}/{total_stocks} {code} {name}", flush=True)
        try:
            weekly = _fetch_weekly_bars(ak, code, args.start_date, end_date, Path(args.cache_dir))
            if len(weekly) < 60:
                failures.append({"code": code, "name": name, "reason": "insufficient_weekly_bars"})
                continue

            for model_name in model_names:
                signals = find_signals_for_model(model_name, weekly, _default_model_params(model_name))
                trades = backtest_signals(
                    code=code,
                    name=name,
                    bars=weekly,
                    signals=signals,
                    max_holding_bars=26,
                    max_initial_loss=0.10,
                    trailing_drawdown=0.20,
                )
                if args.fee_rate or args.slippage_rate:
                    trades = [_with_transaction_costs(trade, args.fee_rate, args.slippage_rate) for trade in trades]
                trades_by_model[model_name].extend(trades)
        except Exception as exc:  # pragma: no cover - network/upstream dependent
            failures.append({"code": code, "name": name, "reason": str(exc)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.start_date}_{end_date}_limit{args.limit or 'all'}"
    output_paths: list[Path] = []
    for model_name, trades in trades_by_model.items():
        csv_path = output_dir / f"{model_name}_trades_{stamp}.csv"
        md_path = output_dir / f"{model_name}_summary_{stamp}.md"
        _trades_frame(trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        md_path.write_text(_summary_markdown(model_name, args, trades, failures), encoding="utf-8")
        output_paths.extend([csv_path, md_path])
    return output_paths


def run_sweep(args: argparse.Namespace) -> list[Path]:
    cache_dir = Path(args.cache_dir)
    end_date = args.end_date or date.today().strftime("%Y%m%d")
    cache_files = sorted(cache_dir.glob(f"*_{args.start_date}_{end_date}_weekly.csv"))
    if args.offset:
        cache_files = cache_files[args.offset :]
    if args.limit:
        cache_files = cache_files[: args.limit]
    if not cache_files:
        raise RuntimeError(f"no cached weekly bars found in {cache_dir}; run research mode first")

    cached_bars = [(path.name.split("_")[0], normalize_price_bars(pd.read_csv(path))) for path in cache_files]
    model_names = MODEL_NAMES if args.model == "all" else [args.model]
    results: list[SweepResult] = []
    grids = [(model_name, params) for model_name in model_names for params in _sweep_param_grid(model_name)]

    for index, (model_name, params) in enumerate(grids, start=1):
        print(f"sweep {index}/{len(grids)} {model_name} {params}", flush=True)
        signal_params, trade_params = _split_sweep_params(model_name, params)
        trades: list[CompletedTrade] = []
        for code, bars in cached_bars:
            signals = find_signals_for_model(model_name, bars, signal_params)
            model_trades = backtest_signals(
                code=code,
                name=code,
                bars=bars,
                signals=signals,
                max_holding_bars=int(trade_params["max_holding_bars"]),
                max_initial_loss=float(trade_params["max_initial_loss"]),
                trailing_drawdown=float(trade_params["trailing_drawdown"]),
            )
            if args.fee_rate or args.slippage_rate:
                model_trades = [_with_transaction_costs(trade, args.fee_rate, args.slippage_rate) for trade in model_trades]
            trades.extend(model_trades)
        results.append(SweepResult(model_name, params, summarize_trades(trades)))

    ranked = rank_sweep_results(results, min_trades=args.min_trades)[: args.sweep_top_n]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_{args.model}_{args.start_date}_{end_date}_limit{args.limit or 'all'}.csv"
    _sweep_results_frame(ranked).to_csv(output_path, index=False, encoding="utf-8-sig")
    return [output_path]


def _sweep_param_grid(model_name: str) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    if model_name == "weekly_box_breakout":
        for box_bars in [10, 20, 30]:
            for volume_multiplier in [1.5, 2.0]:
                for max_holding_bars in [13, 26]:
                    for max_initial_loss in [0.08, 0.10]:
                        configs.append(
                            {
                                "box_bars": box_bars,
                                "volume_multiplier": volume_multiplier,
                                "ma_fast": 20,
                                "ma_slow": 40,
                                "max_holding_bars": max_holding_bars,
                                "max_initial_loss": max_initial_loss,
                                "trailing_drawdown": 0.20,
                            }
                        )
        return configs
    if model_name == "trend_pullback_restart":
        for ma_fast, ma_slow in [(10, 30), (20, 40)]:
            for pullback_tolerance in [0.02, 0.03]:
                for max_holding_bars in [13, 26]:
                    for max_initial_loss in [0.08, 0.10]:
                        configs.append(
                            {
                                "ma_fast": ma_fast,
                                "ma_slow": ma_slow,
                                "pullback_tolerance": pullback_tolerance,
                                "max_holding_bars": max_holding_bars,
                                "max_initial_loss": max_initial_loss,
                                "trailing_drawdown": 0.20,
                            }
                        )
        return configs
    if model_name == "uptrend_bullish_engulfing":
        for ma_fast, ma_slow in [(10, 30), (20, 40)]:
            for pullback_tolerance in [0.02, 0.03, 0.05]:
                for max_holding_bars in [13, 26]:
                    for max_initial_loss in [0.08, 0.10, 0.12]:
                        configs.append(
                            {
                                "ma_fast": ma_fast,
                                "ma_slow": ma_slow,
                                "pullback_tolerance": pullback_tolerance,
                                "max_holding_bars": max_holding_bars,
                                "max_initial_loss": max_initial_loss,
                                "trailing_drawdown": 0.20,
                            }
                        )
        return configs
    if model_name == "consolidation_breakout":
        for consolidation_bars in [4, 5, 8]:
            for max_range_pct in [0.08, 0.10]:
                for volume_multiplier in [1.2, 1.5]:
                    configs.append(
                        {
                            "uptrend_lookback": 12,
                            "consolidation_bars": consolidation_bars,
                            "min_uptrend_return": 0.20,
                            "max_range_pct": max_range_pct,
                            "volume_multiplier": volume_multiplier,
                            "max_holding_bars": 26,
                            "max_initial_loss": 0.10,
                            "trailing_drawdown": 0.20,
                        }
                    )
        return configs
    if model_name == "breakout_pullback_restart":
        for box_bars in [10, 20, 30]:
            for breakout_volume_multiplier in [1.5, 2.0]:
                for pullback_bars in [3, 5]:
                    for max_pullback_pct in [0.06, 0.08, 0.10]:
                        configs.append(
                            {
                                "box_bars": box_bars,
                                "breakout_volume_multiplier": breakout_volume_multiplier,
                                "pullback_bars": pullback_bars,
                                "max_pullback_pct": max_pullback_pct,
                                "ma_fast": 20,
                                "ma_slow": 40,
                                "max_holding_bars": 26,
                                "max_initial_loss": 0.10,
                                "trailing_drawdown": 0.20,
                            }
                        )
        return configs
    if model_name == "stage_high_breakout_retest":
        for lookback_bars in [40, 60, 90]:
            for support_tolerance in [0.01, 0.015, 0.02]:
                for max_retest_bars in [5, 10]:
                    configs.append(
                        {
                            "lookback_bars": lookback_bars,
                            "min_base_bars": 30,
                            "max_retest_bars": max_retest_bars,
                            "breakout_buffer": 0.005,
                            "support_tolerance": support_tolerance,
                            "min_pullback_pct": 0.03,
                            "max_extension_pct": 0.08,
                            "ma_fast": 20,
                            "ma_slow": 60,
                            "max_holding_bars": 60,
                            "max_initial_loss": 0.08,
                            "trailing_drawdown": 0.18,
                        }
                    )
        return configs
    if model_name == "jianghua_acceleration_retest":
        for min_flagpole_pct in [0.15, 0.20]:
            for min_peak_drawdown_pct in [0.08, 0.10]:
                for support_tolerance in [0.015, 0.02]:
                    configs.append(
                        {
                            "structure_lookback_bars": 60,
                            "min_base_bars": 30,
                            "max_peak_bars": 5,
                            "max_retest_bars": 10,
                            "breakout_buffer": 0.005,
                            "support_tolerance": support_tolerance,
                            "min_flagpole_pct": min_flagpole_pct,
                            "max_flagpole_pct": 0.45,
                            "min_peak_drawdown_pct": min_peak_drawdown_pct,
                            "max_peak_drawdown_pct": 0.28,
                            "min_close_above_support_pct": 0.02,
                            "max_close_above_support_pct": 0.09,
                            "min_breakout_volume_ratio": 1.5,
                            "max_pullback_volume_ratio": 0.8,
                            "ma_fast": 20,
                            "ma_slow": 60,
                            "max_holding_bars": 60,
                            "max_initial_loss": 0.08,
                            "trailing_drawdown": 0.18,
                        }
                    )
        return configs
    raise ValueError(f"unknown model: {model_name}")


def _split_sweep_params(model_name: str, params: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    trade_keys = {"max_holding_bars", "max_initial_loss", "trailing_drawdown"}
    trade_params = {key: params[key] for key in trade_keys}
    signal_params = {key: value for key, value in params.items() if key not in trade_keys}
    return signal_params, trade_params


def _sweep_results_frame(results: list[SweepResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.model, **result.params}
        for key in [
            "trade_count",
            "win_rate",
            "average_return",
            "median_return",
            "average_winner",
            "average_loser",
            "profit_loss_ratio",
            "expectancy",
            "max_drawdown",
        ]:
            row[key] = result.summary.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_weekly_bars(
    ak_module: object,
    code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    if cache_dir is not None:
        cache_path = cache_dir / f"{code}_{start_date}_{end_date}_weekly.csv"
        if cache_path.exists():
            return normalize_price_bars(pd.read_csv(cache_path))

    try:
        raw = _retry(
            lambda: ak_module.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
        )
        daily = normalize_price_bars(raw, source="akshare_a_hist")
        weekly = aggregate_weekly_bars(daily)
    except Exception:
        from darvas_weekly_backtest import fetch_weekly_history_tx

        weekly = normalize_price_bars(fetch_weekly_history_tx(code, start_date, end_date))

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(cache_dir / f"{code}_{start_date}_{end_date}_weekly.csv", index=False, encoding="utf-8-sig")
    return weekly


def _retry(callable_, attempts: int = 3, delay_seconds: float = 1.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return callable_()
        except Exception as exc:  # pragma: no cover - depends on upstream network failures
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (attempt + 1))
    raise last_error


def _default_model_params(model_name: str) -> dict[str, object]:
    if model_name == "weekly_box_breakout":
        return {"box_bars": 20, "volume_multiplier": 1.5, "ma_fast": 20, "ma_slow": 40}
    if model_name == "trend_pullback_restart":
        return {"ma_fast": 10, "ma_slow": 30, "pullback_tolerance": 0.03}
    if model_name == "consolidation_breakout":
        return {
            "uptrend_lookback": 12,
            "consolidation_bars": 5,
            "min_uptrend_return": 0.20,
            "max_range_pct": 0.10,
            "volume_multiplier": 1.3,
        }
    if model_name == "breakout_pullback_restart":
        return {
            "box_bars": 20,
            "breakout_volume_multiplier": 1.8,
            "pullback_bars": 5,
            "max_pullback_pct": 0.08,
            "ma_fast": 20,
            "ma_slow": 40,
        }
    if model_name == "stage_high_breakout_retest":
        return {
            "lookback_bars": 60,
            "min_base_bars": 30,
            "max_retest_bars": 10,
            "breakout_buffer": 0.005,
            "support_tolerance": 0.015,
            "min_pullback_pct": 0.03,
            "max_extension_pct": 0.08,
            "ma_fast": 20,
            "ma_slow": 60,
        }
    if model_name == "jianghua_acceleration_retest":
        return {
            "structure_lookback_bars": 60,
            "min_base_bars": 30,
            "max_peak_bars": 5,
            "max_retest_bars": 10,
            "breakout_buffer": 0.005,
            "support_tolerance": 0.02,
            "min_flagpole_pct": 0.15,
            "max_flagpole_pct": 0.45,
            "min_peak_drawdown_pct": 0.10,
            "max_peak_drawdown_pct": 0.28,
            "min_close_above_support_pct": 0.02,
            "max_close_above_support_pct": 0.09,
            "min_breakout_volume_ratio": 1.5,
            "max_pullback_volume_ratio": 0.8,
            "ma_fast": 20,
            "ma_slow": 60,
        }
    if model_name == "uptrend_bullish_engulfing":
        return {"ma_fast": 10, "ma_slow": 30, "pullback_tolerance": 0.03}
    raise ValueError(f"unknown model: {model_name}")


def _with_transaction_costs(trade: CompletedTrade, fee_rate: float, slippage_rate: float) -> CompletedTrade:
    round_trip_cost = 2 * (float(fee_rate) + float(slippage_rate))
    net_return = trade.realized_return - round_trip_cost
    return CompletedTrade(
        code=trade.code,
        name=trade.name,
        signal_date=trade.signal_date,
        entry_date=trade.entry_date,
        entry_price=trade.entry_price,
        exit_date=trade.exit_date,
        exit_price=trade.entry_price * (1 + net_return),
        exit_reason=trade.exit_reason,
        realized_return=net_return,
        holding_bars=trade.holding_bars,
        max_high=trade.max_high,
    )


def _trades_frame(trades: list[CompletedTrade]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": trade.code,
                "name": trade.name,
                "signal_date": trade.signal_date.strftime("%Y-%m-%d"),
                "entry_date": trade.entry_date.strftime("%Y-%m-%d"),
                "entry_price": trade.entry_price,
                "exit_date": trade.exit_date.strftime("%Y-%m-%d"),
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "realized_return": trade.realized_return,
                "holding_bars": trade.holding_bars,
                "max_high": trade.max_high,
            }
            for trade in trades
        ]
    )


def _summary_markdown(
    model_name: str,
    args: argparse.Namespace,
    trades: list[CompletedTrade],
    failures: list[dict[str, str]],
) -> str:
    summary = summarize_trades(trades)
    lines = [
        f"# {model_name} Research Summary",
        "",
        "## Parameters",
        "",
        f"- start_date: {args.start_date}",
        f"- end_date: {args.end_date or date.today().strftime('%Y%m%d')}",
        f"- universe: {args.universe}",
        f"- limit: {args.limit or 'all'}",
        f"- fee_rate: {args.fee_rate}",
        f"- slippage_rate: {args.slippage_rate}",
        "",
        "## Overall",
        "",
        f"- trade_count: {summary['trade_count']}",
        f"- win_rate: {_format_pct(summary['win_rate'])}",
        f"- average_return: {_format_pct(summary['average_return'])}",
        f"- median_return: {_format_pct(summary['median_return'])}",
        f"- average_winner: {_format_pct(summary['average_winner'])}",
        f"- average_loser: {_format_pct(summary['average_loser'])}",
        f"- profit_loss_ratio: {summary['profit_loss_ratio']:.2f}",
        f"- expectancy: {_format_pct(summary['expectancy'])}",
        f"- max_drawdown: {_format_pct(summary['max_drawdown'])}",
        "",
        "## Exit Reasons",
        "",
    ]
    for reason, count in summary["exit_reason_counts"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Data Issues", "", f"- failed_or_skipped_stocks: {len(failures)}"])
    for failure in failures[:20]:
        lines.append(f"- {failure['code']} {failure['name']}: {failure['reason']}")
    return "\n".join(lines) + "\n"


def _format_pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    df = bars[required].copy()
    df["date"] = pd.to_datetime(df["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    main()
