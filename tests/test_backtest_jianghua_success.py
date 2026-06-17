from __future__ import annotations

import unittest

import pandas as pd

from backtest_jianghua_success import (
    attach_turnover_rate_from_float_share,
    build_market_context_from_frames,
    classify_market_regime,
    evaluate_signal,
    find_jianghua_acceleration_retests_fast,
    load_structural_code_pool,
    market_context_passes_filter,
)
from kline_model_research import PatternSignal, find_jianghua_acceleration_retests


def bar(day: str, open_: float, high: float, low: float, close: float, volume: float = 1000) -> dict[str, object]:
    return {
        "date": pd.Timestamp(day),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def sample_signal(index: int = 0) -> PatternSignal:
    return PatternSignal(
        model="jianghua_acceleration_retest",
        signal_index=index,
        signal_date=pd.Timestamp("2026-01-02"),
        trigger_price=10.0,
        initial_stop=9.0,
        metadata={
            "breakout_index": 0.0,
            "peak_index": 0.0,
            "structure_high": 10.0,
            "breakout_close": 10.5,
            "breakout_high": 10.8,
            "peak_high": 12.0,
            "pullback_low": 10.1,
            "days_since_breakout": 5.0,
            "days_since_peak": 3.0,
            "flagpole_pct": 0.20,
            "peak_drawdown_pct": 0.15,
            "close_above_support_pct": 0.05,
            "prior_avg_volume": 1000.0,
            "impulse_volume": 1800.0,
            "pullback_volume": 1200.0,
            "pullback_volume_ratio": 0.6667,
            "similarity_score": 80.0,
        },
    )


class JianghuaSuccessBacktestTests(unittest.TestCase):
    def test_attach_turnover_rate_uses_tdx_share_volume_and_float_share_10k_units(self) -> None:
        rows = pd.DataFrame([bar("2026-01-02", 10.0, 10.2, 9.8, 10.0, volume=1_000_000)])

        result = attach_turnover_rate_from_float_share(rows, "000001", {"000001": 1000.0})

        self.assertAlmostEqual(10.0, result.at[0, "turnover_rate"])

    def test_fast_signal_finder_matches_current_jianghua_detector_on_fixture(self) -> None:
        rows = []
        for index in range(60):
            close = 10.0 + index * 0.03
            row = bar(str(pd.Timestamp("2025-01-02") + pd.Timedelta(days=index)), close, close + 0.1, close - 0.1, close, 1000)
            row["turnover_rate"] = 2.0
            rows.append(row)
        rows.extend(
            [
                {**bar("2025-03-03", 12.10, 12.40, 12.00, 12.35, 1800), "turnover_rate": 2.0},
                {**bar("2025-03-04", 12.40, 15.50, 12.30, 15.00, 2400), "turnover_rate": 2.0},
                {**bar("2025-03-05", 14.80, 15.10, 13.00, 13.20, 1300), "turnover_rate": 2.0},
                {**bar("2025-03-06", 13.10, 13.50, 12.60, 12.90, 1100), "turnover_rate": 2.0},
                {**bar("2025-03-07", 12.80, 13.00, 12.30, 12.75, 900), "turnover_rate": 2.0},
            ]
        )
        kwargs = {
            "structure_lookback_bars": 40,
            "min_base_bars": 20,
            "max_peak_bars": 5,
            "max_retest_bars": 8,
            "max_days_since_peak": 5,
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
            "ma_fast": 5,
            "ma_slow": 20,
        }

        slow = find_jianghua_acceleration_retests(
            pd.DataFrame(rows),
            first_retest_only=True,
            min_platform_turnover_pct=0,
            min_platform_amplitude_pct=0,
            max_platform_amplitude_pct=999,
            max_platform_gain_pct=999,
            **kwargs,
        )
        fast = find_jianghua_acceleration_retests_fast(
            pd.DataFrame(rows),
            min_platform_turnover_pct=0,
            min_platform_amplitude_pct=0,
            max_platform_amplitude_pct=999,
            max_platform_gain_pct=999,
            **kwargs,
        )

        self.assertEqual(len(slow), len(fast))
        self.assertEqual(slow[0].signal_date, fast[0].signal_date)
        self.assertAlmostEqual(slow[0].trigger_price, fast[0].trigger_price)
        self.assertAlmostEqual(slow[0].metadata["similarity_score"], fast[0].metadata["similarity_score"])

    def test_fast_signal_finder_requires_long_quiet_high_turnover_platform(self) -> None:
        rows = []
        for index in range(120):
            close = 10.0 + index * 0.01
            row = bar(str(pd.Timestamp("2025-01-02") + pd.Timedelta(days=index)), close, close + 0.1, close - 0.1, close, 1000)
            row["turnover_rate"] = 1.0
            rows.append(row)
        rows.extend(
            [
                {**bar("2025-05-02", 11.30, 11.60, 11.20, 11.55, 1800), "turnover_rate": 2.0},
                {**bar("2025-05-03", 11.60, 14.10, 11.50, 13.80, 2400), "turnover_rate": 2.0},
                {**bar("2025-05-04", 13.70, 13.90, 12.20, 12.40, 1300), "turnover_rate": 2.0},
                {**bar("2025-05-05", 12.30, 12.80, 11.70, 12.00, 1100), "turnover_rate": 2.0},
                {**bar("2025-05-06", 12.10, 12.35, 11.60, 11.95, 900), "turnover_rate": 2.0},
            ]
        )

        accepted = find_jianghua_acceleration_retests_fast(
            pd.DataFrame(rows),
            structure_lookback_bars=120,
            min_base_bars=120,
            min_platform_amplitude_pct=0,
            ma_fast=5,
            ma_slow=20,
            min_platform_turnover_pct=100,
            max_platform_amplitude_pct=100,
            max_platform_gain_pct=30,
        )
        low_turnover = pd.DataFrame(rows)
        low_turnover["turnover_rate"] = 0.2

        rejected = find_jianghua_acceleration_retests_fast(
            low_turnover,
            structure_lookback_bars=120,
            min_base_bars=120,
            min_platform_amplitude_pct=0,
            ma_fast=5,
            ma_slow=20,
            min_platform_turnover_pct=100,
            max_platform_amplitude_pct=100,
            max_platform_gain_pct=30,
        )

        self.assertEqual(1, len(accepted))
        self.assertEqual([], rejected)
        self.assertGreaterEqual(accepted[0].metadata["platform_turnover_pct"], 100)
        self.assertLessEqual(accepted[0].metadata["platform_gain_pct"], 30)

    def test_fast_signal_finder_rejects_slow_peak_retest_when_peak_window_is_tight(self) -> None:
        rows = []
        for index in range(120):
            close = 10.0 + index * 0.01
            row = bar(str(pd.Timestamp("2025-01-02") + pd.Timedelta(days=index)), close, close + 0.1, close - 0.1, close, 1000)
            row["turnover_rate"] = 1.0
            rows.append(row)
        rows.extend(
            [
                {**bar("2025-05-02", 11.30, 11.60, 11.20, 11.55, 1800), "turnover_rate": 2.0},
                {**bar("2025-05-03", 11.60, 15.50, 11.50, 15.00, 2600), "turnover_rate": 2.0},
                {**bar("2025-05-04", 14.80, 15.00, 12.70, 13.10, 1200), "turnover_rate": 2.0},
                {**bar("2025-05-05", 13.00, 13.20, 12.25, 12.40, 1100), "turnover_rate": 2.0},
                {**bar("2025-05-06", 11.90, 12.10, 11.55, 11.80, 900), "turnover_rate": 2.0},
            ]
        )

        loose = find_jianghua_acceleration_retests_fast(
            pd.DataFrame(rows),
            structure_lookback_bars=120,
            min_base_bars=120,
            max_days_since_peak=3,
            min_platform_amplitude_pct=0,
            ma_fast=5,
            ma_slow=20,
        )
        tight = find_jianghua_acceleration_retests_fast(
            pd.DataFrame(rows),
            structure_lookback_bars=120,
            min_base_bars=120,
            max_days_since_peak=2,
            min_platform_amplitude_pct=0,
            ma_fast=5,
            ma_slow=20,
        )

        self.assertEqual(1, len(loose))
        self.assertEqual([], tight)

    def test_fast_signal_finder_accepts_steady_climb_after_mild_breakout(self) -> None:
        rows = []
        for index in range(120):
            day = str(pd.Timestamp("2025-01-02") + pd.Timedelta(days=index))
            close = 10.8 + (index % 20) * 0.03
            rows.append({**bar(day, close, 12.0, 8.8, close, 1000), "turnover_rate": 1.0})
        rows.extend(
            [
                {**bar("2025-05-02", 12.05, 12.25, 11.75, 12.12, 1300), "turnover_rate": 2.0},
                {**bar("2025-05-03", 12.18, 12.55, 11.85, 12.35, 1200), "turnover_rate": 2.0},
                {**bar("2025-05-04", 12.35, 12.95, 12.10, 12.80, 1250), "turnover_rate": 2.0},
                {**bar("2025-05-05", 12.80, 13.35, 12.45, 13.10, 1350), "turnover_rate": 2.0},
                {**bar("2025-05-06", 13.05, 13.25, 12.70, 12.95, 1050), "turnover_rate": 2.0},
            ]
        )

        signals = find_jianghua_acceleration_retests_fast(
            pd.DataFrame(rows),
            structure_lookback_bars=120,
            min_base_bars=120,
            min_platform_amplitude_pct=35,
            max_platform_amplitude_pct=100,
            min_platform_turnover_pct=100,
            max_platform_gain_pct=20,
            ma_fast=5,
            ma_slow=20,
        )

        self.assertTrue(signals)
        self.assertEqual("steady_climb_retest", signals[0].metadata["pattern_subtype"])
        self.assertLess(signals[0].metadata["flagpole_pct"], 0.22)
        self.assertGreaterEqual(signals[0].metadata["breakout_volume_ratio"], 1.1)

    def test_market_context_builds_breadth_rates_and_filter(self) -> None:
        frames = {
            "000001": pd.DataFrame(
                [
                    bar(str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)), 10 + index, 10 + index, 10 + index, 10 + index)
                    for index in range(130)
                ]
            ),
            "000002": pd.DataFrame(
                [
                    bar(str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)), 20 - index * 0.1, 20 - index * 0.1, 20 - index * 0.1, 20 - index * 0.1)
                    for index in range(130)
                ]
            ),
        }

        context = build_market_context_from_frames(frames)
        latest = context.iloc[-1].to_dict()

        self.assertEqual(2, latest["stock_count"])
        self.assertAlmostEqual(0.5, latest["advance_rate"])
        self.assertAlmostEqual(0.5, latest["above_ma20_rate"])
        self.assertGreater(latest["ret120_p95"], latest["ret120_median"])
        self.assertTrue(market_context_passes_filter(latest, 0.45, 0.45, 0.0))
        self.assertFalse(market_context_passes_filter(latest, 0.60, 0.45, 0.0, allow_structural_bull=False))

    def test_market_regime_allows_structural_bull_when_breadth_is_weak_but_leaders_are_strong(self) -> None:
        context = {
            "advance_rate": 0.35,
            "above_ma20_rate": 0.20,
            "above_ma60_rate": 0.25,
            "ret120_median": -0.08,
            "ret120_p95": 0.55,
        }

        self.assertEqual("structural_bull", classify_market_regime(context, 0.45, 0.35, 0.0, True, 0.30, 0.35))
        self.assertFalse(market_context_passes_filter(context, 0.45, 0.35, 0.0, code="000001"))
        self.assertTrue(
            market_context_passes_filter(
                context,
                0.45,
                0.35,
                0.0,
                code="000001",
                structural_code_pool={"000001"},
            )
        )

    def test_market_regime_rejects_weak_market_without_broad_or_structural_strength(self) -> None:
        context = {
            "advance_rate": 0.35,
            "above_ma20_rate": 0.20,
            "above_ma60_rate": 0.25,
            "ret120_median": -0.08,
            "ret120_p95": 0.12,
        }

        self.assertEqual("weak_or_no_trend", classify_market_regime(context, 0.45, 0.35, 0.0, True, 0.30, 0.35))
        self.assertFalse(market_context_passes_filter(context, 0.45, 0.35, 0.0))

    def test_load_structural_code_pool_reads_code_column(self) -> None:
        path = "data/cache/test_structural_pool.csv"
        pd.DataFrame(
            [
                {"code": "300750", "board_name": "AI"},
                {"code": 688256, "board_name": "算力"},
            ]
        ).to_csv(path, index=False)

        self.assertEqual({"300750", "688256"}, load_structural_code_pool(path))

    def test_evaluate_signal_marks_success_when_30pct_high_is_reached_within_20_bars(self) -> None:
        rows = [bar("2026-01-02", 10.0, 10.2, 9.8, 10.0)]
        for offset in range(1, 25):
            high = 13.2 if offset == 8 else 11.0
            rows.append(bar(f"2026-01-{2 + offset:02d}", 10.0, high, 9.8, 10.5))

        result = evaluate_signal("000001", "测试股票", pd.DataFrame(rows), sample_signal(), 0.30, 20, 40)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["success_20d_30pct"])
        self.assertEqual(8, result["bars_to_30pct"])
        self.assertEqual("2026-01-10", result["success_date"])

    def test_evaluate_signal_returns_none_without_next_day_entry(self) -> None:
        rows = [bar("2026-01-02", 10.0, 10.2, 9.8, 10.0)]

        result = evaluate_signal("000001", "测试股票", pd.DataFrame(rows), sample_signal(), 0.30, 20, 40)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
