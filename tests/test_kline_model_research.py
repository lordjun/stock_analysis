import unittest

import pandas as pd

from kline_model_research import (
    CompletedTrade,
    ExitRule,
    MODEL_NAMES,
    PatternSignal,
    SweepResult,
    TradeSetup,
    _default_model_params,
    build_parser,
    backtest_signals,
    find_consolidation_breakouts,
    find_breakout_pullback_restarts,
    find_signals_for_model,
    find_trend_pullback_restarts,
    find_uptrend_bullish_engulfing,
    find_weekly_box_breakouts,
    filter_signals_by_market_trend,
    aggregate_weekly_bars,
    normalize_price_bars,
    rank_sweep_results,
    simulate_position,
    summarize_trades,
)


def bar(date, open_, high, low, close, volume=1000):
    return {
        "date": pd.Timestamp(date),
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


class KlineModelResearchTests(unittest.TestCase):
    def test_initial_stop_exits_before_time_exit(self):
        future = pd.DataFrame(
            [
                bar("2025-01-06", 100, 104, 97, 102),
                bar("2025-01-07", 102, 103, 89, 91),
            ]
        )
        setup = TradeSetup(
            code="000001",
            name="Ping An Bank",
            signal_date=pd.Timestamp("2025-01-03"),
            entry_date=pd.Timestamp("2025-01-06"),
            entry_price=100.0,
            initial_stop=92.0,
            max_holding_bars=20,
            exit_rule=ExitRule.TRAILING_HIGH_DRAWDOWN,
            trailing_drawdown=0.18,
        )

        trade = simulate_position(setup, future)

        self.assertEqual("initial_stop", trade.exit_reason)
        self.assertAlmostEqual(-0.08, trade.realized_return)

    def test_trailing_stop_lets_profit_run_then_exits_positive(self):
        future = pd.DataFrame(
            [
                bar("2025-01-06", 100, 125, 99, 122),
                bar("2025-01-07", 122, 126, 102, 105),
            ]
        )
        setup = TradeSetup(
            code="000001",
            name="Ping An Bank",
            signal_date=pd.Timestamp("2025-01-03"),
            entry_date=pd.Timestamp("2025-01-06"),
            entry_price=100.0,
            initial_stop=92.0,
            max_holding_bars=20,
            exit_rule=ExitRule.TRAILING_HIGH_DRAWDOWN,
            trailing_drawdown=0.18,
        )

        trade = simulate_position(setup, future)

        self.assertEqual("trailing_stop", trade.exit_reason)
        self.assertAlmostEqual(0.0250, trade.realized_return, places=4)

    def test_weekly_box_breakout_requires_box_volume_and_trend(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0, 1000),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0, 1000),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0, 1000),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0, 1000),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0, 1000),
            bar("2025-02-07", 15.5, 16.4, 15.2, 16.0, 1800),
        ]

        signals = find_weekly_box_breakouts(
            pd.DataFrame(rows),
            box_bars=4,
            volume_multiplier=1.5,
            ma_fast=3,
            ma_slow=5,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("weekly_box_breakout", signals[0].model)
        self.assertEqual(pd.Timestamp("2025-02-07"), signals[0].signal_date)
        self.assertAlmostEqual(16.0, signals[0].trigger_price)
        self.assertAlmostEqual(10.6, signals[0].initial_stop)

    def test_trend_pullback_restart_requires_prior_pullback_and_recovery(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0, 1000),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0, 1000),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0, 1000),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0, 1000),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0, 1000),
            bar("2025-02-07", 13.8, 14.1, 12.8, 13.5, 900),
            bar("2025-02-14", 13.7, 15.2, 13.6, 15.0, 1300),
        ]

        signals = find_trend_pullback_restarts(
            pd.DataFrame(rows),
            ma_fast=3,
            ma_slow=5,
            pullback_tolerance=0.02,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("trend_pullback_restart", signals[0].model)
        self.assertEqual(pd.Timestamp("2025-02-14"), signals[0].signal_date)
        self.assertAlmostEqual(15.0, signals[0].trigger_price)
        self.assertAlmostEqual(12.8, signals[0].initial_stop)

    def test_uptrend_bullish_engulfing_requires_pullback_and_reversal_candle(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0),
            bar("2025-02-07", 14.0, 14.2, 12.8, 13.0),
            bar("2025-02-14", 12.9, 14.6, 12.7, 14.4),
        ]

        signals = find_uptrend_bullish_engulfing(
            pd.DataFrame(rows),
            ma_fast=3,
            ma_slow=5,
            pullback_tolerance=0.03,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("uptrend_bullish_engulfing", signals[0].model)
        self.assertEqual(pd.Timestamp("2025-02-14"), signals[0].signal_date)
        self.assertAlmostEqual(14.4, signals[0].trigger_price)
        self.assertAlmostEqual(12.7, signals[0].initial_stop)

    def test_consolidation_breakout_requires_uptrend_tight_range_and_volume(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0, 1000),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0, 1000),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0, 1000),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0, 1000),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0, 1000),
            bar("2025-02-07", 14.0, 14.5, 13.8, 14.1, 1000),
            bar("2025-02-14", 14.1, 14.6, 13.9, 14.2, 1000),
            bar("2025-02-21", 14.2, 14.5, 13.8, 14.1, 1000),
            bar("2025-02-28", 14.5, 15.4, 14.4, 15.2, 1500),
        ]

        signals = find_consolidation_breakouts(
            pd.DataFrame(rows),
            uptrend_lookback=5,
            consolidation_bars=3,
            min_uptrend_return=0.20,
            max_range_pct=0.08,
            volume_multiplier=1.3,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("consolidation_breakout", signals[0].model)
        self.assertEqual(pd.Timestamp("2025-02-28"), signals[0].signal_date)
        self.assertAlmostEqual(15.2, signals[0].trigger_price)
        self.assertAlmostEqual(13.8, signals[0].initial_stop)

    def test_breakout_pullback_restart_waits_for_pullback_confirmation(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0, 1000),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0, 1000),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0, 1000),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0, 1000),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0, 1000),
            bar("2025-02-07", 15.0, 16.5, 14.9, 16.0, 1800),
            bar("2025-02-14", 15.6, 16.0, 14.4, 15.0, 900),
            bar("2025-02-21", 15.1, 16.8, 15.0, 16.6, 1400),
        ]

        signals = find_breakout_pullback_restarts(
            pd.DataFrame(rows),
            box_bars=4,
            breakout_volume_multiplier=1.5,
            pullback_bars=3,
            max_pullback_pct=0.08,
            ma_fast=3,
            ma_slow=5,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("breakout_pullback_restart", signals[0].model)
        self.assertEqual(pd.Timestamp("2025-02-21"), signals[0].signal_date)
        self.assertAlmostEqual(16.6, signals[0].trigger_price)
        self.assertAlmostEqual(14.4, signals[0].initial_stop)

    def test_summarize_trades_reports_expectancy_and_distribution(self):
        trades = [
            completed_trade("000001", "2025-01-03", 0.20, "trailing_stop", 6),
            completed_trade("000002", "2025-02-07", 0.10, "time_exit", 5),
            completed_trade("000003", "2025-03-14", -0.08, "initial_stop", 2),
            completed_trade("000004", "2026-01-09", -0.04, "time_exit", 4),
        ]

        summary = summarize_trades(trades)

        self.assertEqual(4, summary["trade_count"])
        self.assertAlmostEqual(0.50, summary["win_rate"])
        self.assertAlmostEqual(0.15, summary["average_winner"])
        self.assertAlmostEqual(-0.06, summary["average_loser"])
        self.assertAlmostEqual(2.5, summary["profit_loss_ratio"])
        self.assertAlmostEqual(0.045, summary["expectancy"])
        self.assertEqual({"time_exit": 2, "trailing_stop": 1, "initial_stop": 1}, summary["exit_reason_counts"])
        self.assertEqual({"1-4": 2, "5-8": 2, "9-13": 0, "14+": 0}, summary["holding_period_distribution"])
        self.assertEqual(2, summary["annual"]["2025"]["wins"])
        self.assertEqual(1, summary["annual"]["2026"]["losses"])

    def test_backtest_signals_enters_next_bar_and_caps_initial_risk(self):
        bars = pd.DataFrame(
            [
                bar("2025-01-03", 10, 11, 9, 10),
                bar("2025-01-10", 12, 14, 11, 13),
                bar("2025-01-17", 13, 16, 12.5, 15),
                bar("2025-01-24", 15, 15.5, 10.7, 11),
            ]
        )
        signals = [
            PatternSignal(
                model="weekly_box_breakout",
                signal_index=0,
                signal_date=pd.Timestamp("2025-01-03"),
                trigger_price=10.0,
                initial_stop=8.0,
                metadata={},
            )
        ]

        trades = backtest_signals(
            code="000001",
            name="Ping An Bank",
            bars=bars,
            signals=signals,
            max_holding_bars=3,
            max_initial_loss=0.10,
            trailing_drawdown=0.20,
        )

        self.assertEqual(1, len(trades))
        self.assertEqual(pd.Timestamp("2025-01-10"), trades[0].entry_date)
        self.assertAlmostEqual(12.0, trades[0].entry_price)
        self.assertEqual("initial_stop", trades[0].exit_reason)
        self.assertAlmostEqual(-0.10, trades[0].realized_return)

    def test_backtest_signals_skips_overlapping_signals_for_same_stock(self):
        bars = pd.DataFrame(
            [
                bar("2025-01-03", 10, 11, 9, 10),
                bar("2025-01-10", 12, 13, 11.5, 12.5),
                bar("2025-01-17", 12.5, 14, 12.0, 13.5),
                bar("2025-01-24", 13.5, 15, 13.0, 14.5),
                bar("2025-01-31", 14.5, 16, 14.0, 15.5),
            ]
        )
        signals = [
            PatternSignal("model", 0, pd.Timestamp("2025-01-03"), 10.0, 9.0, {}),
            PatternSignal("model", 1, pd.Timestamp("2025-01-10"), 12.5, 11.5, {}),
        ]

        trades = backtest_signals(
            code="000001",
            name="Ping An Bank",
            bars=bars,
            signals=signals,
            max_holding_bars=3,
            max_initial_loss=0.10,
            trailing_drawdown=0.20,
        )

        self.assertEqual(1, len(trades))
        self.assertEqual(pd.Timestamp("2025-01-10"), trades[0].entry_date)
        self.assertEqual(pd.Timestamp("2025-01-24"), trades[0].exit_date)

    def test_normalize_price_bars_accepts_akshare_position_columns(self):
        raw = pd.DataFrame(
            [
                ["2025-01-02", "000001", 10, 10.5, 11, 9.8, 1234],
                ["2025-01-03", "000001", 10.5, 10.8, 11.2, 10.4, 2345],
            ],
            columns=["c0", "c1", "c2", "c3", "c4", "c5", "c6"],
        )

        normalized = normalize_price_bars(raw, source="akshare_a_hist")

        self.assertEqual(["date", "open", "high", "low", "close", "volume"], normalized.columns.tolist())
        self.assertEqual(pd.Timestamp("2025-01-02"), normalized.at[0, "date"])
        self.assertAlmostEqual(10.0, normalized.at[0, "open"])
        self.assertAlmostEqual(11.0, normalized.at[0, "high"])
        self.assertAlmostEqual(9.8, normalized.at[0, "low"])
        self.assertAlmostEqual(10.5, normalized.at[0, "close"])
        self.assertAlmostEqual(1234.0, normalized.at[0, "volume"])

    def test_normalize_price_bars_drops_nonpositive_ohlc_rows(self):
        raw = pd.DataFrame(
            [
                bar("2025-01-03", 10.0, 10.5, 9.5, 10.2),
                bar("2025-01-10", 0.0, 10.5, 9.5, 10.2),
                bar("2025-01-17", 10.0, 10.5, 0.0, 10.2),
                bar("2025-01-24", 10.0, 10.5, 9.5, 0.0),
            ]
        )

        normalized = normalize_price_bars(raw)

        self.assertEqual(1, len(normalized))
        self.assertEqual(pd.Timestamp("2025-01-03"), normalized.at[0, "date"])

    def test_aggregate_weekly_bars_uses_weekly_ohlcv(self):
        daily = pd.DataFrame(
            [
                bar("2025-01-02", 10, 11, 9, 10.5, 100),
                bar("2025-01-03", 10.5, 12, 10, 11.5, 200),
                bar("2025-01-06", 11.5, 13, 11, 12.5, 300),
            ]
        )

        weekly = aggregate_weekly_bars(daily)

        self.assertEqual(2, len(weekly))
        self.assertEqual(pd.Timestamp("2025-01-03"), weekly.at[0, "date"])
        self.assertEqual(pd.Timestamp("2025-01-06"), weekly.at[1, "date"])
        self.assertAlmostEqual(10.0, weekly.at[0, "open"])
        self.assertAlmostEqual(12.0, weekly.at[0, "high"])
        self.assertAlmostEqual(9.0, weekly.at[0, "low"])
        self.assertAlmostEqual(11.5, weekly.at[0, "close"])
        self.assertAlmostEqual(300.0, weekly.at[0, "volume"])

    def test_find_signals_for_model_dispatches_with_parameters(self):
        rows = [
            bar("2025-01-03", 10.0, 10.4, 9.8, 10.0, 1000),
            bar("2025-01-10", 10.8, 11.2, 10.6, 11.0, 1000),
            bar("2025-01-17", 11.8, 12.2, 11.6, 12.0, 1000),
            bar("2025-01-24", 12.8, 13.2, 12.6, 13.0, 1000),
            bar("2025-01-31", 13.8, 14.2, 13.6, 14.0, 1000),
            bar("2025-02-07", 15.5, 16.4, 15.2, 16.0, 1800),
        ]

        signals = find_signals_for_model(
            "weekly_box_breakout",
            pd.DataFrame(rows),
            {"box_bars": 4, "volume_multiplier": 1.5, "ma_fast": 3, "ma_slow": 5},
        )

        self.assertEqual(1, len(signals))
        self.assertEqual("weekly_box_breakout", signals[0].model)

    def test_every_listed_model_has_default_params(self):
        for model_name in MODEL_NAMES:
            params = _default_model_params(model_name)
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)

    def test_find_signals_for_model_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "unknown model"):
            find_signals_for_model("not_a_model", pd.DataFrame([]), {})

    def test_filter_signals_by_market_trend_keeps_only_bull_market_signals(self):
        market = pd.DataFrame(
            [
                bar("2025-01-03", 10, 10.5, 9.8, 10.0),
                bar("2025-01-10", 10, 10.5, 9.8, 10.0),
                bar("2025-01-17", 9, 9.5, 8.8, 9.0),
                bar("2025-01-24", 11, 11.5, 10.8, 11.0),
                bar("2025-01-31", 12, 12.5, 11.8, 12.0),
                bar("2025-02-07", 13, 13.5, 12.8, 13.0),
            ]
        )
        signals = [
            PatternSignal("model", 2, pd.Timestamp("2025-01-17"), 9.0, 8.0, {}),
            PatternSignal("model", 5, pd.Timestamp("2025-02-07"), 13.0, 12.0, {}),
        ]

        filtered = filter_signals_by_market_trend(signals, market, ma_fast=2, ma_slow=3)

        self.assertEqual(1, len(filtered))
        self.assertEqual(pd.Timestamp("2025-02-07"), filtered[0].signal_date)

    def test_build_parser_accepts_research_cli_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--start-date",
                "20160101",
                "--end-date",
                "20260602",
                "--model",
                "weekly_box_breakout",
                "--mode",
                "sweep",
                "--limit",
                "30",
                "--offset",
                "10",
                "--output-dir",
                "reports/kline_models",
                "--cache-dir",
                "data/cache/kline_models",
                "--fee-rate",
                "0.0003",
                "--slippage-rate",
                "0.001",
                "--max-workers",
                "4",
                "--sweep-top-n",
                "15",
                "--min-trades",
                "40",
            ]
        )

        self.assertEqual("20160101", args.start_date)
        self.assertEqual("20260602", args.end_date)
        self.assertEqual("weekly_box_breakout", args.model)
        self.assertEqual("sweep", args.mode)
        self.assertEqual(30, args.limit)
        self.assertEqual(10, args.offset)
        self.assertEqual("reports/kline_models", args.output_dir)
        self.assertEqual("data/cache/kline_models", args.cache_dir)
        self.assertAlmostEqual(0.0003, args.fee_rate)
        self.assertAlmostEqual(0.001, args.slippage_rate)
        self.assertEqual(4, args.max_workers)
        self.assertEqual(15, args.sweep_top_n)
        self.assertEqual(40, args.min_trades)

    def test_rank_sweep_results_filters_small_samples_and_sorts(self):
        results = [
            SweepResult("box", {"box_bars": 20}, {"trade_count": 20, "win_rate": 0.80, "expectancy": 0.10}),
            SweepResult("box", {"box_bars": 30}, {"trade_count": 40, "win_rate": 0.55, "expectancy": 0.03}),
            SweepResult("box", {"box_bars": 10}, {"trade_count": 50, "win_rate": 0.55, "expectancy": 0.06}),
            SweepResult("pullback", {"ma_fast": 10}, {"trade_count": 60, "win_rate": 0.48, "expectancy": 0.08}),
        ]

        ranked = rank_sweep_results(results, min_trades=30)

        self.assertEqual(3, len(ranked))
        self.assertEqual({"box_bars": 10}, ranked[0].params)
        self.assertEqual({"box_bars": 30}, ranked[1].params)
        self.assertEqual({"ma_fast": 10}, ranked[2].params)


def completed_trade(code, entry_date, realized_return, exit_reason, holding_bars):
    entry_price = 100.0
    exit_price = entry_price * (1 + realized_return)
    return CompletedTrade(
        code=code,
        name=code,
        signal_date=pd.Timestamp(entry_date) - pd.Timedelta(days=3),
        entry_date=pd.Timestamp(entry_date),
        entry_price=entry_price,
        exit_date=pd.Timestamp(entry_date) + pd.Timedelta(days=holding_bars),
        exit_price=exit_price,
        exit_reason=exit_reason,
        realized_return=realized_return,
        holding_bars=holding_bars,
        max_high=max(entry_price, exit_price),
    )


if __name__ == "__main__":
    unittest.main()
