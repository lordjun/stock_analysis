import unittest

import pandas as pd

from darvas_weekly_backtest import (
    backtest_stock,
    find_signals,
    market_code,
    normalize_daily_columns,
    normalize_tushare_daily,
    normalize_tx_weekly_records,
    simulate_trade,
)


def weekly_bar(date, open_, high, low, close, volume):
    return {
        "date": pd.Timestamp(date),
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


class DarvasWeeklyBacktestTests(unittest.TestCase):
    def test_find_signal_requires_breakout_volume_and_trend(self):
        rows = []
        for i in range(40):
            close = 10 + i * 0.2
            rows.append(weekly_bar(f"2024-01-05", close - 0.1, close + 0.3, close - 0.4, close, 1000))
            rows[-1]["date"] = pd.Timestamp("2024-01-05") + pd.Timedelta(weeks=i)

        box_top = max(row["high"] for row in rows[-20:])
        rows.append(weekly_bar("2024-10-11", box_top + 0.2, box_top + 1.0, box_top - 0.2, box_top + 0.5, 1800))
        df = pd.DataFrame(rows)

        signals = find_signals(df, box_weeks=20, volume_multiplier=1.5)

        self.assertEqual(1, len(signals))
        self.assertAlmostEqual(box_top, signals[0].box_top)
        self.assertAlmostEqual(box_top + 0.5, signals[0].entry_price)

    def test_fixed_stop_exits_at_ten_percent_loss(self):
        future = pd.DataFrame(
            [
                weekly_bar("2025-01-10", 100, 106, 94, 103, 1000),
                weekly_bar("2025-01-17", 103, 104, 89, 91, 1000),
            ]
        )

        trade = simulate_trade("000001", "平安银行", pd.Timestamp("2025-01-03"), 100, 95, 1.8, future)

        self.assertEqual("fixed_stop", trade.exit_reason)
        self.assertAlmostEqual(-0.10, trade.realized_return)
        self.assertAlmostEqual(90.0, trade.exit_price)

    def test_trailing_drawdown_can_exit_above_fixed_stop(self):
        future = pd.DataFrame(
            [
                weekly_bar("2025-01-10", 100, 130, 108, 126, 1000),
                weekly_bar("2025-01-17", 126, 128, 103, 106, 1000),
            ]
        )

        trade = simulate_trade("000001", "平安银行", pd.Timestamp("2025-01-03"), 100, 95, 1.8, future)

        self.assertEqual("trailing_drawdown", trade.exit_reason)
        self.assertAlmostEqual(0.04, trade.realized_return)
        self.assertAlmostEqual(104.0, trade.exit_price)

    def test_time_exit_reports_best_high_return_for_full_hold(self):
        future = pd.DataFrame(
            [
                weekly_bar("2025-01-10", 100, 106, 98, 103, 1000),
                weekly_bar("2025-01-17", 103, 111, 101, 108, 1000),
                weekly_bar("2025-01-24", 108, 115, 106, 110, 1000),
                weekly_bar("2025-01-31", 110, 114, 107, 109, 1000),
                weekly_bar("2025-02-07", 109, 112, 105, 107, 1000),
            ]
        )

        trade = simulate_trade("000001", "平安银行", pd.Timestamp("2025-01-03"), 100, 95, 1.8, future)

        self.assertEqual("time_exit", trade.exit_reason)
        self.assertAlmostEqual(0.07, trade.realized_return)
        self.assertAlmostEqual(0.15, trade.best_high_return)

    def test_normalize_tushare_daily_orders_ascending_and_uses_adjusted_close(self):
        raw = pd.DataFrame(
            [
                {"trade_date": "20250110", "open": 11, "high": 12, "low": 10, "close": 11.5, "vol": 200},
                {"trade_date": "20250103", "open": 10, "high": 11, "low": 9, "close": 10.5, "vol": 100},
            ]
        )

        normalized = normalize_tushare_daily(raw)

        self.assertEqual([pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-10")], normalized["date"].tolist())
        self.assertEqual([100.0, 200.0], normalized["volume"].tolist())

    def test_normalize_daily_columns_accepts_tencent_amount_as_volume_proxy(self):
        raw = pd.DataFrame(
            [{"date": "2025-01-03", "open": 10, "high": 11, "low": 9, "close": 10.5, "amount": 1234}]
        )

        normalized = normalize_daily_columns(raw)

        self.assertEqual(1234.0, normalized.at[0, "volume"])

    def test_market_code_for_tencent_provider(self):
        self.assertEqual("sh600000", market_code("600000"))
        self.assertEqual("sz000001", market_code("000001"))

    def test_normalize_tx_weekly_records(self):
        records = [
            ["2025-01-03", "10", "10.5", "11", "9", "1234.5"],
            ["2025-01-10", "10.5", "11", "12", "10", "2345.6"],
        ]

        normalized = normalize_tx_weekly_records(records)

        self.assertEqual(["date", "open", "close", "high", "low", "volume"], normalized.columns.tolist())
        self.assertEqual(pd.Timestamp("2025-01-03"), normalized.at[0, "date"])
        self.assertEqual(1234.5, normalized.at[0, "volume"])

    def test_backtest_stock_can_use_daily_bars_without_weekly_aggregation(self):
        rows = []
        start = pd.Timestamp("2025-01-01")
        for i in range(40):
            close = 10 + i * 0.1
            rows.append(weekly_bar(start + pd.Timedelta(days=i), close - 0.1, close + 0.2, close - 0.3, close, 1000))

        box_top = max(row["high"] for row in rows[-40:])
        rows.append(weekly_bar(start + pd.Timedelta(days=40), box_top + 0.1, box_top + 0.5, box_top, box_top + 0.3, 1800))
        for i in range(10):
            rows.append(weekly_bar(start + pd.Timedelta(days=41 + i), box_top + 0.3, box_top + 0.8, box_top + 0.1, box_top + 0.4, 1000))

        trades = backtest_stock(
            "000001",
            "平安银行",
            pd.DataFrame(rows),
            signal_start=start,
            box_weeks=40,
            hold_weeks=10,
            bar_frequency="daily",
        )

        self.assertEqual(1, len(trades))
        self.assertEqual(start + pd.Timedelta(days=50), trades[0].exit_date)


if __name__ == "__main__":
    unittest.main()
