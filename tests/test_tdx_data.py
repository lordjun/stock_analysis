import struct
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tdx_data import aggregate_tdx_weekly_bars, market_prefix, read_tdx_day_file, tdx_day_path


def record(date, open_, high, low, close, amount=1000000.0, volume=10000):
    return struct.pack(
        "<IIIIIfII",
        int(date),
        round(open_ * 100),
        round(high * 100),
        round(low * 100),
        round(close * 100),
        float(amount),
        int(volume),
        0,
    )


class TdxDataTests(unittest.TestCase):
    def test_market_prefix_supports_sh_sz_bj(self):
        self.assertEqual("sh", market_prefix("600000"))
        self.assertEqual("sz", market_prefix("000001"))
        self.assertEqual("bj", market_prefix("920000"))

    def test_tdx_day_path_uses_market_folder(self):
        root = Path("C:/new_tdx/vipdoc")

        self.assertEqual(root / "bj" / "lday" / "bj920000.day", tdx_day_path(root, "920000"))

    def test_read_tdx_day_file_parses_records_and_filters_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sz000001.day"
            path.write_bytes(
                record(20260102, 10.0, 11.0, 9.5, 10.5)
                + record(20260105, 10.5, 12.0, 10.2, 11.5)
            )

            bars = read_tdx_day_file(path, "20260105", "20260105")

            self.assertEqual(1, len(bars))
            self.assertEqual(pd.Timestamp("2026-01-05"), bars.at[0, "date"])
            self.assertAlmostEqual(10.5, bars.at[0, "open"])
            self.assertAlmostEqual(12.0, bars.at[0, "high"])
            self.assertAlmostEqual(10.2, bars.at[0, "low"])
            self.assertAlmostEqual(11.5, bars.at[0, "close"])

    def test_aggregate_tdx_weekly_bars(self):
        daily = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-01"), "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 100},
                {"date": pd.Timestamp("2026-01-02"), "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 200},
                {"date": pd.Timestamp("2026-01-05"), "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 300},
            ]
        )

        weekly = aggregate_tdx_weekly_bars(daily)

        self.assertEqual(2, len(weekly))
        self.assertEqual(pd.Timestamp("2026-01-02"), weekly.at[0, "date"])
        self.assertAlmostEqual(10.0, weekly.at[0, "open"])
        self.assertAlmostEqual(12.0, weekly.at[0, "high"])
        self.assertAlmostEqual(9.5, weekly.at[0, "low"])
        self.assertAlmostEqual(11.5, weekly.at[0, "close"])
        self.assertAlmostEqual(300.0, weekly.at[0, "volume"])


if __name__ == "__main__":
    unittest.main()
