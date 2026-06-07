from __future__ import annotations

import unittest

import pandas as pd

from scan_stage_high_breakout_retest import (
    build_constituent_return_concept_strength,
    build_long_term_concept_strength,
    extract_constituent_codes,
    parse_theme_keywords,
    select_long_term_concept_boards,
)


class ScanStageHighBreakoutRetestTests(unittest.TestCase):
    def test_extract_constituent_codes_handles_varied_column_names(self) -> None:
        df = pd.DataFrame(
            [
                {"证券代码": "300750", "证券简称": "宁德时代", "涨跌幅": 1.2},
                {"code": "sh688256", "name": "寒武纪", "change": 2.3},
            ]
        )

        self.assertEqual([("300750", "宁德时代"), ("688256", "寒武纪")], extract_constituent_codes(df))

    def test_build_long_term_concept_strength_scores_persistent_leaders(self) -> None:
        histories = {
            "AI": self._history(10.0, 26.0, 1.0),
            "短炒": self._history(10.0, 12.0, 3.0),
            "弱势": self._history(10.0, 8.0, 1.0),
        }

        strength = build_long_term_concept_strength(histories)

        self.assertEqual("AI", strength.iloc[0]["board_name"])
        self.assertGreater(strength.iloc[0]["ret250"], strength.iloc[1]["ret250"])
        self.assertGreater(strength.iloc[0]["trend_score"], 0)

    def test_select_long_term_concept_boards_filters_by_minimum_strength(self) -> None:
        histories = {
            "AI": self._history(10.0, 26.0, 1.0),
            "弱势": self._history(10.0, 8.0, 1.0),
        }

        selected = select_long_term_concept_boards(histories, top_n=5, min_ret120=0.20, min_ret250=0.50)

        self.assertEqual(["AI"], [item.name for item in selected])

    def test_build_constituent_return_concept_strength_scores_strong_member_returns(self) -> None:
        board_constituents = {
            "AI": pd.DataFrame([{"code": "000001"}, {"code": "000002"}, {"code": "000003"}]),
            "弱势": pd.DataFrame([{"code": "000004"}, {"code": "000005"}, {"code": "000006"}]),
        }
        stock_returns = {
            "000001": {"ret120": 0.8, "ret250": 1.2},
            "000002": {"ret120": 0.6, "ret250": 1.0},
            "000003": {"ret120": 0.4, "ret250": 0.8},
            "000004": {"ret120": 0.1, "ret250": 0.2},
            "000005": {"ret120": -0.1, "ret250": 0.1},
            "000006": {"ret120": 0.0, "ret250": -0.2},
        }

        strength = build_constituent_return_concept_strength(board_constituents, stock_returns)

        self.assertEqual("AI", strength.iloc[0]["board_name"])
        self.assertGreater(strength.iloc[0]["breadth120"], strength.iloc[1]["breadth120"])

    def test_parse_theme_keywords_accepts_cn_and_en_separators(self) -> None:
        self.assertEqual(["AI", "算力", "机器人"], parse_theme_keywords("AI，算力;机器人"))

    def _history(self, start: float, end: float, late_boost: float) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=260, freq="D")
        closes = pd.Series([start + (end - start) * index / 259 for index in range(260)], dtype=float)
        closes.iloc[-20:] = closes.iloc[-20:] + late_boost
        return pd.DataFrame({"date": dates, "close": closes})


if __name__ == "__main__":
    unittest.main()
