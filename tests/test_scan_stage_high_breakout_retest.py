from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scan_stage_high_breakout_retest import (
    build_constituent_return_concept_strength,
    build_long_term_concept_strength,
    extract_constituent_codes,
    load_or_build_structural_code_pool,
    parse_theme_keywords,
    select_long_term_concept_boards,
    should_prefilter_with_structural_pool,
    stock_matches_structural_theme,
    write_scan_outputs,
)


class ScanStageHighBreakoutRetestTests(unittest.TestCase):
    def test_dynamic_mainline_scan_does_not_prefilter_universe_before_kline_signal(self) -> None:
        args = SimpleNamespace(mainline_mode="dynamic", structural_code_pool_file=None)

        self.assertFalse(should_prefilter_with_structural_pool(args))

    def test_manual_structural_code_pool_scan_can_prefilter_universe(self) -> None:
        args = SimpleNamespace(mainline_mode="dynamic", structural_code_pool_file="pool.csv")

        self.assertTrue(should_prefilter_with_structural_pool(args))

    def test_scan_stock_matches_structural_theme_by_single_stock_concept_fallback(self) -> None:
        args = SimpleNamespace(
            structural_board_pool={"存储芯片"},
            structural_board_rank={"存储芯片": 5},
            stock_concepts_by_code={"688549": ["存储芯片"]},
        )

        matched, concepts, matched_boards = stock_matches_structural_theme(
            args,
            "688549",
            pd.Timestamp("2026-05-07"),
            structural_code_pool=set(),
        )

        self.assertTrue(matched)
        self.assertEqual(["存储芯片"], concepts)
        self.assertEqual(["存储芯片"], matched_boards)

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

    def test_load_or_build_structural_code_pool_uses_dynamic_provider_by_default(self) -> None:
        class FakeProvider:
            def __init__(self, config: object) -> None:
                self.config = config

            def codes_for_date(self, date: str) -> set[str]:
                return {"300843", "688549"}

            def boards_for_date(self, date: str) -> set[str]:
                return {"AI"}

            def board_ranks_for_date(self, date: str) -> dict[str, int]:
                return {"AI": 1}

        args = SimpleNamespace(
            structural_code_pool_file=None,
            mainline_mode="dynamic",
            dynamic_mainline_cache_dir="data/cache/test_mainline",
            tdx_vipdoc="C:/tdx",
            structural_concept_lookback_days=540,
            structural_concept_top_n=10,
            min_structural_concept_ret60=0.1,
            min_structural_concept_ret120=0.2,
            min_structural_concept_ret250=0.5,
            min_structural_concept_breadth120=0.55,
            structural_theme_keywords="AI,CPO",
            end_date="20260618",
            allow_structural_bull=True,
            structural_code_pool_cache=None,
            structural_pool_mode="long_term",
        )

        with patch("scan_stage_high_breakout_retest.DynamicMainlinePoolProvider", FakeProvider):
            self.assertEqual({"300843", "688549"}, load_or_build_structural_code_pool(args))

    def test_write_scan_outputs_skips_files_when_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "scan"

            paths = write_scan_outputs(
                candidates=pd.DataFrame(),
                failures=[],
                output_dir=output_dir,
                total_count=10,
            )

            self.assertEqual([], paths)
            self.assertFalse(output_dir.exists())

    def test_write_scan_outputs_writes_candidate_report_only_when_candidates_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "scan"
            candidates = pd.DataFrame(
                [
                    {
                        "code": "002484",
                        "name": "Jianghai",
                        "pattern_subtype": "steady_climb_retest",
                        "signal_date": "2026-04-29",
                        "breakout_date": "2026-04-21",
                        "latest_close": 38.18,
                        "structure_high": 34.89,
                        "signal_initial_stop_price": 33.84,
                        "breakout_volume_ratio": 1.112,
                        "pullback_volume_ratio": 1.078,
                        "similarity_score": 78.06,
                        "concept_boards": "PCB",
                        "market_regime": "structural_bull",
                    }
                ]
            )

            paths = write_scan_outputs(
                candidates=candidates,
                failures=[],
                output_dir=output_dir,
                total_count=10,
            )

            self.assertTrue(output_dir.exists())
            self.assertEqual([output_dir / "candidates.csv", output_dir / "report.md"], paths)
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("002484", report)
            self.assertIn("steady_climb_retest", report)
            self.assertIn("33.84", report)

    def _history(self, start: float, end: float, late_boost: float) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=260, freq="D")
        closes = pd.Series([start + (end - start) * index / 259 for index in range(260)], dtype=float)
        closes.iloc[-20:] = closes.iloc[-20:] + late_boost
        return pd.DataFrame({"date": dates, "close": closes})


if __name__ == "__main__":
    unittest.main()
