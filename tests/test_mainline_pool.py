from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mainline_pool import (
    DynamicMainlinePoolProvider,
    MainlinePoolConfig,
    business_day_snapshot_dates,
    concept_names_match_mainline,
    core_mainline_concept_matches,
    codes_asof_date,
    codes_from_pool_frame,
    mainline_cache_path,
    parse_theme_keywords,
    parse_ths_stock_concepts,
    parse_ths_max_page,
    select_dynamic_mainline_boards,
    weekly_snapshot_dates,
)


class MainlinePoolTests(unittest.TestCase):
    def test_codes_from_pool_frame_normalizes_codes(self) -> None:
        frame = pd.DataFrame({"code": ["1", "300843", None]})

        self.assertEqual({"000001", "300843"}, codes_from_pool_frame(frame))

    def test_weekly_snapshot_dates_includes_end_date(self) -> None:
        dates = weekly_snapshot_dates("20260401", "20260413")

        self.assertIn("20260403", dates)
        self.assertIn("20260410", dates)
        self.assertIn("20260413", dates)

    def test_business_day_snapshot_dates_uses_weekdays(self) -> None:
        dates = business_day_snapshot_dates("20260403", "20260406")

        self.assertEqual(["20260403", "20260406"], dates)

    def test_dynamic_provider_builds_and_reuses_cache(self) -> None:
        calls: list[str] = []

        def builder(end_date: str, config: MainlinePoolConfig) -> pd.DataFrame:
            calls.append(end_date)
            return pd.DataFrame(
                [
                    {"code": "300843", "name": "A", "board_name": "AI", "board_rank": 1, "pool_source": "test"},
                    {"code": "688549", "name": "B", "board_name": "AI", "board_rank": 1, "pool_source": "test"},
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            config = MainlinePoolConfig(cache_dir=Path(directory), tdx_vipdoc="C:/tdx")
            provider = DynamicMainlinePoolProvider(config, builder=builder)

            self.assertEqual({"300843", "688549"}, provider.codes_for_date("2026-04-13"))
            self.assertEqual({"300843", "688549"}, provider.codes_for_date("20260413"))
            self.assertEqual(["20260413"], calls)
            self.assertTrue(mainline_cache_path(Path(directory), "20260413", config).exists())

    def test_parse_theme_keywords_splits_common_delimiters(self) -> None:
        self.assertEqual(("AI", "CPO", "PCB"), parse_theme_keywords("AI,CPO PCB"))

    def test_parse_ths_max_page_reads_change_page_links(self) -> None:
        html = '<a class="cur" page="1"></a><a class="changePage" page="2"></a><a class="changePage" page="35"></a>'

        self.assertEqual(35, parse_ths_max_page(html))

    def test_parse_ths_stock_concepts_reads_gn_name_cells(self) -> None:
        html = '<td class="gnName" clid="886042"> 存储芯片 </td><td class="gnName">PCB概念</td>'

        self.assertEqual(["存储芯片", "PCB概念"], parse_ths_stock_concepts(html))

    def test_concept_names_match_mainline_with_suffix_and_parentheses(self) -> None:
        self.assertTrue(concept_names_match_mainline(["存储芯片"], {"存储芯片", "芯片概念"}))
        self.assertTrue(concept_names_match_mainline(["数据中心"], {"数据中心(AIDC)"}))
        self.assertFalse(concept_names_match_mainline(["医药商业"], {"存储芯片", "PCB概念"}))

    def test_core_mainline_concept_matches_prefers_top_rank_or_multiple_matches(self) -> None:
        rank_by_board = {"芯片概念": 2, "存储芯片": 5, "机器人概念": 7, "无人机": 8}

        self.assertEqual(["芯片概念"], core_mainline_concept_matches(["芯片概念"], rank_by_board))
        self.assertEqual([], core_mainline_concept_matches(["机器人概念"], rank_by_board))
        self.assertEqual(["机器人概念", "无人机"], core_mainline_concept_matches(["机器人概念", "无人机"], rank_by_board))
        self.assertEqual(["存储芯片", "芯片概念"], core_mainline_concept_matches(["存储芯片", "芯片概念"], rank_by_board))

    def test_codes_asof_date_uses_latest_prior_pool(self) -> None:
        pool_by_date = {
            "20260410": {"300843"},
            "20260417": {"688549"},
        }

        self.assertEqual({"300843"}, codes_asof_date("20260416", pool_by_date))
        self.assertEqual({"688549"}, codes_asof_date("20260417", pool_by_date))
        self.assertEqual(set(), codes_asof_date("20260401", pool_by_date))

    def test_select_dynamic_mainline_boards_falls_back_to_current_strength(self) -> None:
        strength = pd.DataFrame(
            [
                {
                    "board_name": "AI",
                    "ret60_median": 0.12,
                    "ret120_median": 0.18,
                    "ret250_median": 0.05,
                    "breadth120": 0.70,
                    "mainline_score": 0.30,
                },
                {
                    "board_name": "OldHot",
                    "ret60_median": 0.03,
                    "ret120_median": 0.12,
                    "ret250_median": 0.35,
                    "breadth120": 0.60,
                    "mainline_score": 0.20,
                },
            ]
        )

        selected = select_dynamic_mainline_boards(
            strength,
            MainlinePoolConfig(cache_dir=Path("cache"), tdx_vipdoc="C:/tdx", min_ret250=0.50),
        )

        self.assertEqual(["AI"], selected["board_name"].tolist())


if __name__ == "__main__":
    unittest.main()
