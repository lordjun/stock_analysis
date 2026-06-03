import unittest

import pandas as pd

from candidate_priority_ranker import (
    board_persistence_score,
    normalize_series,
    rank_candidates,
    theme_keyword_score,
)


class CandidatePriorityRankerTests(unittest.TestCase):
    def test_normalize_series_handles_equal_values(self):
        values = pd.Series([3.0, 3.0, 3.0])

        normalized = normalize_series(values)

        self.assertEqual([0.5, 0.5, 0.5], normalized.tolist())

    def test_theme_keyword_score_rewards_structural_hot_topics(self):
        self.assertGreater(theme_keyword_score("AI应用"), theme_keyword_score("水泥建材"))
        self.assertGreater(theme_keyword_score("CPO概念"), theme_keyword_score("银行"))

    def test_board_persistence_score_rewards_persistent_uptrend(self):
        strong = pd.DataFrame(
            {
                "close": [100 + i for i in range(80)],
                "volume": [1000 + i for i in range(80)],
            }
        )
        weak = pd.DataFrame(
            {
                "close": [180 - i for i in range(80)],
                "volume": [1000 for _ in range(80)],
            }
        )

        self.assertGreater(board_persistence_score(strong), board_persistence_score(weak))

    def test_rank_candidates_prefers_strong_hot_board_membership(self):
        candidates = pd.DataFrame(
            [
                {"code": "000001", "name": "Alpha", "latest_close": 10.0},
                {"code": "000002", "name": "Beta", "latest_close": 20.0},
            ]
        )
        stock_to_boards = {
            "000001": ["AI应用", "CPO概念"],
            "000002": ["冷门概念"],
        }
        board_scores = {
            "AI应用": 0.95,
            "CPO概念": 0.90,
            "冷门概念": 0.20,
        }
        board_reasons = {
            "AI应用": "10日涨幅靠前; 主力净流入靠前",
            "CPO概念": "一年趋势强",
            "冷门概念": "弱",
        }

        ranked = rank_candidates(candidates, stock_to_boards, board_scores, board_reasons)

        self.assertEqual("000001", ranked.iloc[0]["code"])
        self.assertGreater(ranked.iloc[0]["priority_score"], ranked.iloc[1]["priority_score"])
        self.assertIn("AI应用", ranked.iloc[0]["matched_hot_boards"])


if __name__ == "__main__":
    unittest.main()
