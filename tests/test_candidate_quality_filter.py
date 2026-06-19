from __future__ import annotations

import unittest

import pandas as pd

from candidate_quality_filter import (
    akshare_financial_features,
    evaluate_quality,
    financial_features,
    latest_report_row,
    moneyflow_features,
    normalize_eastmoney_fund_flow,
    parse_eastmoney_quote_snapshot,
    parse_eastmoney_fund_flow_klines,
    parse_tencent_quote,
    public_valuation_liquidity_features,
    request_json_with_retries,
    to_ts_code,
    valuation_liquidity_features,
)


class CandidateQualityFilterTests(unittest.TestCase):
    def test_to_ts_code_infers_exchange(self) -> None:
        self.assertEqual("600183.SH", to_ts_code("600183"))
        self.assertEqual("300843.SZ", to_ts_code("300843"))
        self.assertEqual("688097.SH", to_ts_code("688097"))

    def test_financial_features_prefers_latest_report(self) -> None:
        fina = pd.DataFrame(
            [
                {
                    "end_date": "20231231",
                    "or_yoy": -10,
                    "q_profit_yoy": -5,
                    "dt_netprofit": 100,
                    "roe": 4,
                    "debt_to_assets": 40,
                },
                {
                    "end_date": "20240331",
                    "or_yoy": 18,
                    "q_profit_yoy": 25,
                    "dt_netprofit": 250,
                    "roe": 8,
                    "debt_to_assets": 38,
                },
            ]
        )

        features = financial_features(fina, pd.DataFrame())

        self.assertEqual("20240331", features["latest_report_date"])
        self.assertEqual(18, features["revenue_yoy_latest"])
        self.assertEqual(25, features["profit_yoy_latest"])
        self.assertEqual(250, features["deducted_netprofit_latest"])

    def test_akshare_financial_features_maps_em_indicator_columns(self) -> None:
        indicator = pd.DataFrame(
            [
                {
                    "REPORT_DATE_NAME": "2023年度",
                    "TOTALOPERATEREVETZ": -8.0,
                    "PARENTNETPROFITTZ": -12.0,
                    "KCFJCXSYJLR": 90.0,
                    "ROEJQ": 3.0,
                    "XSMLL": 20.0,
                    "ZCFZL": 45.0,
                    "MGJYXJJE": 0.1,
                },
                {
                    "REPORT_DATE_NAME": "2024年度",
                    "TOTALOPERATEREVETZ": 18.0,
                    "PARENTNETPROFITTZ": 28.0,
                    "KCFJCXSYJLR": 260.0,
                    "ROEJQ": 8.5,
                    "XSMLL": 31.5,
                    "ZCFZL": 38.0,
                    "MGJYXJJE": 0.35,
                },
            ]
        )

        features = akshare_financial_features(indicator)

        self.assertEqual("2024年度", features["latest_report_date"])
        self.assertEqual(18.0, features["revenue_yoy_latest"])
        self.assertEqual(28.0, features["profit_yoy_latest"])
        self.assertEqual(260.0, features["deducted_netprofit_latest"])
        self.assertEqual(8.5, features["roe_latest"])
        self.assertEqual(38.0, features["debt_to_assets_latest"])

    def test_latest_report_row_prefers_max_report_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"REPORT_DATE": "2026-03-31", "REPORT_DATE_NAME": "2026一季报", "ROEJQ": 8.0},
                {"REPORT_DATE": "2014-12-31", "REPORT_DATE_NAME": "2014年报", "ROEJQ": 2.0},
            ]
        )

        latest = latest_report_row(frame)

        self.assertEqual("2026一季报", latest["REPORT_DATE_NAME"])

    def test_moneyflow_features_sums_recent_windows(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": [f"202601{day:02d}" for day in range(1, 22)],
                "net_mf_amount": list(range(1, 22)),
            }
        )

        features = moneyflow_features(frame)

        self.assertEqual(95.0, features["moneyflow_net_5d"])
        self.assertEqual(230.0, features["moneyflow_net_20d"])

    def test_normalize_eastmoney_fund_flow_maps_net_amount_fields(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "main_net_amount": [100.0, 200.0],
                "super_net_amount": [30.0, 40.0],
                "large_net_amount": [20.0, 30.0],
            }
        )

        normalized = normalize_eastmoney_fund_flow(raw)

        self.assertEqual(["trade_date", "net_mf_amount", "buy_elg_amount", "buy_lg_amount"], list(normalized.columns))
        self.assertEqual(["20260101", "20260102"], normalized["trade_date"].tolist())
        self.assertEqual([100.0, 200.0], normalized["net_mf_amount"].tolist())
        self.assertEqual([30.0, 40.0], normalized["buy_elg_amount"].tolist())
        self.assertEqual([20.0, 30.0], normalized["buy_lg_amount"].tolist())

    def test_parse_eastmoney_fund_flow_klines_uses_large_and_super_net_positions(self) -> None:
        normalized = parse_eastmoney_fund_flow_klines(
            [
                "2026-01-01,100,2,3,20,30,6,7,8,9,10,11,12,13,14",
                "2026-01-02,200,2,3,25,35,6,7,8,9,10,11,12,13,14",
            ]
        )

        self.assertEqual([100.0, 200.0], normalized["net_mf_amount"].tolist())
        self.assertEqual([20.0, 25.0], normalized["buy_lg_amount"].tolist())
        self.assertEqual([30.0, 35.0], normalized["buy_elg_amount"].tolist())

    def test_valuation_liquidity_features_uses_latest_and_average_turnover(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": ["20260101", "20260102"],
                "pe_ttm": [30, 35],
                "pb": [3, 4],
                "turnover_rate": [1.0, 2.0],
                "total_mv": [10000, 12000],
            }
        )

        features = valuation_liquidity_features(frame)

        self.assertEqual(35, features["pe_ttm_latest"])
        self.assertEqual(4, features["pb_latest"])
        self.assertEqual(1.5, features["avg_turnover_20d"])

    def test_public_valuation_liquidity_features_maps_snapshot_and_history(self) -> None:
        snapshot = pd.DataFrame(
            {
                "item": ["市盈率-动态", "市净率", "总市值"],
                "value": [42.0, 5.5, 1200000.0],
            }
        )
        history = pd.DataFrame({"日期": ["2026-01-01", "2026-01-02"], "换手率": [1.2, 2.2]})

        features = public_valuation_liquidity_features(snapshot, history)

        self.assertEqual(42.0, features["pe_ttm_latest"])
        self.assertEqual(5.5, features["pb_latest"])
        self.assertEqual(1.7, features["avg_turnover_20d"])
        self.assertEqual(1200000.0, features["latest_total_mv"])

    def test_parse_eastmoney_quote_snapshot_maps_scaled_valuation_fields(self) -> None:
        features = parse_eastmoney_quote_snapshot(
            {
                "data": {
                    "f162": 14391,
                    "f167": 1287,
                    "f116": 20758130400.4,
                }
            }
        )

        self.assertEqual(143.91, features["pe_ttm_latest"])
        self.assertEqual(12.87, features["pb_latest"])
        self.assertEqual(20758130400.4, features["latest_total_mv"])

    def test_parse_tencent_quote_maps_valuation_and_turnover(self) -> None:
        parts = [""] * 88
        parts[38] = "5.63"
        parts[45] = "207.58"
        parts[46] = "12.87"
        parts[52] = "143.91"

        features = parse_tencent_quote("v_sz300843=\"" + "~".join(parts) + "\";")

        self.assertEqual(143.91, features["pe_ttm_latest"])
        self.assertEqual(12.87, features["pb_latest"])
        self.assertEqual(5.63, features["avg_turnover_20d"])
        self.assertEqual(20758000000.0, features["latest_total_mv"])

    def test_request_json_with_retries_recovers_from_transient_failure(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"ok": True}

        def requester(*args: object, **kwargs: object) -> FakeResponse:
            calls["count"] += 1
            if calls["count"] == 1:
                raise ConnectionError("temporary")
            return FakeResponse()

        payload = request_json_with_retries("https://example.invalid", requester=requester, attempts=2, delay_seconds=0)

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, calls["count"])

    def test_evaluate_quality_passes_balanced_candidate(self) -> None:
        result = evaluate_quality(
            {
                "revenue_yoy_latest": 12,
                "profit_yoy_latest": 20,
                "deducted_netprofit_latest": 10000,
                "roe_latest": 9,
                "debt_to_assets_latest": 45,
                "pe_ttm_latest": 45,
                "pb_latest": 5,
                "avg_turnover_20d": 2.5,
                "moneyflow_net_5d": 5000,
                "moneyflow_net_20d": 15000,
            }
        )

        self.assertTrue(result["quality_pass"])
        self.assertEqual("", result["quality_block_reasons"])

    def test_evaluate_quality_blocks_profit_and_outflow_risk(self) -> None:
        result = evaluate_quality(
            {
                "revenue_yoy_latest": -30,
                "profit_yoy_latest": -50,
                "deducted_netprofit_latest": -100,
                "debt_to_assets_latest": 80,
                "pe_ttm_latest": 300,
                "pb_latest": 30,
                "avg_turnover_20d": 0.2,
                "moneyflow_net_5d": -1000,
                "moneyflow_net_20d": -80000,
            }
        )

        self.assertFalse(result["quality_pass"])
        self.assertIn("profit_yoy_too_weak", result["quality_block_reasons"])
        self.assertIn("persistent_main_outflow", result["quality_block_reasons"])

    def test_evaluate_quality_does_not_block_high_valuation(self) -> None:
        result = evaluate_quality(
            {
                "revenue_yoy_latest": 20,
                "profit_yoy_latest": 15,
                "deducted_netprofit_latest": 1000,
                "debt_to_assets_latest": 40,
                "pe_ttm_latest": 300,
                "pb_latest": 30,
                "avg_turnover_20d": 3,
            }
        )

        self.assertNotIn("pe_ttm_outlier", result["quality_block_reasons"])
        self.assertNotIn("pb_outlier", result["quality_block_reasons"])

    def test_evaluate_quality_distinguishes_missing_data_from_low_liquidity(self) -> None:
        result = evaluate_quality({})

        self.assertFalse(result["quality_pass"])
        self.assertIn("quality_data_missing", result["quality_block_reasons"])
        self.assertNotIn("liquidity_too_low", result["quality_block_reasons"])


if __name__ == "__main__":
    unittest.main()
