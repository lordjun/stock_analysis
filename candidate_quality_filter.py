from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import time

import pandas as pd
import requests


@dataclass(frozen=True)
class QualityThresholds:
    min_quality_score: float = 60.0
    min_avg_turnover_20d: float = 0.8
    max_debt_to_assets: float = 75.0
    max_pe_ttm: float = 120.0
    max_pb: float = 15.0
    min_revenue_yoy: float = -15.0
    min_profit_yoy: float = -20.0
    severe_outflow_20d: float = -50000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich K-line candidates with fundamental and fund-flow quality filters.")
    parser.add_argument("--input", required=True, help="CSV with at least code and signal_date columns.")
    parser.add_argument("--output-dir", default="reports/candidate_quality_filter")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--financial-start-date", default="20230101")
    parser.add_argument("--flow-lookback-days", type=int, default=45)
    parser.add_argument("--token-env", default="TUSHARE_TOKEN")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser


def main() -> None:
    paths = run(build_parser().parse_args())
    for path in paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    candidates = pd.read_csv(args.input, dtype={"code": str})
    if candidates.empty:
        raise RuntimeError(f"input is empty: {args.input}")
    if "code" not in candidates.columns:
        raise RuntimeError("input must contain a code column")

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"{args.token_env} is not set")

    import tushare as ts

    pro = ts.pro_api(token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for _, candidate in candidates.iterrows():
        code = str(candidate["code"]).zfill(6)
        signal_date = str(candidate.get("signal_date", args.end_date)).replace("-", "")
        try:
            features = collect_quality_features(
                pro=pro,
                code=code,
                signal_date=signal_date,
                end_date=args.end_date,
                financial_start_date=args.financial_start_date,
                flow_lookback_days=args.flow_lookback_days,
            )
            evaluation = evaluate_quality(features)
            output = candidate.to_dict()
            output.update(features)
            output.update(evaluation)
            rows.append(output)
        except Exception as exc:
            errors.append({"code": code, "reason": str(exc)[:300]})
        time.sleep(max(0.0, float(args.sleep_seconds)))

    enriched = pd.DataFrame(rows)
    if not enriched.empty:
        enriched = enriched.sort_values(["quality_pass", "quality_score", "code"], ascending=[False, False, True])

    enriched_path = output_dir / "quality_enriched_candidates.csv"
    passed_path = output_dir / "quality_passed_candidates.csv"
    errors_path = output_dir / "quality_errors.csv"
    summary_path = output_dir / "summary.md"

    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    passed = enriched[enriched["quality_pass"] == True].copy() if "quality_pass" in enriched else pd.DataFrame()
    passed.to_csv(passed_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(build_summary(enriched, errors), encoding="utf-8")
    return [enriched_path, passed_path, errors_path, summary_path]


def collect_quality_features(
    pro: object,
    code: str,
    signal_date: str,
    end_date: str,
    financial_start_date: str,
    flow_lookback_days: int,
) -> dict[str, object]:
    ts_code = to_ts_code(code)
    flow_start = (pd.Timestamp(signal_date) - pd.Timedelta(days=flow_lookback_days)).strftime("%Y%m%d")
    data_errors: list[str] = []

    fina = safe_frame(
        lambda: pro.fina_indicator(ts_code=ts_code, start_date=financial_start_date, end_date=end_date),
        "fina_indicator",
        data_errors,
    )
    income = safe_frame(
        lambda: pro.income(ts_code=ts_code, start_date=financial_start_date, end_date=end_date),
        "income",
        data_errors,
    )
    daily_basic = safe_frame(
        lambda: pro.daily_basic(ts_code=ts_code, start_date=flow_start, end_date=end_date),
        "daily_basic",
        data_errors,
    )
    moneyflow = safe_frame(
        lambda: pro.moneyflow(ts_code=ts_code, start_date=flow_start, end_date=end_date),
        "moneyflow",
        data_errors,
    )
    moneyflow_source = "tushare" if not moneyflow.empty else ""
    if moneyflow.empty:
        moneyflow = safe_frame(
            lambda: fetch_eastmoney_fund_flow(code, flow_lookback_days),
            "eastmoney_fund_flow",
            data_errors,
        )
        if not moneyflow.empty:
            moneyflow_source = "eastmoney_fund_flow"
    holder_number = safe_frame(
        lambda: pro.stk_holdernumber(ts_code=ts_code, start_date=financial_start_date, end_date=end_date),
        "stk_holdernumber",
        data_errors,
    )
    financial = financial_features(fina, income)
    financial_source = "tushare" if has_any_value(financial) else ""
    if not has_any_value(financial):
        financial = safe_feature_call(
            lambda: fetch_public_financial_features(code),
            "akshare_financial",
            data_errors,
        )
        if has_any_value(financial):
            financial_source = "akshare_financial"

    valuation = valuation_liquidity_features(daily_basic)
    valuation_source = "tushare" if has_any_value(valuation) else ""
    if not has_any_value(valuation):
        valuation = safe_feature_call(
            lambda: fetch_public_valuation_liquidity_features(code, flow_start, end_date, data_errors),
            "akshare_valuation_liquidity",
            data_errors,
        )
        if has_any_value(valuation):
            valuation_source = "akshare_valuation_liquidity"

    return {
        "ts_code": ts_code,
        "quality_data_errors": ";".join(data_errors),
        "financial_source": financial_source,
        "valuation_source": valuation_source,
        "moneyflow_source": moneyflow_source,
        **financial,
        **valuation,
        **moneyflow_features(moneyflow),
        **holder_features(holder_number),
    }


def evaluate_quality(features: dict[str, object], thresholds: QualityThresholds | None = None) -> dict[str, object]:
    t = thresholds or QualityThresholds()
    hard_reasons: list[str] = []
    positive_reasons: list[str] = []
    score = 0.0

    revenue_yoy = to_float(features.get("revenue_yoy_latest"))
    profit_yoy = to_float(features.get("profit_yoy_latest"))
    deducted_profit = to_float(features.get("deducted_netprofit_latest"))
    roe = to_float(features.get("roe_latest"))
    debt = to_float(features.get("debt_to_assets_latest"))
    pe_ttm = to_float(features.get("pe_ttm_latest"))
    pb = to_float(features.get("pb_latest"))
    avg_turnover = to_float(features.get("avg_turnover_20d"))
    net_5d = to_float(features.get("moneyflow_net_5d"))
    net_20d = to_float(features.get("moneyflow_net_20d"))
    has_financial_data = any(value is not None for value in [revenue_yoy, profit_yoy, deducted_profit, roe, debt])
    has_market_data = any(value is not None for value in [pe_ttm, pb, avg_turnover])
    has_flow_data = any(value is not None for value in [net_5d, net_20d])

    if not (has_financial_data or has_market_data or has_flow_data):
        hard_reasons.append("quality_data_missing")

    if revenue_yoy is not None and revenue_yoy >= 0:
        score += 15
        positive_reasons.append("revenue_non_negative")
    elif revenue_yoy is not None and revenue_yoy < t.min_revenue_yoy:
        hard_reasons.append("revenue_yoy_too_weak")

    if profit_yoy is not None and profit_yoy >= 0:
        score += 15
        positive_reasons.append("profit_non_negative")
    elif profit_yoy is not None and profit_yoy < t.min_profit_yoy:
        hard_reasons.append("profit_yoy_too_weak")

    if deducted_profit is not None and deducted_profit > 0:
        score += 10
        positive_reasons.append("deducted_profit_positive")
    elif deducted_profit is not None and deducted_profit <= 0:
        hard_reasons.append("deducted_profit_not_positive")

    if roe is not None and roe >= 5:
        score += 5
        positive_reasons.append("roe_acceptable")

    if debt is not None and debt > t.max_debt_to_assets:
        hard_reasons.append("debt_too_high")
    elif debt is not None:
        score += 10

    if pe_ttm is not None and 0 < pe_ttm <= t.max_pe_ttm:
        score += 8

    if pb is not None and 0 < pb <= t.max_pb:
        score += 7

    if avg_turnover is not None and avg_turnover >= t.min_avg_turnover_20d:
        score += 5
    elif avg_turnover is not None:
        hard_reasons.append("liquidity_too_low")

    if net_20d is not None and net_20d >= 0:
        score += 15
        positive_reasons.append("moneyflow_20d_positive")
    elif net_5d is not None and net_5d >= 0:
        score += 10
        positive_reasons.append("moneyflow_5d_positive")
    if net_20d is not None and net_20d < t.severe_outflow_20d and (net_5d is None or net_5d < 0):
        hard_reasons.append("persistent_main_outflow")

    score = round(min(100.0, score), 2)
    passed = score >= t.min_quality_score and not hard_reasons
    return {
        "quality_score": score,
        "quality_pass": bool(passed),
        "quality_positive_reasons": ";".join(positive_reasons),
        "quality_block_reasons": ";".join(hard_reasons),
    }


def financial_features(fina: pd.DataFrame, income: pd.DataFrame) -> dict[str, object]:
    latest_fina = latest_report_row(fina)
    latest_income = latest_report_row(income)
    return {
        "latest_report_date": latest_value(latest_fina, ["end_date"]),
        "revenue_yoy_latest": first_numeric(latest_fina, ["or_yoy", "q_gr_yoy", "tr_yoy", "q_sales_yoy"]),
        "profit_yoy_latest": first_numeric(latest_fina, ["q_profit_yoy", "netprofit_yoy", "dt_netprofit_yoy"]),
        "roe_latest": first_numeric(latest_fina, ["roe", "roe_dt", "roe_waa", "roe_yearly"]),
        "gross_margin_latest": first_numeric(latest_fina, ["grossprofit_margin"]),
        "debt_to_assets_latest": first_numeric(latest_fina, ["debt_to_assets", "asset_liab_ratio"]),
        "ocf_per_share_latest": first_numeric(latest_fina, ["ocfps", "cfps"]),
        "deducted_netprofit_latest": first_numeric(latest_fina, ["dt_netprofit", "deducted_profit"])
        or first_numeric(latest_income, ["n_income_attr_p"]),
    }


def akshare_financial_features(indicator: pd.DataFrame) -> dict[str, object]:
    latest = latest_report_row(indicator)
    return {
        "latest_report_date": latest_value(latest, ["REPORT_DATE_NAME", "REPORT_DATE", "end_date"]),
        "revenue_yoy_latest": first_numeric(latest, ["TOTALOPERATEREVETZ", "TOTALOPERATEREVEYOY", "营业总收入同比增长"]),
        "profit_yoy_latest": first_numeric(latest, ["PARENTNETPROFITTZ", "NETPROFITRPHBZCYOY", "归母净利润同比增长"]),
        "roe_latest": first_numeric(latest, ["ROEJQ", "ROE", "净资产收益率"]),
        "gross_margin_latest": first_numeric(latest, ["XSMLL", "GROSSPROFITMARGIN", "销售毛利率"]),
        "debt_to_assets_latest": first_numeric(latest, ["ZCFZL", "资产负债率"]),
        "ocf_per_share_latest": first_numeric(latest, ["MGJYXJJE", "每股经营现金流"]),
        "deducted_netprofit_latest": first_numeric(latest, ["KCFJCXSYJLR", "扣非净利润"]),
    }


def valuation_liquidity_features(daily_basic: pd.DataFrame) -> dict[str, object]:
    if daily_basic.empty:
        return {"pe_ttm_latest": None, "pb_latest": None, "avg_turnover_20d": None, "latest_total_mv": None}
    frame = daily_basic.copy()
    if "trade_date" in frame:
        frame = frame.sort_values("trade_date")
    latest = frame.iloc[-1].to_dict()
    turnover = pd.to_numeric(frame.get("turnover_rate"), errors="coerce") if "turnover_rate" in frame else pd.Series(dtype=float)
    return {
        "pe_ttm_latest": first_numeric(latest, ["pe_ttm", "pe"]),
        "pb_latest": first_numeric(latest, ["pb"]),
        "avg_turnover_20d": round(float(turnover.tail(20).mean()), 4) if turnover.notna().any() else None,
        "latest_total_mv": first_numeric(latest, ["total_mv", "circ_mv"]),
    }


def public_valuation_liquidity_features(snapshot: pd.DataFrame, history: pd.DataFrame) -> dict[str, object]:
    snapshot_values = normalize_snapshot_items(snapshot)
    turnover = pd.Series(dtype=float)
    if not history.empty:
        turnover_column = first_existing_column(history, ["换手率", "turnover_rate", "turnover"])
        if turnover_column:
            turnover = pd.to_numeric(history[turnover_column], errors="coerce")
    return {
        "pe_ttm_latest": first_numeric(snapshot_values, ["市盈率-动态", "市盈率TTM", "市盈率-滚动", "PE(TTM)", "pe_ttm", "pe"]),
        "pb_latest": first_numeric(snapshot_values, ["市净率", "PB", "pb"]),
        "avg_turnover_20d": round(float(turnover.tail(20).mean()), 4) if turnover.notna().any() else None,
        "latest_total_mv": first_numeric(snapshot_values, ["总市值", "流通市值", "total_mv", "circ_mv"]),
    }


def moneyflow_features(moneyflow: pd.DataFrame) -> dict[str, object]:
    if moneyflow.empty:
        return {"moneyflow_net_5d": None, "moneyflow_net_20d": None, "moneyflow_elg_lg_net_20d": None}
    frame = moneyflow.copy()
    if "trade_date" in frame:
        frame = frame.sort_values("trade_date")
    net_series = numeric_series(frame, "net_mf_amount")
    if net_series.empty:
        buy = numeric_series(frame, "buy_elg_amount") + numeric_series(frame, "buy_lg_amount")
        sell = numeric_series(frame, "sell_elg_amount") + numeric_series(frame, "sell_lg_amount")
        net_series = buy - sell
    elg_lg = numeric_series(frame, "buy_elg_amount") + numeric_series(frame, "buy_lg_amount")
    elg_lg = elg_lg - numeric_series(frame, "sell_elg_amount") - numeric_series(frame, "sell_lg_amount")
    return {
        "moneyflow_net_5d": round(float(net_series.tail(5).sum()), 2) if not net_series.empty else None,
        "moneyflow_net_20d": round(float(net_series.tail(20).sum()), 2) if not net_series.empty else None,
        "moneyflow_elg_lg_net_20d": round(float(elg_lg.tail(20).sum()), 2) if not elg_lg.empty else None,
    }


def fetch_eastmoney_fund_flow(code: str, limit: int = 120) -> pd.DataFrame:
    market_prefix = 1 if str(code).startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": str(max(20, int(limit))),
        "klt": "101",
        "secid": f"{market_prefix}.{str(code).zfill(6)}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "_": int(time.time() * 1000),
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
    data = request_json_with_retries(url, params=params, headers=headers).get("data") or {}
    klines = data.get("klines") or []
    return parse_eastmoney_fund_flow_klines(klines)


def parse_eastmoney_fund_flow_klines(klines: list[str]) -> pd.DataFrame:
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "main_net_amount": to_float(parts[1]),
                "super_net_amount": to_float(parts[5]),
                "large_net_amount": to_float(parts[4]),
            }
        )
    return normalize_eastmoney_fund_flow(pd.DataFrame(rows))


def normalize_eastmoney_fund_flow(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["trade_date", "net_mf_amount", "buy_elg_amount", "buy_lg_amount"])
    frame = raw.copy()
    date = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y%m%d")
    output = pd.DataFrame(
        {
            "trade_date": date,
            "net_mf_amount": pd.to_numeric(frame.get("main_net_amount"), errors="coerce"),
            "buy_elg_amount": pd.to_numeric(frame.get("super_net_amount"), errors="coerce"),
            "buy_lg_amount": pd.to_numeric(frame.get("large_net_amount"), errors="coerce"),
        }
    )
    return output.dropna(subset=["trade_date"]).reset_index(drop=True)


def holder_features(holder_number: pd.DataFrame) -> dict[str, object]:
    if holder_number.empty or "holder_num" not in holder_number:
        return {"holder_num_latest": None, "holder_num_change_pct": None}
    frame = holder_number.copy()
    if "end_date" in frame:
        frame = frame.sort_values("end_date")
    nums = pd.to_numeric(frame["holder_num"], errors="coerce").dropna()
    if nums.empty:
        return {"holder_num_latest": None, "holder_num_change_pct": None}
    change = None
    if len(nums) >= 2 and float(nums.iloc[-2]) > 0:
        change = round((float(nums.iloc[-1]) / float(nums.iloc[-2]) - 1) * 100, 2)
    return {"holder_num_latest": int(nums.iloc[-1]), "holder_num_change_pct": change}


def latest_report_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    work = frame.copy()
    sort_columns = [column for column in ["end_date", "ann_date", "f_ann_date"] if column in work]
    if not sort_columns and "REPORT_DATE" in work:
        parsed = pd.to_datetime(work["REPORT_DATE"], errors="coerce")
        if parsed.notna().any():
            work = work.assign(_report_date_sort=parsed)
            sort_columns = ["_report_date_sort"]
    if sort_columns:
        work = work.sort_values(sort_columns)
    row = work.iloc[-1].to_dict()
    row.pop("_report_date_sort", None)
    return row


def latest_value(row: dict[str, object], names: list[str]) -> object | None:
    for name in names:
        value = row.get(name)
        if pd.notna(value):
            return value
    return None


def first_numeric(row: dict[str, object], names: list[str]) -> float | None:
    for name in names:
        value = to_float(row.get(name))
        if value is not None:
            return value
    return None


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def safe_frame(callable_: object, source: str, errors: list[str]) -> pd.DataFrame:
    try:
        result = callable_()
        if result is None:
            return pd.DataFrame()
        return result.copy()
    except Exception as exc:
        errors.append(f"{source}:{str(exc)[:120]}")
        return pd.DataFrame()


def safe_feature_call(callable_: object, source: str, errors: list[str]) -> dict[str, object]:
    try:
        result = callable_()
        return result.copy() if result else {}
    except Exception as exc:
        errors.append(f"{source}:{str(exc)[:120]}")
        return {}


def fetch_public_financial_features(code: str) -> dict[str, object]:
    import akshare as ak

    indicator = ak.stock_financial_analysis_indicator_em(
        symbol=to_ts_code(code),
        indicator="\u6309\u62a5\u544a\u671f",
    )
    return akshare_financial_features(indicator)


def fetch_public_valuation_liquidity_features(
    code: str,
    start_date: str,
    end_date: str,
    errors: list[str] | None = None,
) -> dict[str, object]:
    import akshare as ak

    symbol = str(code).zfill(6)
    try:
        snapshot_features = fetch_tencent_quote_snapshot(symbol)
    except Exception as exc:
        if errors is not None:
            errors.append(f"tencent_quote:{str(exc)[:120]}")
        snapshot_features = fetch_eastmoney_quote_snapshot(symbol)
    try:
        history = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
    except Exception as exc:
        if errors is not None:
            errors.append(f"akshare_turnover_history:{str(exc)[:120]}")
        history = pd.DataFrame()
    history_features = public_valuation_liquidity_features(pd.DataFrame(), history)
    return {
        **snapshot_features,
        "avg_turnover_20d": history_features.get("avg_turnover_20d") or snapshot_features.get("avg_turnover_20d"),
    }


def has_any_value(features: dict[str, object]) -> bool:
    return any(value is not None and not (isinstance(value, str) and value == "") for value in features.values())


def normalize_snapshot_items(snapshot: pd.DataFrame) -> dict[str, object]:
    if snapshot.empty:
        return {}
    columns = list(snapshot.columns)
    if "item" in snapshot and "value" in snapshot:
        return {str(row["item"]): row["value"] for _, row in snapshot.iterrows()}
    if "\u9879\u76ee" in snapshot and "\u503c" in snapshot:
        return {str(row["\u9879\u76ee"]): row["\u503c"] for _, row in snapshot.iterrows()}
    if len(columns) >= 2:
        return {str(row[columns[0]]): row[columns[1]] for _, row in snapshot.iterrows()}
    return {}


def first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def fetch_eastmoney_quote_snapshot(code: str) -> dict[str, object]:
    market_prefix = 1 if str(code).startswith("6") else 0
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{market_prefix}.{str(code).zfill(6)}",
        "fields": "f57,f58,f162,f167,f116,f117",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    return parse_eastmoney_quote_snapshot(request_json_with_retries(url, params=params, headers=headers))


def fetch_tencent_quote_snapshot(code: str) -> dict[str, object]:
    prefix = "sh" if str(code).startswith("6") else "sz"
    url = f"https://qt.gtimg.cn/q={prefix}{str(code).zfill(6)}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    return parse_tencent_quote(response.text)


def parse_tencent_quote(text: str) -> dict[str, object]:
    body = str(text)
    if '="' in body:
        body = body.split('="', 1)[1]
    body = body.strip().rstrip('";')
    parts = body.split("~")
    return {
        "pe_ttm_latest": value_at(parts, 52),
        "pb_latest": value_at(parts, 46),
        "avg_turnover_20d": value_at(parts, 38),
        "latest_total_mv": multiply(value_at(parts, 45), 100000000.0),
    }


def request_json_with_retries(
    url: str,
    params: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 3,
    delay_seconds: float = 0.6,
    timeout: int = 20,
    requester: object | None = None,
) -> dict[str, object]:
    request = requester or requests.get
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = request(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    raise RuntimeError(str(last_error) if last_error else "request failed")


def parse_eastmoney_quote_snapshot(payload: dict[str, object]) -> dict[str, object]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    data = data if isinstance(data, dict) else {}
    return {
        "pe_ttm_latest": scale_quote_ratio(data.get("f162")),
        "pb_latest": scale_quote_ratio(data.get("f167")),
        "latest_total_mv": to_float(data.get("f116")),
    }


def scale_quote_ratio(value: object) -> float | None:
    number = to_float(value)
    if number is None or number <= 0:
        return None
    return round(number / 100, 4)


def value_at(parts: list[str], index: int) -> float | None:
    if index >= len(parts):
        return None
    return to_float(parts[index])


def multiply(value: float | None, factor: float) -> float | None:
    return round(value * factor, 4) if value is not None else None


def to_ts_code(code: str) -> str:
    normalized = str(code).strip().upper()
    if "." in normalized:
        return normalized
    symbol = normalized.zfill(6)
    suffix = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
    return f"{symbol}.{suffix}"


def to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_summary(enriched: pd.DataFrame, errors: list[dict[str, str]]) -> str:
    total = len(enriched)
    passed = int(enriched["quality_pass"].sum()) if "quality_pass" in enriched else 0
    lines = [
        "# Candidate Quality Filter",
        "",
        f"- Candidates enriched: {total}",
        f"- Quality passed: {passed}",
        f"- Data errors: {len(errors)}",
        "",
        "## Filter Logic",
        "- Hard blocks: severe revenue/profit deterioration, non-positive deducted profit, high debt, low liquidity, or persistent main-fund outflow.",
        "- Score positives: revenue/profit stability, positive deducted profit, acceptable ROE/debt, reasonable valuation, sufficient turnover, and positive 5/20-day money flow.",
        "",
        "This is a research filter for candidate reduction, not an execution recommendation.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
