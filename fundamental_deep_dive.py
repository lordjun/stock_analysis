from __future__ import annotations

import argparse
from pathlib import Path
import time

import akshare as ak
import pandas as pd
import requests


STOCKS = {"301218": "华是科技", "600936": "北投科技"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a basic fundamental and fund-flow deep-dive report.")
    parser.add_argument("--output-dir", default="reports/fundamental_deep_dive_20260603")
    parser.add_argument("--start-year", default="2023")
    parser.add_argument("--stocks", nargs="*", help="Stock specs like 002518:科士达")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stocks = parse_stock_specs(args.stocks) if args.stocks else STOCKS
    for code, name in stocks.items():
        export_stock(code, name, output_dir, args.start_year)


def parse_stock_specs(specs: list[str]) -> dict[str, str]:
    stocks: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid stock spec: {spec}. Expected CODE:NAME")
        code, name = spec.split(":", 1)
        code = code.strip()
        name = name.strip()
        if not code or not name:
            raise ValueError(f"Invalid stock spec: {spec}. Expected CODE:NAME")
        stocks[code] = name
    return stocks


def export_stock(code: str, name: str, output_dir: Path, start_year: str) -> None:
    frames: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    frames["business_intro"] = fetch_frame(lambda: ak.stock_zyjs_ths(symbol=code), errors, "stock_zyjs_ths")
    frames["business_segments"] = latest_business_segments(code, errors)
    frames["financial_abstract"] = selected_financial_abstract(code, errors)
    frames["financial_indicators"] = selected_financial_indicators(code, errors)
    frames["top_holders"] = latest_top_holders(code, errors)
    frames["fund_holders"] = latest_fund_holders(code, errors)
    frames["fund_flow_summary"] = fund_flow_summary(code, errors)
    frames["fund_flow_recent"] = recent_fund_flow(code, errors)
    frames["northbound_summary"] = northbound_summary(code, errors)
    frames["northbound_recent"] = recent_northbound(code, errors)

    prefix = output_dir / f"{code}_{name}"
    for key, frame in frames.items():
        frame.to_csv(f"{prefix}_{key}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(f"{prefix}_errors.csv", index=False, encoding="utf-8-sig")


def fetch_frame(callable_, errors: list[dict[str, str]], source: str, attempts: int = 4) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = callable_()
            if result is None:
                return pd.DataFrame()
            return result.copy()
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    errors.append({"source": source, "reason": str(last_error)[:240] if last_error else "unknown"})
    return pd.DataFrame()


def exchange_code(code: str) -> str:
    return f"{code}.SZ" if code.startswith(("0", "3")) else f"{code}.SH"


def em_symbol(code: str) -> str:
    return f"SZ{code}" if code.startswith(("0", "3")) else f"SH{code}"


def market(code: str) -> str:
    return "sz" if code.startswith(("0", "3")) else "sh"


def latest_business_segments(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(lambda: ak.stock_zygc_em(symbol=em_symbol(code)), errors, "stock_zygc_em")
    if df.empty or "报告日期" not in df:
        return df
    latest = str(df["报告日期"].max())
    keep = df[(df["报告日期"].astype(str) == latest) & df["分类类型"].isin(["按产品分类", "按行业分类", "按地区分类"])].copy()
    columns = ["报告日期", "分类类型", "主营构成", "主营收入", "收入比例", "主营利润", "利润比例", "毛利率"]
    return keep[[column for column in columns if column in keep]].head(30)


def selected_financial_abstract(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(lambda: ak.stock_financial_abstract(symbol=code), errors, "stock_financial_abstract")
    if df.empty or "指标" not in df:
        return df
    metrics = [
        "营业总收入",
        "归母净利润",
        "扣非净利润",
        "经营现金流量净额",
        "股东权益合计(净资产)",
        "资产负债率",
        "毛利率",
        "销售净利率",
        "净资产收益率",
    ]
    date_columns = [column for column in df.columns if str(column).isdigit()]
    rows = []
    for metric in metrics:
        match = df[df["指标"].astype(str) == metric]
        if match.empty:
            continue
        row = {"指标": metric}
        for column in date_columns[:10]:
            row[column] = to_float(match.iloc[0][column])
        rows.append(row)
    return pd.DataFrame(rows)


def selected_financial_indicators(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(
        lambda: ak.stock_financial_analysis_indicator_em(symbol=exchange_code(code), indicator="按报告期"),
        errors,
        "stock_financial_analysis_indicator_em",
    )
    if df.empty:
        return df
    columns = [
        "REPORT_DATE_NAME",
        "TOTALOPERATEREVE",
        "TOTALOPERATEREVETZ",
        "PARENTNETPROFIT",
        "PARENTNETPROFITTZ",
        "KCFJCXSYJLR",
        "KCFJCXSYJLRTZ",
        "ROEJQ",
        "XSMLL",
        "XSJLL",
        "ZCFZL",
        "MGJYXJJE",
        "STAFF_NUM",
    ]
    return df[[column for column in columns if column in df]].head(10)


def latest_top_holders(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(lambda: ak.stock_main_stock_holder(stock=code), errors, "stock_main_stock_holder")
    if df.empty or "截至日期" not in df:
        return df
    latest = str(df["截至日期"].max())
    return df[df["截至日期"].astype(str) == latest].head(10).copy()


def latest_fund_holders(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(lambda: ak.stock_fund_stock_holder(symbol=code), errors, "stock_fund_stock_holder")
    if df.empty or "截止日期" not in df:
        return df
    latest = str(df["截止日期"].max())
    return df[df["截止日期"].astype(str) == latest].copy()


def fund_flow_frame(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(
        lambda: ak.stock_individual_fund_flow(stock=code, market=market(code)),
        errors,
        "stock_individual_fund_flow",
        attempts=5,
    )
    if df.empty:
        df = direct_eastmoney_fund_flow(code, errors)
    if df.empty or "日期" not in df:
        return df
    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期").reset_index(drop=True)


def direct_eastmoney_fund_flow(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    market_map = {"sh": 1, "sz": 0, "bj": 0}
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": "120",
        "klt": "101",
        "secid": f"{market_map[market(code)]}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "_": int(time.time() * 1000),
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json().get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame()
        frame = pd.DataFrame([item.split(",") for item in klines])
        frame.columns = [
            "日期",
            "主力净流入-净额",
            "小单净流入-净额",
            "中单净流入-净额",
            "大单净流入-净额",
            "超大单净流入-净额",
            "主力净流入-净占比",
            "小单净流入-净占比",
            "中单净流入-净占比",
            "大单净流入-净占比",
            "超大单净流入-净占比",
            "收盘价",
            "涨跌幅",
            "-",
            "--",
        ]
        keep = [
            "日期",
            "收盘价",
            "涨跌幅",
            "主力净流入-净额",
            "主力净流入-净占比",
            "超大单净流入-净额",
            "超大单净流入-净占比",
            "大单净流入-净额",
            "大单净流入-净占比",
            "中单净流入-净额",
            "中单净流入-净占比",
            "小单净流入-净额",
            "小单净流入-净占比",
        ]
        frame = frame[keep].copy()
        for column in keep[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame
    except Exception as exc:
        errors.append({"source": "direct_eastmoney_fund_flow", "reason": str(exc)[:240]})
        return pd.DataFrame()


def fund_flow_summary(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fund_flow_frame(code, errors)
    if df.empty:
        return df
    rows = []
    for window in [5, 10, 20, 60]:
        tail = df.tail(window)
        row = {"窗口": window, "开始日期": tail["日期"].min().strftime("%Y-%m-%d"), "结束日期": tail["日期"].max().strftime("%Y-%m-%d")}
        for column in df.columns:
            if "净流入-净额" in str(column):
                row[column] = pd.to_numeric(tail[column], errors="coerce").sum()
            if str(column).endswith("净占比"):
                row[f"{column}_均值"] = pd.to_numeric(tail[column], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def recent_fund_flow(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fund_flow_frame(code, errors)
    if df.empty:
        return df
    return df.tail(20)


def northbound_frame(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = fetch_frame(lambda: ak.stock_hsgt_individual_em(symbol=code), errors, "stock_hsgt_individual_em", attempts=4)
    if df.empty or "持股日期" not in df:
        return df
    df["持股日期"] = pd.to_datetime(df["持股日期"])
    return df.sort_values("持股日期").reset_index(drop=True)


def northbound_summary(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = northbound_frame(code, errors)
    if df.empty:
        return df
    rows = []
    for window in [5, 20, 60]:
        tail = df.tail(window)
        rows.append(
            {
                "窗口": window,
                "开始日期": tail["持股日期"].min().strftime("%Y-%m-%d"),
                "结束日期": tail["持股日期"].max().strftime("%Y-%m-%d"),
                "增持股数合计": pd.to_numeric(tail["今日增持股数"], errors="coerce").sum(),
                "增持资金合计": pd.to_numeric(tail["今日增持资金"], errors="coerce").sum(),
                "最新持股数量": pd.to_numeric(tail["持股数量"], errors="coerce").iloc[-1],
                "最新持股占A股百分比": pd.to_numeric(tail["持股数量占A股百分比"], errors="coerce").iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def recent_northbound(code: str, errors: list[dict[str, str]]) -> pd.DataFrame:
    df = northbound_frame(code, errors)
    if df.empty:
        return df
    return df.tail(20)


def to_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
