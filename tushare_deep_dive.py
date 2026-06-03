from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import tushare as ts


STOCKS = ["301218.SZ", "600936.SH"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Tushare fundamental and fund-flow data for selected stocks.")
    parser.add_argument("--output-dir", default="reports/tushare_deep_dive_20260603")
    parser.add_argument("--start-date", default="20250501")
    parser.add_argument("--end-date", default="20260603")
    parser.add_argument("--stocks", nargs="*", default=STOCKS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not set")
    pro = ts.pro_api(token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []

    smoke = safe_call(lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date"), errors, "stock_basic")
    if not smoke.empty:
        smoke[smoke["ts_code"].isin(args.stocks)].to_csv(output_dir / "stock_basic.csv", index=False, encoding="utf-8-sig")

    for ts_code in args.stocks:
        prefix = output_dir / ts_code.replace(".", "_")
        calls = {
            "stock_company": lambda code=ts_code: pro.stock_company(ts_code=code),
            "income": lambda code=ts_code: pro.income(ts_code=code, start_date="20230101", end_date=args.end_date),
            "balancesheet": lambda code=ts_code: pro.balancesheet(ts_code=code, start_date="20230101", end_date=args.end_date),
            "cashflow": lambda code=ts_code: pro.cashflow(ts_code=code, start_date="20230101", end_date=args.end_date),
            "fina_indicator": lambda code=ts_code: pro.fina_indicator(ts_code=code, start_date="20230101", end_date=args.end_date),
            "daily_basic": lambda code=ts_code: pro.daily_basic(ts_code=code, start_date=args.start_date, end_date=args.end_date),
            "moneyflow": lambda code=ts_code: pro.moneyflow(ts_code=code, start_date=args.start_date, end_date=args.end_date),
            "top10_holders": lambda code=ts_code: pro.top10_holders(ts_code=code, start_date="20240101", end_date=args.end_date),
            "top10_floatholders": lambda code=ts_code: pro.top10_floatholders(ts_code=code, start_date="20240101", end_date=args.end_date),
            "stk_holdernumber": lambda code=ts_code: pro.stk_holdernumber(ts_code=code, start_date="20240101", end_date=args.end_date),
            "hk_hold": lambda code=ts_code: pro.hk_hold(ts_code=code, start_date="20240101", end_date=args.end_date),
        }
        for name, call in calls.items():
            frame = safe_call(call, errors, f"{ts_code}:{name}")
            frame.to_csv(f"{prefix}_{name}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    print(output_dir)


def safe_call(callable_, errors: list[dict[str, str]], source: str) -> pd.DataFrame:
    try:
        frame = callable_()
        if frame is None:
            return pd.DataFrame()
        return frame.copy()
    except Exception as exc:
        errors.append({"source": source, "reason": str(exc)[:260]})
        return pd.DataFrame()


if __name__ == "__main__":
    main()
