from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

from darvas_weekly_backtest import fetch_weekly_history_tx
from kline_model_research import aggregate_weekly_bars, find_signals_for_model, normalize_price_bars
from tdx_data import read_tdx_weekly_bars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan A-share K-line candidates with optional TDX local fallback.")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-source", choices=["tx", "tdx", "auto"], default="auto")
    parser.add_argument("--tdx-vipdoc", default=r"C:\new_tdx\vipdoc")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run(args)
    for path in output_paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not market_filter_passes(args.start_date, args.end_date):
        empty = output_dir / "candidates.csv"
        pd.DataFrame([]).to_csv(empty, index=False, encoding="utf-8-sig")
        return [empty]

    universe = ak.stock_info_a_code_name()
    universe = universe[~universe["name"].astype(str).str.upper().str.contains("ST", na=False)].reset_index(drop=True)
    if args.limit:
        universe = universe.head(args.limit)
    items = [(str(row.code).zfill(6), str(row.name)) for row in universe.itertuples(index=False)]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(scan_one, code, name, args): (code, name) for code, name in items}
        for done, future in enumerate(as_completed(futures), start=1):
            code, name = futures[future]
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception as exc:
                failures.append({"code": code, "name": name, "reason": str(exc)[:220]})
            if done % 250 == 0:
                print(f"scan_done {done}/{len(items)} candidates={len(rows)} failures={len(failures)}", flush=True)

    candidates = pd.DataFrame(rows).sort_values("code") if rows else pd.DataFrame(rows)
    candidates_path = output_dir / "candidates.csv"
    failures_path = output_dir / "failures.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(failures_path, index=False, encoding="utf-8-sig")
    print(f"DONE total={len(items)} candidates={len(rows)} failures={len(failures)}", flush=True)
    return [candidates_path, failures_path]


def market_filter_passes(start_date: str, end_date: str) -> bool:
    index = ak.stock_zh_index_daily(symbol="sh000300")
    index["date"] = pd.to_datetime(index["date"])
    index = index[(index["date"] >= pd.Timestamp(start_date)) & (index["date"] <= pd.Timestamp(end_date))]
    market = aggregate_weekly_bars(index[["date", "open", "high", "low", "close", "volume"]])
    market["ma10"] = market["close"].rolling(10).mean()
    market["ma20"] = market["close"].rolling(20).mean()
    market["ma40"] = market["close"].rolling(40).mean()
    row = market.iloc[-1]
    passes = bool(row["close"] > row["ma20"] > row["ma40"] and row["ma10"] > row["ma20"])
    print("market_pass", passes, "market_week", row["date"], "close", round(float(row["close"]), 2), flush=True)
    return passes


def scan_one(code: str, name: str, args: argparse.Namespace) -> dict[str, object] | None:
    weekly = fetch_weekly_bars(code, args)
    if len(weekly) < 60 or not stock_strength_ok(weekly):
        return None
    signals = find_signals_for_model(
        "trend_pullback_restart",
        weekly,
        {"ma_fast": 10, "ma_slow": 30, "pullback_tolerance": 0.02},
    )
    if not signals:
        return None
    signal = signals[-1]
    if signal.signal_index != len(weekly) - 1:
        return None

    latest = weekly.iloc[-1]
    close = float(latest["close"])
    return {
        "code": code,
        "name": name,
        "signal_date": pd.Timestamp(signal.signal_date).strftime("%Y-%m-%d"),
        "latest_week_date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "latest_close": close,
        "next_buy_reference": "next_open_after_signal",
        "stop_loss_ref_12pct_from_entry": "entry_price * 0.88",
        "profit_target_ref_15pct_from_entry": "entry_price * 1.15",
        "signal_initial_stop_price": round(float(signal.initial_stop), 3),
        "trigger_price": round(float(signal.trigger_price), 3),
        "data_source": weekly.attrs.get("data_source", "unknown"),
        "note": f"partial_week_signal_as_of_{args.end_date}_close; confirm_again_at_week_close",
    }


def fetch_weekly_bars(code: str, args: argparse.Namespace) -> pd.DataFrame:
    errors: list[str] = []
    if args.data_source in {"tx", "auto"}:
        try:
            bars = normalize_price_bars(fetch_weekly_history_tx(code, args.start_date, args.end_date))
            bars.attrs["data_source"] = "tx"
            return bars
        except Exception as exc:
            errors.append(f"tx: {exc}")
            if args.data_source == "tx":
                raise
    if args.data_source in {"tdx", "auto"}:
        try:
            bars = normalize_price_bars(read_tdx_weekly_bars(args.tdx_vipdoc, code, args.start_date, args.end_date))
            bars.attrs["data_source"] = "tdx"
            return bars
        except Exception as exc:
            errors.append(f"tdx: {exc}")
    raise RuntimeError("; ".join(errors) or "no data source attempted")


def stock_strength_ok(weekly: pd.DataFrame) -> bool:
    close = weekly["close"]
    if len(close) < 60 or close.iloc[-26] <= 0:
        return False
    ma20 = close.rolling(20).mean().iloc[-1]
    ma40 = close.rolling(40).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    return bool(
        close.iloc[-1] > ma20 > ma40 > ma60
        and close.iloc[-1] >= 0.95 * close.tail(52).max()
        and close.iloc[-1] / close.iloc[-26] - 1 > 0.35
    )


if __name__ == "__main__":
    main()
