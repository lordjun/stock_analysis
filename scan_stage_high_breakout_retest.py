from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

from darvas_weekly_backtest import fetch_daily_history_akshare_tx, normalize_daily_columns
from kline_model_research import (
    find_jianghua_acceleration_retests,
    find_stage_high_breakout_retests,
    normalize_price_bars,
)
from tdx_data import read_tdx_daily_bars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan daily A-share stage-high breakout retest candidates.")
    parser.add_argument("--start-date", default="20250101")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["stage_high_breakout_retest", "jianghua_acceleration_retest"],
        default="stage_high_breakout_retest",
    )
    parser.add_argument("--data-source", choices=["tx", "tdx", "auto"], default="auto")
    parser.add_argument("--tdx-vipdoc", default=r"C:\new_tdx\vipdoc")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lookback-bars", type=int, default=60)
    parser.add_argument("--min-base-bars", type=int, default=30)
    parser.add_argument("--max-retest-bars", type=int, default=10)
    parser.add_argument("--breakout-buffer", type=float, default=0.005)
    parser.add_argument("--support-tolerance", type=float, default=0.015)
    parser.add_argument("--min-pullback-pct", type=float, default=0.03)
    parser.add_argument("--max-extension-pct", type=float, default=0.08)
    parser.add_argument("--ma-fast", type=int, default=20)
    parser.add_argument("--ma-slow", type=int, default=60)
    parser.add_argument("--structure-lookback-bars", type=int, default=60)
    parser.add_argument("--max-peak-bars", type=int, default=5)
    parser.add_argument("--min-flagpole-pct", type=float, default=0.15)
    parser.add_argument("--max-flagpole-pct", type=float, default=0.45)
    parser.add_argument("--min-peak-drawdown-pct", type=float, default=0.10)
    parser.add_argument("--max-peak-drawdown-pct", type=float, default=0.28)
    parser.add_argument("--min-close-above-support-pct", type=float, default=0.02)
    parser.add_argument("--max-close-above-support-pct", type=float, default=0.09)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=1.5)
    parser.add_argument("--max-pullback-volume-ratio", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run(args)
    for path in output_paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    candidates = pd.DataFrame(rows).sort_values(["signal_date", "code"], ascending=[False, True]) if rows else pd.DataFrame(rows)
    candidates_path = output_dir / "candidates.csv"
    failures_path = output_dir / "failures.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(failures_path, index=False, encoding="utf-8-sig")
    print(f"DONE total={len(items)} candidates={len(rows)} failures={len(failures)}", flush=True)
    return [candidates_path, failures_path]


def scan_one(code: str, name: str, args: argparse.Namespace) -> dict[str, object] | None:
    daily = fetch_daily_bars(code, args)
    min_history = max(args.lookback_bars, args.structure_lookback_bars, args.ma_slow) + args.max_retest_bars + 1
    if len(daily) < min_history:
        return None

    if args.model == "jianghua_acceleration_retest":
        signals = find_jianghua_acceleration_retests(
            daily,
            structure_lookback_bars=args.structure_lookback_bars,
            min_base_bars=args.min_base_bars,
            max_peak_bars=args.max_peak_bars,
            max_retest_bars=args.max_retest_bars,
            breakout_buffer=args.breakout_buffer,
            support_tolerance=args.support_tolerance,
            min_flagpole_pct=args.min_flagpole_pct,
            max_flagpole_pct=args.max_flagpole_pct,
            min_peak_drawdown_pct=args.min_peak_drawdown_pct,
            max_peak_drawdown_pct=args.max_peak_drawdown_pct,
            min_close_above_support_pct=args.min_close_above_support_pct,
            max_close_above_support_pct=args.max_close_above_support_pct,
            min_breakout_volume_ratio=args.min_breakout_volume_ratio,
            max_pullback_volume_ratio=args.max_pullback_volume_ratio,
            ma_fast=args.ma_fast,
            ma_slow=args.ma_slow,
            first_retest_only=False,
        )
    else:
        signals = find_stage_high_breakout_retests(
            daily,
            lookback_bars=args.lookback_bars,
            min_base_bars=args.min_base_bars,
            max_retest_bars=args.max_retest_bars,
            breakout_buffer=args.breakout_buffer,
            support_tolerance=args.support_tolerance,
            min_pullback_pct=args.min_pullback_pct,
            max_extension_pct=args.max_extension_pct,
            ma_fast=args.ma_fast,
            ma_slow=args.ma_slow,
            first_retest_only=False,
        )
    if not signals:
        return None

    latest = daily.iloc[-1]
    latest_signals = [signal for signal in signals if signal.signal_index == len(daily) - 1]
    if not latest_signals:
        return None
    signal = max(latest_signals, key=lambda item: float(item.metadata.get("similarity_score", item.metadata.get("prior_high", 0.0))))

    metadata = signal.metadata
    breakout_index = int(metadata["breakout_index"])
    breakout_date = pd.Timestamp(daily.iloc[breakout_index]["date"])
    close = float(latest["close"])
    support_level = float(metadata.get("structure_high", metadata.get("prior_high")))
    return {
        "code": code,
        "name": name,
        "model": args.model,
        "signal_date": pd.Timestamp(signal.signal_date).strftime("%Y-%m-%d"),
        "latest_close": round(close, 3),
        "breakout_date": breakout_date.strftime("%Y-%m-%d"),
        "breakout_close": round(float(metadata["breakout_close"]), 3),
        "breakout_high": round(float(metadata["breakout_high"]), 3),
        "prior_high": round(support_level, 3),
        "structure_high": round(support_level, 3),
        "peak_high": round(float(metadata.get("peak_high", metadata.get("breakout_high"))), 3),
        "pullback_low": round(float(metadata["pullback_low"]), 3),
        "days_since_breakout": int(metadata["days_since_breakout"]),
        "days_since_peak": int(metadata.get("days_since_peak", 0)),
        "distance_to_prior_high_pct": round(
            float(metadata.get("close_above_support_pct", metadata.get("distance_to_prior_high"))) * 100,
            2,
        ),
        "flagpole_pct": round(float(metadata.get("flagpole_pct", 0.0)) * 100, 2),
        "peak_drawdown_pct": round(float(metadata.get("peak_drawdown_pct", 0.0)) * 100, 2),
        "drawdown_from_breakout_high_pct": round(float(metadata.get("drawdown_from_breakout_high", 0.0)) * 100, 2),
        "pullback_volume_ratio": round(float(metadata.get("pullback_volume_ratio", 0.0)), 3),
        "similarity_score": round(float(metadata.get("similarity_score", 0.0)), 2),
        "signal_initial_stop_price": round(float(signal.initial_stop), 3),
        "trigger_price": round(float(signal.trigger_price), 3),
        "ma_fast": round(float(metadata["ma_fast"]), 3),
        "ma_slow": round(float(metadata["ma_slow"]), 3),
        "data_source": daily.attrs.get("data_source", "unknown"),
        "note": f"{args.model}_daily; watchlist_only",
    }


def fetch_daily_bars(code: str, args: argparse.Namespace) -> pd.DataFrame:
    errors: list[str] = []
    if args.data_source in {"tx", "auto"} and code.startswith(("0", "3", "6")):
        try:
            bars = normalize_daily_columns(fetch_daily_history_akshare_tx(code, args.start_date, args.end_date))
            bars.attrs["data_source"] = "tx"
            return bars
        except Exception as exc:
            errors.append(f"tx: {exc}")
            if args.data_source == "tx":
                raise
    if args.data_source in {"tdx", "auto"}:
        try:
            bars = normalize_price_bars(read_tdx_daily_bars(args.tdx_vipdoc, code, args.start_date, args.end_date))
            bars.attrs["data_source"] = "tdx"
            return bars
        except Exception as exc:
            errors.append(f"tdx: {exc}")
    raise RuntimeError("; ".join(errors) or "no data source attempted")


if __name__ == "__main__":
    main()
