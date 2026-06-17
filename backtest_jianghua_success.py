from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import struct
import time

import akshare as ak
import numpy as np
import pandas as pd
import requests
import tushare as ts

from kline_model_research import PatternSignal, normalize_price_bars
from tdx_data import read_tdx_daily_bars


MODEL_NAME = "jianghua_acceleration_retest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest Jianghua-style acceleration retest signals by one-month 30% success definition."
    )
    parser.add_argument("--signal-start-date", default="20210607")
    parser.add_argument("--signal-end-date", default="20260605")
    parser.add_argument("--history-start-date", default="20201201")
    parser.add_argument("--tdx-vipdoc", default=r"C:\new_tdx\vipdoc")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-return", type=float, default=0.30)
    parser.add_argument("--success-bars", type=int, default=20)
    parser.add_argument("--reversal-bars", type=int, default=60)
    parser.add_argument("--structure-lookback-bars", type=int, default=120)
    parser.add_argument("--min-base-bars", type=int, default=120)
    parser.add_argument("--max-peak-bars", type=int, default=5)
    parser.add_argument("--max-retest-bars", type=int, default=5)
    parser.add_argument("--max-days-since-peak", type=int, default=2)
    parser.add_argument("--breakout-buffer", type=float, default=0.005)
    parser.add_argument("--support-tolerance", type=float, default=0.02)
    parser.add_argument("--min-flagpole-pct", type=float, default=0.22)
    parser.add_argument("--max-flagpole-pct", type=float, default=0.45)
    parser.add_argument("--min-peak-drawdown-pct", type=float, default=0.16)
    parser.add_argument("--max-peak-drawdown-pct", type=float, default=0.28)
    parser.add_argument("--min-close-above-support-pct", type=float, default=0.02)
    parser.add_argument("--max-close-above-support-pct", type=float, default=0.09)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=1.5)
    parser.add_argument("--max-pullback-volume-ratio", type=float, default=0.70)
    parser.add_argument("--min-platform-turnover-pct", type=float, default=100.0)
    parser.add_argument("--min-platform-amplitude-pct", type=float, default=35.0)
    parser.add_argument("--max-platform-amplitude-pct", type=float, default=100.0)
    parser.add_argument("--max-platform-gain-pct", type=float, default=20.0)
    parser.add_argument("--disable-market-filter", action="store_true")
    parser.add_argument("--market-context-cache", default=None)
    parser.add_argument("--min-market-above-ma20-rate", type=float, default=0.45)
    parser.add_argument("--min-market-above-ma60-rate", type=float, default=0.35)
    parser.add_argument("--min-market-advance-rate", type=float, default=0.0)
    parser.add_argument("--allow-structural-bull", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-structural-ret120-p95", type=float, default=0.30)
    parser.add_argument("--min-structural-return-spread", type=float, default=0.35)
    parser.add_argument("--structural-code-pool-file", default=None)
    parser.add_argument("--float-share-cache", default=None)
    parser.add_argument("--float-share-provider", choices=["auto", "tdx", "tushare", "eastmoney"], default="auto")
    parser.add_argument("--tdx-base-dbf", default=r"C:\new_tdx\T0002\hq_cache\base.dbf")
    parser.add_argument("--tushare-token-env", default="TUSHARE_TOKEN")
    parser.add_argument("--ma-fast", type=int, default=20)
    parser.add_argument("--ma-slow", type=int, default=60)
    return parser


def main() -> None:
    paths = run(build_parser().parse_args())
    for path in paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir or _default_output_dir(args))
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = fetch_current_non_st_universe(args.limit)
    items = [(str(row.code).zfill(6), str(row.name)) for row in universe.itertuples(index=False)]
    args.float_share_by_code = load_float_share_by_code(args)
    args.structural_code_pool = load_structural_code_pool(args.structural_code_pool_file)
    args.market_context_by_date = {}
    if not args.disable_market_filter:
        market_context = load_or_build_market_context(items, args)
        args.market_context_by_date = _market_context_lookup(market_context)

    trade_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(backtest_one_stock, code, name, args): (code, name) for code, name in items}
        for done, future in enumerate(as_completed(futures), start=1):
            code, name = futures[future]
            try:
                trade_rows.extend(future.result())
            except Exception as exc:
                failures.append({"code": code, "name": name, "reason": str(exc)[:300]})
            if done % 250 == 0:
                print(
                    f"backtest_done {done}/{len(items)} trades={len(trade_rows)} failures={len(failures)}",
                    flush=True,
                )

    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades = trades.sort_values(["signal_date", "code", "entry_date"]).reset_index(drop=True)
    failures_df = pd.DataFrame(failures)
    summary_text = build_summary_markdown(trades, failures_df, args, len(items))

    trades_path = output_dir / "trades.csv"
    failures_path = output_dir / "failures.csv"
    summary_path = output_dir / "summary.md"
    annual_path = output_dir / "annual_summary.csv"
    similarity_path = output_dir / "similarity_summary.csv"

    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(summary_text, encoding="utf-8")
    build_annual_summary(trades).to_csv(annual_path, index=False, encoding="utf-8-sig")
    build_similarity_summary(trades).to_csv(similarity_path, index=False, encoding="utf-8-sig")

    print(
        f"DONE stocks={len(items)} trades={len(trades)} failures={len(failures)} output={output_dir}",
        flush=True,
    )
    return [summary_path, trades_path, annual_path, similarity_path, failures_path]


def fetch_current_non_st_universe(limit: int | None = None) -> pd.DataFrame:
    universe = ak.stock_info_a_code_name()
    universe = universe[~universe["name"].astype(str).str.upper().str.contains("ST", na=False)].reset_index(drop=True)
    if limit:
        universe = universe.head(limit)
    return universe


def load_float_share_by_code(args: argparse.Namespace) -> dict[str, float]:
    cache_path = Path(args.float_share_cache or f"data/cache/float_share/float_share_{args.signal_end_date}.csv")
    if cache_path.exists():
        raw = pd.read_csv(cache_path, dtype={"ts_code": str})
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw = fetch_float_share_snapshot(args)
        raw.to_csv(cache_path, index=False, encoding="utf-8-sig")
    if raw.empty or "ts_code" not in raw.columns or "float_share" not in raw.columns:
        raise RuntimeError(f"invalid float share cache: {cache_path}")
    raw["code"] = raw["ts_code"].astype(str).str.slice(0, 6)
    raw["float_share"] = pd.to_numeric(raw["float_share"], errors="coerce")
    raw = raw.dropna(subset=["float_share"])
    return {str(row.code).zfill(6): float(row.float_share) for row in raw.itertuples(index=False)}


def fetch_float_share_snapshot(args: argparse.Namespace) -> pd.DataFrame:
    if args.float_share_provider in {"auto", "tdx"}:
        try:
            return read_tdx_base_float_share(args.tdx_base_dbf)
        except Exception as exc:
            if args.float_share_provider == "tdx":
                raise
            print(f"float_share_tdx_failed {exc}; fallback=tushare/eastmoney", flush=True)
    if args.float_share_provider in {"auto", "tushare"}:
        token = os.environ.get(args.tushare_token_env)
        if token:
            try:
                pro = ts.pro_api(token)
                raw = pro.bak_basic(trade_date=args.signal_end_date, fields="ts_code,name,float_share,total_share")
                raw["float_share"] = pd.to_numeric(raw["float_share"], errors="coerce") * 10000
                return raw
            except Exception as exc:
                if args.float_share_provider == "tushare":
                    raise
                print(f"float_share_tushare_failed {exc}; fallback=eastmoney", flush=True)
        elif args.float_share_provider == "tushare":
            raise RuntimeError(f"missing {args.tushare_token_env}; cannot fetch float_share from tushare")
    return fetch_float_share_eastmoney()


def fetch_float_share_eastmoney() -> pd.DataFrame:
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    rows: list[dict[str, object]] = []
    page = 1
    page_size = 100
    total = None
    while total is None or len(rows) < total:
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "fields": "f12,f14,f2,f20,f21",
        }
        last_error = None
        for attempt in range(5):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=20)
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or {}
                total = int(data.get("total") or 0)
                batch = data.get("diff") or []
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(f"eastmoney float share snapshot failed: {last_error}")
        if not batch:
            break
        rows.extend(batch)
        page += 1
        time.sleep(0.2)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("eastmoney float share snapshot returned no rows")
    raw["latest_price"] = pd.to_numeric(raw["f2"], errors="coerce")
    raw["float_mv"] = pd.to_numeric(raw["f21"], errors="coerce")
    raw = raw[(raw["latest_price"] > 0) & (raw["float_mv"] > 0)].copy()
    raw["float_share"] = raw["float_mv"] / raw["latest_price"] / 10000
    raw["ts_code"] = raw["f12"].astype(str).str.zfill(6)
    raw["name"] = raw["f14"].astype(str)
    raw = raw.drop_duplicates(subset=["ts_code"], keep="first")
    return raw[["ts_code", "name", "float_share"]].reset_index(drop=True)


def read_tdx_base_float_share(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    data = file_path.read_bytes()
    record_count = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]
    fields: list[tuple[str, str, int, int, int]] = []
    field_offset = 1
    offset = 32
    while offset < header_len and data[offset] != 0x0D:
        name = data[offset : offset + 11].split(b"\x00", 1)[0].decode("gbk", "ignore")
        field_type = chr(data[offset + 11])
        field_len = data[offset + 16]
        decimals = data[offset + 17]
        fields.append((name, field_type, field_len, decimals, field_offset))
        field_offset += field_len
        offset += 32

    required = {"GPDM", "LTAG"}
    available = {name for name, *_ in fields}
    missing = required - available
    if missing:
        raise RuntimeError(f"TDX base dbf missing fields: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for index in range(record_count):
        record = data[header_len + index * record_len : header_len + (index + 1) * record_len]
        if not record or record[0:1] == b"*":
            continue
        values: dict[str, str] = {}
        for name, _field_type, field_len, _decimals, position in fields:
            values[name] = record[position : position + field_len].decode("gbk", "ignore").strip()
        code = values.get("GPDM", "").zfill(6)
        float_share = pd.to_numeric(values.get("LTAG", ""), errors="coerce")
        if code and pd.notna(float_share) and float(float_share) > 0:
            rows.append({"ts_code": code, "name": "", "float_share": float(float_share)})
    return pd.DataFrame(rows)


def attach_turnover_rate_from_float_share(
    bars: pd.DataFrame,
    code: str,
    float_share_by_code: dict[str, float],
) -> pd.DataFrame:
    float_share_10k = float_share_by_code.get(str(code).zfill(6))
    if not float_share_10k or float_share_10k <= 0:
        output = bars.copy()
        output["turnover_rate"] = np.nan
        return output
    output = bars.copy()
    output["turnover_rate"] = pd.to_numeric(output["volume"], errors="coerce") / (float_share_10k * 100)
    return output


def load_structural_code_pool(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(str(pool_path))
    raw = pd.read_csv(pool_path, dtype=str)
    if raw.empty:
        return set()
    code_column = next((column for column in ["code", "ts_code", "stock_code", "证券代码", "股票代码"] if column in raw.columns), None)
    if code_column is None:
        raise RuntimeError(f"structural code pool missing code column: {pool_path}")
    codes = raw[code_column].astype(str).str.extract(r"(\d{6})", expand=False).dropna()
    return {str(code).zfill(6) for code in codes}


def load_or_build_market_context(items: list[tuple[str, str]], args: argparse.Namespace) -> pd.DataFrame:
    universe_tag = f"limit{args.limit}" if args.limit else "all"
    cache_path = Path(
        args.market_context_cache
        or f"data/cache/market_context/tdx_breadth_{universe_tag}_{args.history_start_date}_{args.signal_end_date}.csv"
    )
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"date": str})
        if _market_context_has_required_columns(cached):
            return cached
        print(f"market_context_cache_stale path={cache_path}; rebuild=missing_structural_columns", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(read_tdx_daily_bars, args.tdx_vipdoc, code, args.history_start_date, args.signal_end_date): code
            for code, _name in items
        }
        for done, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                frames[code] = normalize_price_bars(future.result())
            except Exception:
                failures += 1
            if done % 500 == 0:
                print(f"market_context_done {done}/{len(items)} failures={failures}", flush=True)

    context = build_market_context_from_frames(frames)
    context.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"market_context_cached rows={len(context)} failures={failures} path={cache_path}", flush=True)
    return context


def build_market_context_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    component_frames: list[pd.DataFrame] = []
    for bars in frames.values():
        if bars.empty:
            continue
        df = normalize_price_bars(bars)
        if len(df) < 60:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        work = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
                "advance": close.gt(close.shift(1)).astype("int8"),
                "above_ma20": close.gt(close.rolling(20).mean()).astype("int8"),
                "above_ma60": close.gt(close.rolling(60).mean()).astype("int8"),
                "new_high_60": close.ge(close.rolling(60).max()).astype("int8"),
                "ret120": close / close.shift(120) - 1,
                "valid": close.rolling(60).mean().notna().astype("int8"),
            }
        )
        work = work[work["valid"] == 1]
        if not work.empty:
            component_frames.append(work)
    if not component_frames:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_count",
                "advance_rate",
                "above_ma20_rate",
                "above_ma60_rate",
                "new_high_60_rate",
                "ret120_median",
                "ret120_p95",
                "ret120_spread_p95_median",
            ]
        )

    components = pd.concat(component_frames, ignore_index=True)
    grouped = components.groupby("date", sort=True)
    context = grouped.agg(
        stock_count=("valid", "sum"),
        advances=("advance", "sum"),
        above_ma20=("above_ma20", "sum"),
        above_ma60=("above_ma60", "sum"),
        new_high_60=("new_high_60", "sum"),
    ).reset_index()
    context["advance_rate"] = context["advances"] / context["stock_count"]
    context["above_ma20_rate"] = context["above_ma20"] / context["stock_count"]
    context["above_ma60_rate"] = context["above_ma60"] / context["stock_count"]
    context["new_high_60_rate"] = context["new_high_60"] / context["stock_count"]
    ret120 = components.dropna(subset=["ret120"]).groupby("date", sort=True)["ret120"]
    context = context.merge(
        ret120.agg(
            ret120_median="median",
            ret120_p95=lambda series: float(series.quantile(0.95)),
        ).reset_index(),
        on="date",
        how="left",
    )
    context["ret120_spread_p95_median"] = context["ret120_p95"] - context["ret120_median"]
    return context[
        [
            "date",
            "stock_count",
            "advance_rate",
            "above_ma20_rate",
            "above_ma60_rate",
            "new_high_60_rate",
            "ret120_median",
            "ret120_p95",
            "ret120_spread_p95_median",
        ]
    ].reset_index(drop=True)


def market_context_passes_filter(
    context: dict[str, object] | None,
    min_above_ma20_rate: float,
    min_above_ma60_rate: float,
    min_advance_rate: float,
    allow_structural_bull: bool = True,
    min_structural_ret120_p95: float = 0.30,
    min_structural_return_spread: float = 0.35,
    code: str | None = None,
    structural_code_pool: set[str] | None = None,
) -> bool:
    regime = classify_market_regime(
        context,
        min_above_ma20_rate,
        min_above_ma60_rate,
        min_advance_rate,
        allow_structural_bull,
        min_structural_ret120_p95,
        min_structural_return_spread,
    )
    if regime == "broad_bull":
        return True
    if regime != "structural_bull":
        return False
    if not code or not structural_code_pool:
        return False
    return str(code).zfill(6) in structural_code_pool


def classify_market_regime(
    context: dict[str, object] | None,
    min_above_ma20_rate: float,
    min_above_ma60_rate: float,
    min_advance_rate: float,
    allow_structural_bull: bool,
    min_structural_ret120_p95: float,
    min_structural_return_spread: float,
) -> str:
    if not context:
        return "unknown"

    broad_bull = (
        float(context.get("above_ma20_rate", 0.0)) >= min_above_ma20_rate
        and float(context.get("above_ma60_rate", 0.0)) >= min_above_ma60_rate
        and float(context.get("advance_rate", 0.0)) >= min_advance_rate
    )
    if broad_bull:
        return "broad_bull"

    ret120_p95 = float(context.get("ret120_p95", float("nan")))
    ret120_median = float(context.get("ret120_median", float("nan")))
    spread = float(context.get("ret120_spread_p95_median", ret120_p95 - ret120_median))
    structural_bull = (
        allow_structural_bull
        and pd.notna(ret120_p95)
        and pd.notna(spread)
        and ret120_p95 >= min_structural_ret120_p95
        and spread >= min_structural_return_spread
    )
    return "structural_bull" if structural_bull else "weak_or_no_trend"


def _market_context_has_required_columns(context: pd.DataFrame) -> bool:
    required = {
        "date",
        "stock_count",
        "advance_rate",
        "above_ma20_rate",
        "above_ma60_rate",
        "new_high_60_rate",
        "ret120_median",
        "ret120_p95",
        "ret120_spread_p95_median",
    }
    return required.issubset(context.columns)


def _market_context_lookup(context: pd.DataFrame) -> dict[str, dict[str, object]]:
    if context.empty:
        return {}
    work = context.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    return {str(row["date"]): row for row in work.to_dict("records")}


def backtest_one_stock(code: str, name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    bars = normalize_price_bars(read_tdx_daily_bars(args.tdx_vipdoc, code, args.history_start_date, args.signal_end_date))
    bars = attach_turnover_rate_from_float_share(bars, code, args.float_share_by_code)
    if len(bars) < args.structure_lookback_bars + args.max_retest_bars + args.reversal_bars:
        return []

    signals = find_jianghua_acceleration_retests_fast(
        bars,
        structure_lookback_bars=args.structure_lookback_bars,
        min_base_bars=args.min_base_bars,
        max_peak_bars=args.max_peak_bars,
        max_retest_bars=args.max_retest_bars,
        max_days_since_peak=args.max_days_since_peak,
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
        min_platform_turnover_pct=args.min_platform_turnover_pct,
        min_platform_amplitude_pct=args.min_platform_amplitude_pct,
        max_platform_amplitude_pct=args.max_platform_amplitude_pct,
        max_platform_gain_pct=args.max_platform_gain_pct,
        ma_fast=args.ma_fast,
        ma_slow=args.ma_slow,
    )

    signal_start = pd.Timestamp(args.signal_start_date)
    signal_end = pd.Timestamp(args.signal_end_date)
    rows: list[dict[str, object]] = []
    next_allowed_signal_index = 0
    for signal in sorted(signals, key=lambda item: item.signal_index):
        signal_date = pd.Timestamp(signal.signal_date)
        if signal_date < signal_start or signal_date > signal_end:
            continue
        if signal.signal_index < next_allowed_signal_index:
            continue
        market_context = args.market_context_by_date.get(_fmt_date(signal_date))
        market_regime = classify_market_regime(
            market_context,
            args.min_market_above_ma20_rate,
            args.min_market_above_ma60_rate,
            args.min_market_advance_rate,
            args.allow_structural_bull,
            args.min_structural_ret120_p95,
            args.min_structural_return_spread,
        )
        if not args.disable_market_filter and not market_context_passes_filter(
            market_context,
            args.min_market_above_ma20_rate,
            args.min_market_above_ma60_rate,
            args.min_market_advance_rate,
            args.allow_structural_bull,
            args.min_structural_ret120_p95,
            args.min_structural_return_spread,
            code=code,
            structural_code_pool=args.structural_code_pool,
        ):
            continue
        row = evaluate_signal(code, name, bars, signal, args.target_return, args.success_bars, args.reversal_bars)
        if row is None:
            continue
        if market_context:
            row.update(
                {
                    "market_regime": market_regime,
                    "in_structural_code_pool": str(code).zfill(6) in args.structural_code_pool,
                    "market_stock_count": int(market_context["stock_count"]),
                    "market_advance_rate": round(float(market_context["advance_rate"]), 4),
                    "market_above_ma20_rate": round(float(market_context["above_ma20_rate"]), 4),
                    "market_above_ma60_rate": round(float(market_context["above_ma60_rate"]), 4),
                    "market_new_high_60_rate": round(float(market_context["new_high_60_rate"]), 4),
                    "market_ret120_median": round(float(market_context.get("ret120_median", 0.0)), 4),
                    "market_ret120_p95": round(float(market_context.get("ret120_p95", 0.0)), 4),
                    "market_ret120_spread_p95_median": round(
                        float(market_context.get("ret120_spread_p95_median", 0.0)),
                        4,
                    ),
                }
            )
        rows.append(row)
        next_allowed_signal_index = int(row["entry_index"]) + args.success_bars
    return rows


def find_jianghua_acceleration_retests_fast(
    bars: pd.DataFrame,
    structure_lookback_bars: int = 120,
    min_base_bars: int = 120,
    max_peak_bars: int = 5,
    max_retest_bars: int = 5,
    max_days_since_peak: int = 2,
    breakout_buffer: float = 0.005,
    support_tolerance: float = 0.02,
    min_flagpole_pct: float = 0.22,
    max_flagpole_pct: float = 0.45,
    min_peak_drawdown_pct: float = 0.16,
    max_peak_drawdown_pct: float = 0.28,
    min_close_above_support_pct: float = 0.02,
    max_close_above_support_pct: float = 0.09,
    min_breakout_volume_ratio: float = 1.5,
    max_pullback_volume_ratio: float = 0.70,
    min_platform_turnover_pct: float = 100.0,
    min_platform_amplitude_pct: float = 35.0,
    max_platform_amplitude_pct: float = 100.0,
    max_platform_gain_pct: float = 20.0,
    ma_fast: int = 20,
    ma_slow: int = 60,
    steady_max_retest_bars: int = 15,
    steady_min_breakout_volume_ratio: float = 1.10,
    steady_min_climb_pct: float = 0.08,
    steady_max_climb_pct: float = 0.25,
    steady_support_tolerance: float = 0.03,
    steady_max_drawdown_pct: float = 0.12,
    steady_min_close_above_support_pct: float = 0.02,
    steady_max_close_above_support_pct: float = 0.18,
    steady_max_pullback_volume_ratio: float = 1.30,
    steady_ma_slow_tolerance: float = 0.005,
) -> list[PatternSignal]:
    df = normalize_price_bars(bars)
    if df.empty:
        return []
    df = _attach_optional_turnover_column(df, bars)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"]).to_numpy()

    structure_high = df["close"].rolling(structure_lookback_bars).max().shift(1).to_numpy(dtype=float)
    base_low = df["low"].rolling(min_base_bars).min().shift(1).to_numpy(dtype=float)
    platform_high = df["high"].rolling(min_base_bars).max().shift(1).to_numpy(dtype=float)
    platform_low = df["low"].rolling(min_base_bars).min().shift(1).to_numpy(dtype=float)
    platform_start_close = df["close"].shift(min_base_bars).to_numpy(dtype=float)
    platform_end_close = df["close"].shift(1).to_numpy(dtype=float)
    platform_turnover = (
        df["turnover_rate"].rolling(min_base_bars).sum().shift(1).to_numpy(dtype=float)
        if "turnover_rate" in df.columns
        else np.full(len(df), np.nan)
    )
    ma_fast_values = df["close"].rolling(ma_fast).mean().to_numpy(dtype=float)
    ma_slow_values = df["close"].rolling(ma_slow).mean().to_numpy(dtype=float)
    prior_avg_volume = df["volume"].rolling(20).mean().shift(1).to_numpy(dtype=float)
    platform_ok = _platform_mask(
        platform_high,
        platform_low,
        platform_start_close,
        platform_end_close,
        platform_turnover,
        min_platform_turnover_pct,
        min_platform_amplitude_pct,
        max_platform_amplitude_pct,
        max_platform_gain_pct,
    )

    first_breakout = max(structure_lookback_bars, min_base_bars, ma_slow)
    breakout_mask = (
        (np.arange(len(df)) >= first_breakout)
        & (np.arange(len(df)) < len(df) - 2)
        & np.isfinite(structure_high)
        & np.isfinite(ma_fast_values)
        & np.isfinite(ma_slow_values)
        & np.isfinite(prior_avg_volume)
        & (prior_avg_volume > 0)
        & (close >= structure_high * (1 + breakout_buffer))
        & (close > ma_fast_values)
        & (ma_fast_values > ma_slow_values)
        & platform_ok
    )

    signals: list[PatternSignal] = []
    for breakout_index in np.flatnonzero(breakout_mask):
        latest_peak_search_index = min(len(df), breakout_index + max_peak_bars + 1)
        latest_retest_index = min(len(df), breakout_index + max_retest_bars + 1)
        for retest_index in range(breakout_index + 2, latest_retest_index):
            impulse_end = min(retest_index, latest_peak_search_index)
            if impulse_end <= breakout_index:
                continue
            impulse_high = high[breakout_index:impulse_end]
            peak_index = int(breakout_index + np.argmax(impulse_high))
            if peak_index <= breakout_index or peak_index >= retest_index:
                continue
            if retest_index - peak_index > max_days_since_peak:
                continue

            peak_high = float(high[peak_index])
            flagpole_pct = peak_high / float(structure_high[breakout_index]) - 1
            if not (min_flagpole_pct <= flagpole_pct <= max_flagpole_pct):
                continue

            pullback_low = float(np.min(low[breakout_index + 1 : retest_index + 1]))
            latest_close = float(close[retest_index])
            peak_drawdown_pct = 1 - pullback_low / peak_high
            close_above_support_pct = latest_close / float(structure_high[breakout_index]) - 1
            impulse_volume = float(np.mean(volume[breakout_index : peak_index + 1]))
            pullback_volume = float(np.mean(volume[peak_index + 1 : retest_index + 1]))

            held_structure_high = pullback_low >= float(structure_high[breakout_index]) * (1 - support_tolerance)
            drawdown_ok = min_peak_drawdown_pct <= peak_drawdown_pct <= max_peak_drawdown_pct
            location_ok = min_close_above_support_pct <= close_above_support_pct <= max_close_above_support_pct
            volume_ok = (
                impulse_volume >= float(prior_avg_volume[breakout_index]) * min_breakout_volume_ratio
                and pullback_volume <= impulse_volume * max_pullback_volume_ratio
            )
            if not (held_structure_high and drawdown_ok and location_ok and volume_ok):
                continue

            pullback_volume_ratio = pullback_volume / impulse_volume if impulse_volume else 0.0
            breakout_volume_ratio = float(volume[breakout_index]) / float(prior_avg_volume[breakout_index])
            signals.append(
                PatternSignal(
                    model=MODEL_NAME,
                    signal_index=int(retest_index),
                    signal_date=pd.Timestamp(dates[retest_index]),
                    trigger_price=latest_close,
                    initial_stop=float(structure_high[breakout_index]) * (1 - support_tolerance),
                    metadata={
                        "breakout_index": float(breakout_index),
                        "peak_index": float(peak_index),
                        "structure_high": float(structure_high[breakout_index]),
                        "base_low": float(base_low[breakout_index]),
                        "breakout_close": float(close[breakout_index]),
                        "breakout_high": float(high[breakout_index]),
                        "peak_high": peak_high,
                        "pullback_low": pullback_low,
                        "days_since_breakout": float(retest_index - breakout_index),
                        "days_since_peak": float(retest_index - peak_index),
                        "flagpole_pct": flagpole_pct,
                        "peak_drawdown_pct": peak_drawdown_pct,
                        "close_above_support_pct": close_above_support_pct,
                        "prior_avg_volume": float(prior_avg_volume[breakout_index]),
                        "impulse_volume": impulse_volume,
                        "pullback_volume": pullback_volume,
                        "pullback_volume_ratio": pullback_volume_ratio,
                        "breakout_volume_ratio": breakout_volume_ratio,
                        "platform_turnover_pct": float(platform_turnover[breakout_index]),
                        "platform_amplitude_pct": float(platform_high[breakout_index] / platform_low[breakout_index] - 1) * 100,
                        "platform_gain_pct": float(platform_end_close[breakout_index] / platform_start_close[breakout_index] - 1) * 100,
                        "pattern_subtype": "acceleration_retest",
                        "similarity_score": _jianghua_similarity_score(
                            flagpole_pct=flagpole_pct,
                            peak_drawdown_pct=peak_drawdown_pct,
                            close_above_support_pct=close_above_support_pct,
                            days_since_breakout=retest_index - breakout_index,
                            pullback_volume_ratio=pullback_volume_ratio,
                        ),
                        "ma_fast": float(ma_fast_values[breakout_index]),
                        "ma_slow": float(ma_slow_values[breakout_index]),
                    },
                )
            )
            break

    steady_breakout_mask = (
        (np.arange(len(df)) >= first_breakout)
        & (np.arange(len(df)) < len(df) - 2)
        & np.isfinite(structure_high)
        & np.isfinite(ma_fast_values)
        & np.isfinite(ma_slow_values)
        & np.isfinite(prior_avg_volume)
        & (prior_avg_volume > 0)
        & (close >= structure_high * (1 + breakout_buffer))
        & (close > ma_fast_values)
        & (close > ma_slow_values)
        & (ma_fast_values >= ma_slow_values * (1 - steady_ma_slow_tolerance))
        & (volume >= prior_avg_volume * steady_min_breakout_volume_ratio)
        & platform_ok
    )
    signaled_indices = {signal.signal_index for signal in signals}
    for breakout_index in np.flatnonzero(steady_breakout_mask):
        latest_retest_index = min(len(df), breakout_index + steady_max_retest_bars + 1)
        for retest_index in range(breakout_index + 2, latest_retest_index):
            if retest_index in signaled_indices:
                continue
            climb_highs = high[breakout_index + 1 : retest_index + 1]
            if len(climb_highs) == 0:
                continue
            peak_index = int(breakout_index + 1 + np.argmax(climb_highs))
            if peak_index >= retest_index:
                continue

            support_level = float(structure_high[breakout_index])
            peak_high = float(high[peak_index])
            climb_pct = peak_high / support_level - 1
            if not (steady_min_climb_pct <= climb_pct <= steady_max_climb_pct):
                continue

            pullback_low = float(np.min(low[breakout_index + 1 : retest_index + 1]))
            post_peak_low = float(np.min(low[peak_index : retest_index + 1]))
            peak_drawdown_pct = 1 - post_peak_low / peak_high
            close_above_support_pct = float(close[retest_index]) / support_level - 1
            impulse_volume = float(np.mean(volume[breakout_index : peak_index + 1]))
            pullback_volume = float(np.mean(volume[peak_index + 1 : retest_index + 1]))
            pullback_volume_ratio = pullback_volume / impulse_volume if impulse_volume else 0.0
            breakout_volume_ratio = float(volume[breakout_index]) / float(prior_avg_volume[breakout_index])
            trend_window = close[breakout_index : retest_index + 1] > ma_fast_values[breakout_index : retest_index + 1]
            trend_quality = float(np.mean(trend_window)) if len(trend_window) else 0.0

            if pullback_low < support_level * (1 - steady_support_tolerance):
                continue
            if peak_drawdown_pct > steady_max_drawdown_pct:
                continue
            if not (steady_min_close_above_support_pct <= close_above_support_pct <= steady_max_close_above_support_pct):
                continue
            if pullback_volume_ratio > steady_max_pullback_volume_ratio:
                continue
            if trend_quality < 0.60:
                continue

            signals.append(
                PatternSignal(
                    model=MODEL_NAME,
                    signal_index=int(retest_index),
                    signal_date=pd.Timestamp(dates[retest_index]),
                    trigger_price=float(close[retest_index]),
                    initial_stop=support_level * (1 - steady_support_tolerance),
                    metadata={
                        "breakout_index": float(breakout_index),
                        "peak_index": float(peak_index),
                        "structure_high": support_level,
                        "base_low": float(base_low[breakout_index]),
                        "breakout_close": float(close[breakout_index]),
                        "breakout_high": float(high[breakout_index]),
                        "peak_high": peak_high,
                        "pullback_low": pullback_low,
                        "days_since_breakout": float(retest_index - breakout_index),
                        "days_since_peak": float(retest_index - peak_index),
                        "flagpole_pct": climb_pct,
                        "peak_drawdown_pct": peak_drawdown_pct,
                        "close_above_support_pct": close_above_support_pct,
                        "prior_avg_volume": float(prior_avg_volume[breakout_index]),
                        "impulse_volume": impulse_volume,
                        "pullback_volume": pullback_volume,
                        "pullback_volume_ratio": pullback_volume_ratio,
                        "breakout_volume_ratio": breakout_volume_ratio,
                        "platform_turnover_pct": float(platform_turnover[breakout_index]),
                        "platform_amplitude_pct": float(platform_high[breakout_index] / platform_low[breakout_index] - 1) * 100,
                        "platform_gain_pct": float(platform_end_close[breakout_index] / platform_start_close[breakout_index] - 1) * 100,
                        "pattern_subtype": "steady_climb_retest",
                        "similarity_score": _steady_climb_similarity_score(
                            climb_pct=climb_pct,
                            peak_drawdown_pct=peak_drawdown_pct,
                            close_above_support_pct=close_above_support_pct,
                            days_since_breakout=retest_index - breakout_index,
                            pullback_volume_ratio=pullback_volume_ratio,
                        ),
                        "ma_fast": float(ma_fast_values[breakout_index]),
                        "ma_slow": float(ma_slow_values[breakout_index]),
                    },
                )
            )
            signaled_indices.add(int(retest_index))
            break
    return signals


def evaluate_signal(
    code: str,
    name: str,
    bars: pd.DataFrame,
    signal: PatternSignal,
    target_return: float = 0.30,
    success_bars: int = 20,
    reversal_bars: int = 60,
) -> dict[str, object] | None:
    entry_index = signal.signal_index + 1
    if entry_index >= len(bars):
        return None

    entry = bars.iloc[entry_index]
    entry_price = float(entry["open"])
    success_window = bars.iloc[entry_index : min(len(bars), entry_index + success_bars)].copy()
    if success_window.empty:
        return None

    target_price = entry_price * (1 + target_return)
    hit_window = success_window[success_window["high"] >= target_price]
    success = not hit_window.empty
    success_date = pd.Timestamp(hit_window.iloc[0]["date"]) if success else pd.NaT
    bars_to_30 = int(hit_window.index[0] - entry_index + 1) if success else None

    reversal_window = bars.iloc[entry_index : min(len(bars), entry_index + reversal_bars)].copy()
    peak_index_20 = int(success_window["high"].idxmax())
    peak_index_60 = int(reversal_window["high"].idxmax())
    peak_high_60 = float(bars.at[peak_index_60, "high"])
    after_peak = bars.iloc[peak_index_60 + 1 : min(len(bars), peak_index_60 + 21)].copy()

    ma_frame = bars.copy()
    ma_frame["ma5"] = ma_frame["close"].rolling(5).mean()
    ma_frame["ma10"] = ma_frame["close"].rolling(10).mean()
    ma_after_peak = ma_frame.iloc[peak_index_60 + 1 : min(len(ma_frame), peak_index_60 + 21)]

    metadata = signal.metadata
    row = {
        "code": code,
        "name": name,
        "model": MODEL_NAME,
        "pattern_subtype": str(metadata.get("pattern_subtype", "")),
        "signal_index": int(signal.signal_index),
        "entry_index": int(entry_index),
        "signal_date": _fmt_date(signal.signal_date),
        "entry_date": _fmt_date(entry["date"]),
        "entry_price": round(entry_price, 4),
        "target_price": round(target_price, 4),
        "success_20d_30pct": bool(success),
        "success_date": _fmt_date(success_date) if success else "",
        "bars_to_30pct": bars_to_30,
        "max_high_20d": round(float(success_window["high"].max()), 4),
        "max_return_20d_pct": round((float(success_window["high"].max()) / entry_price - 1) * 100, 2),
        "close_return_20d_pct": round((float(success_window.iloc[-1]["close"]) / entry_price - 1) * 100, 2),
        "min_low_20d": round(float(success_window["low"].min()), 4),
        "max_drawdown_from_entry_20d_pct": round((float(success_window["low"].min()) / entry_price - 1) * 100, 2),
        "peak_date_20d": _fmt_date(bars.at[peak_index_20, "date"]),
        "peak_bar_20d": int(peak_index_20 - entry_index + 1),
        "max_high_60d": round(peak_high_60, 4),
        "max_return_60d_pct": round((peak_high_60 / entry_price - 1) * 100, 2),
        "peak_date_60d": _fmt_date(bars.at[peak_index_60, "date"]),
        "peak_bar_60d": int(peak_index_60 - entry_index + 1),
        "after_peak_5d_low_drawdown_pct": _future_low_drawdown(after_peak, peak_high_60, 5),
        "after_peak_10d_low_drawdown_pct": _future_low_drawdown(after_peak, peak_high_60, 10),
        "after_peak_20d_low_drawdown_pct": _future_low_drawdown(after_peak, peak_high_60, 20),
        "first_10pct_drawdown_after_peak_date": _first_peak_drawdown_date(after_peak, peak_high_60, 0.10),
        "first_15pct_drawdown_after_peak_date": _first_peak_drawdown_date(after_peak, peak_high_60, 0.15),
        "first_close_below_ma5_after_peak_date": _first_close_below_ma_date(ma_after_peak, "ma5"),
        "first_close_below_ma10_after_peak_date": _first_close_below_ma_date(ma_after_peak, "ma10"),
        "signal_initial_stop_price": round(float(signal.initial_stop), 4),
        "structure_high": round(float(metadata["structure_high"]), 4),
        "breakout_index": int(metadata["breakout_index"]),
        "breakout_date": _fmt_date(bars.at[int(metadata["breakout_index"]), "date"]),
        "breakout_close": round(float(metadata["breakout_close"]), 4),
        "breakout_high": round(float(metadata["breakout_high"]), 4),
        "peak_high_pattern": round(float(metadata["peak_high"]), 4),
        "pullback_low_pattern": round(float(metadata["pullback_low"]), 4),
        "days_since_breakout": int(metadata["days_since_breakout"]),
        "days_since_peak": int(metadata["days_since_peak"]),
        "flagpole_pct": round(float(metadata["flagpole_pct"]) * 100, 2),
        "peak_drawdown_pct": round(float(metadata["peak_drawdown_pct"]) * 100, 2),
        "close_above_support_pct": round(float(metadata["close_above_support_pct"]) * 100, 2),
        "prior_avg_volume": round(float(metadata["prior_avg_volume"]), 2),
        "breakout_volume_ratio": round(float(metadata.get("breakout_volume_ratio", 0.0)), 4),
        "impulse_volume": round(float(metadata["impulse_volume"]), 2),
        "pullback_volume": round(float(metadata["pullback_volume"]), 2),
        "pullback_volume_ratio": round(float(metadata["pullback_volume_ratio"]), 4),
        "platform_turnover_pct": round(float(metadata.get("platform_turnover_pct", 0.0)), 2),
        "platform_amplitude_pct": round(float(metadata.get("platform_amplitude_pct", 0.0)), 2),
        "platform_gain_pct": round(float(metadata.get("platform_gain_pct", 0.0)), 2),
        "similarity_score": round(float(metadata["similarity_score"]), 2),
    }
    return row


def build_annual_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["year", "trades", "successes", "success_rate_pct", "median_max_return_20d_pct"])
    work = trades.copy()
    work["year"] = pd.to_datetime(work["signal_date"]).dt.year
    grouped = work.groupby("year", dropna=False)
    return grouped.agg(
        trades=("code", "count"),
        successes=("success_20d_30pct", "sum"),
        success_rate_pct=("success_20d_30pct", lambda series: round(float(series.mean()) * 100, 2)),
        median_max_return_20d_pct=("max_return_20d_pct", "median"),
    ).reset_index()


def build_similarity_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["similarity_bucket", "trades", "successes", "success_rate_pct"])
    work = trades.copy()
    work["similarity_bucket"] = pd.cut(
        work["similarity_score"],
        bins=[-0.01, 40, 60, 70, 80, 100],
        labels=["<40", "40-60", "60-70", "70-80", "80+"],
    )
    grouped = work.groupby("similarity_bucket", dropna=False, observed=False)
    return grouped.agg(
        trades=("code", "count"),
        successes=("success_20d_30pct", "sum"),
        success_rate_pct=("success_20d_30pct", lambda series: round(float(series.mean()) * 100, 2) if len(series) else 0.0),
        median_max_return_20d_pct=("max_return_20d_pct", "median"),
    ).reset_index()


def build_market_regime_summary(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["market_regime", "trades", "successes", "success_rate_pct", "median_max_return_20d_pct"]
    if trades.empty or "market_regime" not in trades.columns:
        return pd.DataFrame(columns=columns)
    grouped = trades.groupby("market_regime", dropna=False)
    return grouped.agg(
        trades=("code", "count"),
        successes=("success_20d_30pct", "sum"),
        success_rate_pct=("success_20d_30pct", lambda series: round(float(series.mean()) * 100, 2) if len(series) else 0.0),
        median_max_return_20d_pct=("max_return_20d_pct", "median"),
    ).reset_index()


def build_summary_markdown(trades: pd.DataFrame, failures: pd.DataFrame, args: argparse.Namespace, stock_count: int) -> str:
    lines = [
        "# Jianghua acceleration retest success backtest",
        "",
        "## Scope",
        f"- Universe: current listed non-ST A-shares from AKShare, stocks={stock_count}",
        f"- Data: local TongDaXin daily bars, {args.history_start_date} to {args.signal_end_date}",
        f"- Signal window: {args.signal_start_date} to {args.signal_end_date}",
        f"- Entry: next trading day open after signal day",
        f"- Success: high reaches entry price * {1 + args.target_return:.2f} within {args.success_bars} trading days",
        f"- Pattern filter: flagpole >= {args.min_flagpole_pct * 100:.0f}%, peak drawdown >= {args.min_peak_drawdown_pct * 100:.0f}%, pullback volume <= {args.max_pullback_volume_ratio * 100:.0f}% of impulse volume, retest within {args.max_retest_bars} bars and within {args.max_days_since_peak} bars after peak",
        f"- Platform filter: previous {args.min_base_bars} trading days, turnover >= {args.min_platform_turnover_pct:.0f}%, amplitude {args.min_platform_amplitude_pct:.0f}%-{args.max_platform_amplitude_pct:.0f}%, gain <= {args.max_platform_gain_pct:.0f}%",
        f"- Market filter: {'disabled' if args.disable_market_filter else f'broad bull if above MA20 >= {args.min_market_above_ma20_rate:.0%}, above MA60 >= {args.min_market_above_ma60_rate:.0%}, advance >= {args.min_market_advance_rate:.0%}; structural bull allowed={args.allow_structural_bull}, ret120_p95 >= {args.min_structural_ret120_p95:.0%}, p95-median spread >= {args.min_structural_return_spread:.0%}'}",
        f"- Reversal diagnostics: first {args.reversal_bars} trading days after entry",
        "",
    ]
    if trades.empty:
        lines.extend(
            [
                "## Results",
                "- Evaluated trades: 0",
                f"- Data failures: {len(failures)}",
            ]
        )
        return "\n".join(lines) + "\n"

    total = len(trades)
    wins = int(trades["success_20d_30pct"].sum())
    success_rate = wins / total if total else 0.0
    winners = trades[trades["success_20d_30pct"]].copy()

    lines.extend(
        [
            "## Results",
            f"- Evaluated non-overlapping trades: {total}",
            f"- Successes: {wins}",
            f"- Success rate: {success_rate * 100:.2f}%",
            f"- Median max return in 20 trading days: {trades['max_return_20d_pct'].median():.2f}%",
            f"- Mean max return in 20 trading days: {trades['max_return_20d_pct'].mean():.2f}%",
            f"- Median platform turnover: {trades['platform_turnover_pct'].median():.2f}%",
            f"- Median platform amplitude: {trades['platform_amplitude_pct'].median():.2f}%",
            f"- Median platform gain: {trades['platform_gain_pct'].median():.2f}%",
            *(
                [
                    f"- Median market above MA20 rate: {trades['market_above_ma20_rate'].median() * 100:.2f}%",
                    f"- Median market above MA60 rate: {trades['market_above_ma60_rate'].median() * 100:.2f}%",
                    f"- Median market ret120 p95: {trades['market_ret120_p95'].median() * 100:.2f}%",
                    f"- Median market ret120 p95-median spread: {trades['market_ret120_spread_p95_median'].median() * 100:.2f}%",
                ]
                if "market_above_ma20_rate" in trades.columns and "market_ret120_p95" in trades.columns
                else []
            ),
            f"- Data failures: {len(failures)}",
            "",
            "## Annual Summary",
            build_annual_summary(trades).to_markdown(index=False),
            "",
            "## Similarity Buckets",
            build_similarity_summary(trades).to_markdown(index=False),
            "",
            "## Market Regimes",
            build_market_regime_summary(trades).to_markdown(index=False),
            "",
            "## Reversal Diagnostics",
        ]
    )
    if winners.empty:
        lines.append("- No successful trades, so reversal diagnostics are unavailable.")
    else:
        lines.extend(
            [
                f"- Winners median bars to +30%: {winners['bars_to_30pct'].dropna().median():.1f}",
                f"- Winners median 60-day peak bar: {winners['peak_bar_60d'].median():.1f}",
                f"- Winners median 60-day max return: {winners['max_return_60d_pct'].median():.2f}%",
                f"- Winners median 10-day drawdown after 60-day peak: {winners['after_peak_10d_low_drawdown_pct'].median():.2f}%",
                f"- Winners with 10% drawdown within 10 bars after peak: {_nonempty_rate(winners, 'first_10pct_drawdown_after_peak_date') * 100:.2f}%",
                f"- Winners with close below MA5 within 20 bars after peak: {_nonempty_rate(winners, 'first_close_below_ma5_after_peak_date') * 100:.2f}%",
                f"- Winners with close below MA10 within 20 bars after peak: {_nonempty_rate(winners, 'first_close_below_ma10_after_peak_date') * 100:.2f}%",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Local TongDaXin `.day` files are treated as raw daily bars; no forward-adjustment is applied.",
            "- The universe is current listed non-ST A-shares, so delisted names and stocks that are currently ST are not included.",
            "- This is a research backtest, not an execution recommendation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_output_dir(args: argparse.Namespace) -> str:
    return f"reports/jianghua_success_backtest_{args.signal_start_date}_{args.signal_end_date}"


def _fmt_date(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _future_low_drawdown(after_peak: pd.DataFrame, peak_high: float, bars: int) -> float | None:
    window = after_peak.head(bars)
    if window.empty:
        return None
    return round((float(window["low"].min()) / peak_high - 1) * 100, 2)


def _first_peak_drawdown_date(after_peak: pd.DataFrame, peak_high: float, drawdown: float) -> str:
    if after_peak.empty:
        return ""
    hit = after_peak[after_peak["low"] <= peak_high * (1 - drawdown)]
    return _fmt_date(hit.iloc[0]["date"]) if not hit.empty else ""


def _first_close_below_ma_date(ma_after_peak: pd.DataFrame, ma_column: str) -> str:
    if ma_after_peak.empty:
        return ""
    hit = ma_after_peak[ma_after_peak["close"] < ma_after_peak[ma_column]]
    return _fmt_date(hit.iloc[0]["date"]) if not hit.empty else ""


def _nonempty_rate(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    return float(df[column].astype(str).str.len().gt(0).mean())


def _attach_optional_turnover_column(df: pd.DataFrame, original_bars: pd.DataFrame) -> pd.DataFrame:
    if "turnover_rate" not in original_bars.columns:
        return df
    turnover = original_bars[["date", "turnover_rate"]].copy()
    turnover["date"] = pd.to_datetime(turnover["date"])
    turnover["turnover_rate"] = pd.to_numeric(turnover["turnover_rate"], errors="coerce")
    return df.merge(turnover, on="date", how="left")


def _platform_mask(
    platform_high: np.ndarray,
    platform_low: np.ndarray,
    platform_start_close: np.ndarray,
    platform_end_close: np.ndarray,
    platform_turnover: np.ndarray,
    min_platform_turnover_pct: float,
    min_platform_amplitude_pct: float,
    max_platform_amplitude_pct: float,
    max_platform_gain_pct: float,
) -> np.ndarray:
    platform_amplitude_pct = (platform_high / platform_low - 1) * 100
    platform_gain_pct = (platform_end_close / platform_start_close - 1) * 100
    return (
        np.isfinite(platform_amplitude_pct)
        & np.isfinite(platform_gain_pct)
        & np.isfinite(platform_turnover)
        & (platform_low > 0)
        & (platform_start_close > 0)
        & (platform_turnover >= min_platform_turnover_pct)
        & (platform_amplitude_pct >= min_platform_amplitude_pct)
        & (platform_amplitude_pct <= max_platform_amplitude_pct)
        & (platform_gain_pct <= max_platform_gain_pct)
    )


def _jianghua_similarity_score(
    flagpole_pct: float,
    peak_drawdown_pct: float,
    close_above_support_pct: float,
    days_since_breakout: int,
    pullback_volume_ratio: float,
) -> float:
    targets = {
        "flagpole_pct": 0.27,
        "peak_drawdown_pct": 0.19,
        "close_above_support_pct": 0.06,
        "days_since_breakout": 7.0,
        "pullback_volume_ratio": 0.75,
    }
    tolerances = {
        "flagpole_pct": 0.14,
        "peak_drawdown_pct": 0.10,
        "close_above_support_pct": 0.04,
        "days_since_breakout": 5.0,
        "pullback_volume_ratio": 0.35,
    }
    values = {
        "flagpole_pct": flagpole_pct,
        "peak_drawdown_pct": peak_drawdown_pct,
        "close_above_support_pct": close_above_support_pct,
        "days_since_breakout": float(days_since_breakout),
        "pullback_volume_ratio": pullback_volume_ratio,
    }
    scores = [max(0.0, 1 - abs(values[key] - targets[key]) / tolerances[key]) for key in targets]
    return float(sum(scores) / len(scores) * 100)


def _steady_climb_similarity_score(
    climb_pct: float,
    peak_drawdown_pct: float,
    close_above_support_pct: float,
    days_since_breakout: int,
    pullback_volume_ratio: float,
) -> float:
    targets = {
        "climb_pct": 0.13,
        "peak_drawdown_pct": 0.06,
        "close_above_support_pct": 0.09,
        "days_since_breakout": 8.0,
        "pullback_volume_ratio": 0.90,
    }
    tolerances = {
        "climb_pct": 0.08,
        "peak_drawdown_pct": 0.06,
        "close_above_support_pct": 0.09,
        "days_since_breakout": 7.0,
        "pullback_volume_ratio": 0.45,
    }
    values = {
        "climb_pct": climb_pct,
        "peak_drawdown_pct": peak_drawdown_pct,
        "close_above_support_pct": close_above_support_pct,
        "days_since_breakout": float(days_since_breakout),
        "pullback_volume_ratio": pullback_volume_ratio,
    }
    scores = [max(0.0, 1 - abs(values[key] - targets[key]) / tolerances[key]) for key in targets]
    return float(sum(scores) / len(scores) * 100)


if __name__ == "__main__":
    main()
