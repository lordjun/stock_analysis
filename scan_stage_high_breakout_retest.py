from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
import re

import akshare as ak
import pandas as pd
import requests

from daily_sector_report import Board, fetch_board_cons, fetch_boards, fetch_top_fund_flow_boards, ths_headers
from darvas_weekly_backtest import fetch_daily_history_akshare_tx, normalize_daily_columns
from kline_model_research import (
    find_jianghua_acceleration_retests,
    find_stage_high_breakout_retests,
    normalize_price_bars,
)
from mainline_pool import (
    DynamicMainlinePoolProvider,
    MainlinePoolConfig,
    core_mainline_concept_matches,
    load_or_fetch_stock_concepts,
    parse_theme_keywords as parse_mainline_theme_keywords,
)
from tdx_data import read_tdx_daily_bars
from backtest_jianghua_success import (
    _market_context_lookup,
    attach_turnover_rate_from_float_share,
    classify_market_regime,
    find_jianghua_acceleration_retests_fast,
    load_float_share_by_code,
    load_or_build_market_context,
    load_structural_code_pool,
    market_context_passes_filter,
    signal_quality_passes_filter,
)


CANDIDATE_COLUMNS = [
    "code",
    "name",
    "model",
    "pattern_subtype",
    "signal_date",
    "latest_close",
    "breakout_date",
    "breakout_close",
    "breakout_high",
    "prior_high",
    "structure_high",
    "peak_high",
    "pullback_low",
    "days_since_breakout",
    "days_since_peak",
    "distance_to_prior_high_pct",
    "flagpole_pct",
    "peak_drawdown_pct",
    "drawdown_from_breakout_high_pct",
    "breakout_volume_ratio",
    "pullback_volume_ratio",
    "similarity_score",
    "signal_initial_stop_price",
    "trigger_price",
    "ma_fast",
    "ma_slow",
    "market_advance_rate",
    "market_regime",
    "in_structural_code_pool",
    "market_above_ma20_rate",
    "market_above_ma60_rate",
    "market_new_high_60_rate",
    "market_ret120_median",
    "market_ret120_p95",
    "market_ret120_spread_p95_median",
    "data_source",
    "concept_boards",
    "matched_mainline_boards",
    "note",
]
FAILURE_COLUMNS = ["code", "name", "reason"]
STRUCTURAL_POOL_COLUMNS = ["code", "name", "board_name", "board_rank", "pool_source"]


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
    parser.add_argument("--min-base-bars", type=int, default=120)
    parser.add_argument("--max-retest-bars", type=int, default=5)
    parser.add_argument("--breakout-buffer", type=float, default=0.005)
    parser.add_argument("--support-tolerance", type=float, default=0.015)
    parser.add_argument("--min-pullback-pct", type=float, default=0.03)
    parser.add_argument("--max-extension-pct", type=float, default=0.08)
    parser.add_argument("--ma-fast", type=int, default=20)
    parser.add_argument("--ma-slow", type=int, default=60)
    parser.add_argument("--structure-lookback-bars", type=int, default=120)
    parser.add_argument("--max-peak-bars", type=int, default=5)
    parser.add_argument("--min-flagpole-pct", type=float, default=0.22)
    parser.add_argument("--max-flagpole-pct", type=float, default=0.45)
    parser.add_argument("--min-peak-drawdown-pct", type=float, default=0.16)
    parser.add_argument("--max-peak-drawdown-pct", type=float, default=0.28)
    parser.add_argument("--max-days-since-peak", type=int, default=2)
    parser.add_argument("--min-close-above-support-pct", type=float, default=0.02)
    parser.add_argument("--max-close-above-support-pct", type=float, default=0.09)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=1.5)
    parser.add_argument("--max-pullback-volume-ratio", type=float, default=0.70)
    parser.add_argument("--min-platform-turnover-pct", type=float, default=100.0)
    parser.add_argument("--min-platform-amplitude-pct", type=float, default=35.0)
    parser.add_argument("--max-platform-amplitude-pct", type=float, default=100.0)
    parser.add_argument("--max-platform-gain-pct", type=float, default=20.0)
    parser.add_argument("--min-signal-platform-gain-pct", type=float, default=12.0)
    parser.add_argument("--max-signal-pullback-volume-ratio", type=float, default=0.80)
    parser.add_argument("--semiconductor-exception-min-breakout-volume-ratio", type=float, default=3.0)
    parser.add_argument("--semiconductor-exception-max-pullback-volume-ratio", type=float, default=0.60)
    parser.add_argument("--disable-market-filter", action="store_true")
    parser.add_argument("--market-context-cache", default=None)
    parser.add_argument("--min-market-above-ma20-rate", type=float, default=0.45)
    parser.add_argument("--min-market-above-ma60-rate", type=float, default=0.35)
    parser.add_argument("--min-market-advance-rate", type=float, default=0.0)
    parser.add_argument("--allow-structural-bull", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-structural-ret120-p95", type=float, default=0.30)
    parser.add_argument("--min-structural-return-spread", type=float, default=0.35)
    parser.add_argument("--structural-code-pool-file", default=None)
    parser.add_argument("--mainline-mode", choices=["dynamic", "static", "none"], default="dynamic")
    parser.add_argument("--dynamic-mainline-cache-dir", default="data/cache/mainline_pools")
    parser.add_argument("--structural-code-pool-cache", default=None)
    parser.add_argument("--structural-pool-mode", choices=["long_term", "current"], default="long_term")
    parser.add_argument("--structural-concept-top-n", type=int, default=10)
    parser.add_argument("--structural-concept-lookback-days", type=int, default=540)
    parser.add_argument("--min-structural-concept-ret120", type=float, default=0.10)
    parser.add_argument("--min-structural-concept-ret250", type=float, default=0.20)
    parser.add_argument("--min-structural-concept-ret60", type=float, default=0.05)
    parser.add_argument("--min-structural-concept-breadth120", type=float, default=0.55)
    parser.add_argument("--mainline-fallback-top-rank", type=int, default=5)
    parser.add_argument("--mainline-fallback-min-matches-outside-top-rank", type=int, default=2)
    parser.add_argument(
        "--structural-theme-keywords",
        default="AI,算力,CPO,光模块,机器人,半导体,芯片,6G,数据中心,东数西算,云计算,软件,信创,低空,无人机,卫星,军工信息化,新型工业化,存储,PCB,消费电子",
    )
    parser.add_argument("--float-share-cache", default=None)
    parser.add_argument("--float-share-provider", choices=["auto", "tdx", "tushare", "eastmoney"], default="auto")
    parser.add_argument("--tdx-base-dbf", default=r"C:\new_tdx\T0002\hq_cache\base.dbf")
    parser.add_argument("--tushare-token-env", default="TUSHARE_TOKEN")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run(args)
    for path in output_paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)

    universe = ak.stock_info_a_code_name()
    universe = universe[~universe["name"].astype(str).str.upper().str.contains("ST", na=False)].reset_index(drop=True)
    if args.limit:
        universe = universe.head(args.limit)
    items = [(str(row.code).zfill(6), str(row.name)) for row in universe.itertuples(index=False)]
    market_items = items
    if args.model == "jianghua_acceleration_retest":
        args.signal_end_date = args.end_date
        args.history_start_date = args.start_date
        args.float_share_by_code = load_float_share_by_code(args)
        args.stock_concepts_by_code = {}
        args.structural_board_pool = set()
        args.structural_board_rank = {}
        args.structural_code_pool = load_or_build_structural_code_pool(args)
        args.market_context_by_date = {}
        if not args.disable_market_filter:
            args.market_context_by_date = _market_context_lookup(load_or_build_market_context(market_items, args))
        if should_prefilter_with_structural_pool(args) and args.structural_code_pool:
            items = [(code, name) for code, name in items if code in args.structural_code_pool]
            print(
                f"structural_universe_filter kept={len(items)} original={len(market_items)}",
                flush=True,
            )

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

    candidates = (
        pd.DataFrame(rows).sort_values(["signal_date", "code"], ascending=[False, True])
        if rows
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    output_paths = write_scan_outputs(candidates, failures, output_dir, len(items))
    print(f"DONE total={len(items)} candidates={len(rows)} failures={len(failures)}", flush=True)
    return output_paths


def write_scan_outputs(
    candidates: pd.DataFrame,
    failures: list[dict[str, str]],
    output_dir: Path,
    total_count: int,
) -> list[Path]:
    if candidates.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.csv"
    report_path = output_dir / "report.md"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_candidate_report_markdown(candidates, total_count), encoding="utf-8")

    paths = [candidates_path, report_path]
    if failures:
        failures_path = output_dir / "failures.csv"
        pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False, encoding="utf-8-sig")
        paths.append(failures_path)
    return paths


def build_candidate_report_markdown(candidates: pd.DataFrame, total_count: int) -> str:
    lines = [
        "# K-line Model Candidate Report",
        "",
        f"- Scanned stocks: {total_count}",
        f"- Candidates: {len(candidates)}",
        "",
    ]
    for row in candidates.itertuples(index=False):
        data = row._asdict()
        lines.extend(
            [
                f"## {data.get('code', '')} {data.get('name', '')}",
                "",
                f"- Pattern subtype: {data.get('pattern_subtype', '')}",
                f"- Signal date: {data.get('signal_date', '')}",
                f"- Breakout date: {data.get('breakout_date', '')}",
                f"- Latest close: {data.get('latest_close', '')}",
                f"- Structure high: {data.get('structure_high', '')}",
                f"- Initial stop: {data.get('signal_initial_stop_price', '')}",
                f"- Breakout volume ratio: {data.get('breakout_volume_ratio', '')}",
                f"- Pullback volume ratio: {data.get('pullback_volume_ratio', '')}",
                f"- Similarity score: {data.get('similarity_score', '')}",
                f"- Concept boards: {data.get('concept_boards', '')}",
                f"- Market regime: {data.get('market_regime', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def load_or_build_structural_code_pool(args: argparse.Namespace) -> set[str]:
    if args.structural_code_pool_file:
        return load_structural_code_pool(args.structural_code_pool_file)
    if args.mainline_mode == "none":
        return set()
    if args.mainline_mode == "dynamic":
        config = MainlinePoolConfig(
            cache_dir=Path(args.dynamic_mainline_cache_dir),
            tdx_vipdoc=args.tdx_vipdoc,
            lookback_days=args.structural_concept_lookback_days,
            top_n=args.structural_concept_top_n,
            min_ret60=args.min_structural_concept_ret60,
            min_ret120=args.min_structural_concept_ret120,
            min_ret250=args.min_structural_concept_ret250,
            min_breadth120=args.min_structural_concept_breadth120,
            theme_keywords=parse_mainline_theme_keywords(args.structural_theme_keywords),
        )
        provider = DynamicMainlinePoolProvider(config)
        codes = provider.codes_for_date(args.end_date)
        args.structural_board_pool = provider.boards_for_date(args.end_date)
        args.structural_board_rank = provider.board_ranks_for_date(args.end_date)
        print(f"dynamic_mainline_ready date={args.end_date} codes={len(codes)}", flush=True)
        return codes
    if not args.allow_structural_bull:
        return set()

    cache_path = Path(
        args.structural_code_pool_cache
        or f"data/cache/concept_strength/structural_code_pool_{args.structural_pool_mode}_{args.end_date}.csv"
    )
    if cache_path.exists():
        return load_structural_code_pool(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.structural_pool_mode == "long_term":
        pool = build_long_term_structural_concept_pool(
            end_date=args.end_date,
            tdx_vipdoc=args.tdx_vipdoc,
            lookback_days=args.structural_concept_lookback_days,
            top_n=args.structural_concept_top_n,
            min_ret120=args.min_structural_concept_ret120,
            min_ret250=args.min_structural_concept_ret250,
            theme_keywords=parse_theme_keywords(args.structural_theme_keywords),
        )
        if pool.empty:
            print("long_term_structural_pool_empty fallback=current", flush=True)
            pool = build_structural_concept_pool(args.structural_concept_top_n)
    else:
        pool = build_structural_concept_pool(args.structural_concept_top_n)
    pool.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"structural_code_pool_cached rows={len(pool)} path={cache_path}", flush=True)
    return load_structural_code_pool(cache_path)


def should_prefilter_with_structural_pool(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "structural_code_pool_file", None))


def build_structural_concept_pool(top_n: int) -> pd.DataFrame:
    boards = select_structural_concept_boards(top_n)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for rank, board in enumerate(boards, start=1):
        try:
            constituents = fetch_board_cons(board)
        except Exception as exc:
            print(f"structural_board_cons_failed board={board.name} reason={str(exc)[:160]}", flush=True)
            continue
        for code, name in extract_constituent_codes(constituents):
            key = (code, board.name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "board_name": board.name,
                    "board_rank": rank,
                    "pool_source": board.raw.get("_ranking_provider", board.raw.get("_provider", "unknown")),
                }
            )
    return pd.DataFrame(rows, columns=STRUCTURAL_POOL_COLUMNS)


def build_long_term_structural_concept_pool(
    end_date: str,
    tdx_vipdoc: str,
    lookback_days: int,
    top_n: int,
    min_ret120: float,
    min_ret250: float,
    theme_keywords: list[str],
) -> pd.DataFrame:
    histories = fetch_concept_histories(end_date, lookback_days)
    boards = select_long_term_concept_boards(histories, top_n, min_ret120, min_ret250)
    if not boards:
        return build_constituent_return_structural_concept_pool(
            end_date=end_date,
            tdx_vipdoc=tdx_vipdoc,
            lookback_days=lookback_days,
            top_n=top_n,
            min_ret120=min_ret120,
            min_ret250=min_ret250,
            theme_keywords=theme_keywords,
        )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for rank, board in enumerate(boards, start=1):
        try:
            constituents = ak.stock_board_concept_cons_em(symbol=board.name)
        except Exception as exc:
            print(f"long_term_board_cons_failed board={board.name} reason={str(exc)[:160]}", flush=True)
            continue
        for code, name in extract_constituent_codes(constituents):
            key = (code, board.name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "board_name": board.name,
                    "board_rank": rank,
                    "pool_source": "long_term_concept_strength",
                }
            )
    return pd.DataFrame(rows, columns=STRUCTURAL_POOL_COLUMNS)


def build_constituent_return_structural_concept_pool(
    end_date: str,
    tdx_vipdoc: str,
    lookback_days: int,
    top_n: int,
    min_ret120: float,
    min_ret250: float,
    theme_keywords: list[str],
) -> pd.DataFrame:
    boards = load_theme_concept_boards(theme_keywords)
    board_constituents: dict[str, pd.DataFrame] = {}
    for board in boards:
        try:
            board_constituents[board.name] = fetch_ths_concept_constituents(str(board.raw["board_code"]), board.name)
        except Exception as exc:
            print(f"theme_board_cons_failed board={board.name} reason={str(exc)[:160]}", flush=True)

    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    stock_returns: dict[str, dict[str, float]] = {}
    for constituents in board_constituents.values():
        for code, _name in extract_constituent_codes(constituents):
            if code in stock_returns:
                continue
            stock_returns[code] = read_stock_return_features(tdx_vipdoc, code, start_date, end_date)

    strength = build_constituent_return_concept_strength(board_constituents, stock_returns)
    selected = strength[(strength["ret120_median"] >= min_ret120) & (strength["ret250_median"] >= min_ret250)].head(top_n)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for rank, board_name in enumerate(selected["board_name"].astype(str), start=1):
        constituents = board_constituents.get(board_name, pd.DataFrame())
        for code, name in extract_constituent_codes(constituents):
            key = (code, board_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "board_name": board_name,
                    "board_rank": rank,
                    "pool_source": "constituent_return_strength",
                }
            )
    return pd.DataFrame(rows, columns=STRUCTURAL_POOL_COLUMNS)


def fetch_concept_histories(end_date: str, lookback_days: int) -> dict[str, pd.DataFrame]:
    end_ts = pd.Timestamp(end_date)
    start_date = (end_ts - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_raw = end_ts.strftime("%Y%m%d")
    names = load_em_concept_names_for_history()

    histories: dict[str, pd.DataFrame] = {}
    for name in names:
        try:
            hist = ak.stock_board_concept_hist_em(symbol=name, start_date=start_date, end_date=end_raw, period="日k", adjust="")
            histories[name] = normalize_concept_history(hist)
        except Exception as exc:
            print(f"concept_hist_failed board={name} reason={str(exc)[:120]}", flush=True)
    return histories


def load_em_concept_names_for_history() -> list[str]:
    try:
        em_names = ak.stock_board_concept_name_em()
        names = [str(row.get("板块名称", row.get("name", ""))).strip() for _, row in em_names.iterrows()]
        return [name for name in names if name]
    except Exception as exc:
        print(f"concept_name_em_failed reason={str(exc)[:160]}", flush=True)
        return []

def normalize_concept_history(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame(columns=["date", "close"])
    date_col = next((column for column in ["日期", "date"] if column in hist.columns), hist.columns[0])
    close_col = next((column for column in ["收盘", "close"] if column in hist.columns), None)
    if close_col is None:
        numeric_cols = [column for column in hist.columns if pd.api.types.is_numeric_dtype(hist[column])]
        if not numeric_cols:
            return pd.DataFrame(columns=["date", "close"])
        close_col = numeric_cols[-1]
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(hist[date_col]),
            "close": pd.to_numeric(hist[close_col], errors="coerce"),
        }
    ).dropna(subset=["date", "close"])
    return output.sort_values("date").reset_index(drop=True)


def build_long_term_concept_strength(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, hist in histories.items():
        history = normalize_concept_history(hist)
        if len(history) < 121:
            continue
        close = pd.to_numeric(history["close"], errors="coerce").dropna().reset_index(drop=True)
        if len(close) < 121 or close.iloc[-1] <= 0:
            continue
        ret20 = _window_return(close, 20)
        ret60 = _window_return(close, 60)
        ret120 = _window_return(close, 120)
        ret250 = _window_return(close, 250)
        ma20 = float(close.tail(20).mean())
        ma60 = float(close.tail(60).mean()) if len(close) >= 60 else ma20
        trend_score = (1 if close.iloc[-1] > ma20 else 0) + (1 if ma20 > ma60 else 0) + max(0.0, min(ret20, 0.50))
        long_term_score = ret120 * 0.35 + ret250 * 0.45 + ret60 * 0.15 + trend_score * 0.05
        rows.append(
            {
                "board_name": name,
                "ret20": ret20,
                "ret60": ret60,
                "ret120": ret120,
                "ret250": ret250,
                "trend_score": trend_score,
                "long_term_score": long_term_score,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["board_name", "ret20", "ret60", "ret120", "ret250", "trend_score", "long_term_score"])
    return pd.DataFrame(rows).sort_values(["long_term_score", "ret250", "ret120"], ascending=False).reset_index(drop=True)


def select_long_term_concept_boards(
    histories: dict[str, pd.DataFrame],
    top_n: int,
    min_ret120: float,
    min_ret250: float,
) -> list[Board]:
    strength = build_long_term_concept_strength(histories)
    if strength.empty:
        return []
    filtered = strength[(strength["ret120"] >= min_ret120) & (strength["ret250"] >= min_ret250)].head(top_n)
    boards: list[Board] = []
    for _, row in filtered.iterrows():
        raw = row.to_dict()
        raw["_ranking_provider"] = "long_term_concept_strength"
        boards.append(Board(str(row["board_name"]), "concept", float(row["ret20"]) * 100, raw))
    return boards


def parse_theme_keywords(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [item.strip() for item in raw if str(item).strip()]
    return [item.strip() for item in re.split(r"[,，;；\s]+", str(raw)) if item.strip()]


def load_theme_concept_boards(theme_keywords: list[str]) -> list[Board]:
    if not theme_keywords:
        return []
    try:
        names = ak.stock_board_concept_name_ths()
    except Exception as exc:
        print(f"theme_concept_name_failed reason={str(exc)[:160]}", flush=True)
        return []
    boards: list[Board] = []
    for _, row in names.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip()
        if not name or not code:
            continue
        if any(keyword.lower() in name.lower() for keyword in theme_keywords):
            boards.append(Board(name, "concept", 0.0, {"board_code": code, "_provider": "ths_theme"}))
    return boards


def fetch_ths_concept_constituents(board_code: str, board_name: str) -> pd.DataFrame:
    url = f"http://q.10jqka.com.cn/gn/detail/code/{board_code}/"
    response = requests.get(url, headers=ths_headers(), timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    try:
        return pd.read_html(StringIO(response.text))[0]
    except ValueError as exc:
        raise RuntimeError(f"THS concept constituents parse failed: {board_name}") from exc


def read_stock_return_features(tdx_vipdoc: str, code: str, start_date: str, end_date: str) -> dict[str, float]:
    try:
        bars = normalize_price_bars(read_tdx_daily_bars(tdx_vipdoc, code, start_date, end_date))
    except Exception:
        return {"ret120": float("nan"), "ret250": float("nan"), "ret60": float("nan")}
    if bars.empty:
        return {"ret120": float("nan"), "ret250": float("nan"), "ret60": float("nan")}
    close = pd.to_numeric(bars["close"], errors="coerce").dropna().reset_index(drop=True)
    return {
        "ret60": _window_return(close, 60),
        "ret120": _window_return(close, 120),
        "ret250": _window_return(close, 250),
    }


def build_constituent_return_concept_strength(
    board_constituents: dict[str, pd.DataFrame],
    stock_returns: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for board_name, constituents in board_constituents.items():
        codes = [code for code, _name in extract_constituent_codes(constituents)]
        features = [stock_returns.get(code, {}) for code in codes]
        frame = pd.DataFrame(features)
        if frame.empty or "ret120" not in frame or "ret250" not in frame:
            continue
        frame = frame.apply(pd.to_numeric, errors="coerce")
        valid = frame.dropna(subset=["ret120", "ret250"])
        if len(valid) < 3:
            continue
        ret120_median = float(valid["ret120"].median())
        ret250_median = float(valid["ret250"].median())
        ret120_p75 = float(valid["ret120"].quantile(0.75))
        ret250_p75 = float(valid["ret250"].quantile(0.75))
        breadth120 = float((valid["ret120"] > 0).mean())
        score = ret250_median * 0.40 + ret120_median * 0.35 + ret250_p75 * 0.15 + breadth120 * 0.10
        rows.append(
            {
                "board_name": board_name,
                "member_count": len(valid),
                "ret120_median": ret120_median,
                "ret250_median": ret250_median,
                "ret120_p75": ret120_p75,
                "ret250_p75": ret250_p75,
                "breadth120": breadth120,
                "long_term_score": score,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "board_name",
                "member_count",
                "ret120_median",
                "ret250_median",
                "ret120_p75",
                "ret250_p75",
                "breadth120",
                "long_term_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(["long_term_score", "ret250_median", "ret120_median"], ascending=False).reset_index(drop=True)


def _window_return(close: pd.Series, bars: int) -> float:
    if len(close) <= bars or close.iloc[-bars - 1] <= 0:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-bars - 1] - 1)


def select_structural_concept_boards(top_n: int) -> list[Board]:
    selected: list[Board] = []
    seen_names: set[str] = set()
    sources = [
        ("fund_flow", lambda: fetch_top_fund_flow_boards(top_n)),
        ("ths_pct", lambda: fetch_boards("concept", "ths")[:top_n]),
    ]
    for source_name, loader in sources:
        try:
            boards = loader()
        except Exception as exc:
            print(f"structural_boards_failed source={source_name} reason={str(exc)[:160]}", flush=True)
            continue
        for board in boards:
            if board.name in seen_names:
                continue
            seen_names.add(board.name)
            raw = dict(board.raw)
            raw["_ranking_provider"] = raw.get("_ranking_provider", source_name)
            selected.append(Board(board.name, board.source, board.pct_chg, raw, board.metric_value, board.metric_label))
    return selected


def extract_constituent_codes(df: pd.DataFrame) -> list[tuple[str, str]]:
    if df.empty:
        return []
    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        code = ""
        for value in row.tolist():
            match = re.search(r"(?<!\d)(\d{6})(?!\d)", str(value))
            if match:
                code = match.group(1)
                break
        if not code:
            continue
        name = _extract_name_from_row(row, code)
        rows.append((code, name))
    return rows


def _extract_name_from_row(row: pd.Series, code: str) -> str:
    for value in row.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != code and not re.fullmatch(r"[-+]?\d+(\.\d+)?%?", text):
            if not re.search(r"\d{6}", text):
                return text
    return ""


def scan_one(code: str, name: str, args: argparse.Namespace) -> dict[str, object] | None:
    daily = fetch_daily_bars(code, args)
    if args.model == "jianghua_acceleration_retest":
        daily = attach_turnover_rate_from_float_share(daily, code, args.float_share_by_code)
    min_history = max(args.lookback_bars, args.structure_lookback_bars, args.ma_slow) + args.max_retest_bars + 1
    if len(daily) < min_history:
        return None

    if args.model == "jianghua_acceleration_retest":
        signals = find_jianghua_acceleration_retests_fast(
            daily,
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
    market_context = None
    if args.model == "jianghua_acceleration_retest":
        market_context = args.market_context_by_date.get(pd.Timestamp(signal.signal_date).strftime("%Y-%m-%d"))
        market_regime = classify_market_regime(
            market_context,
            args.min_market_above_ma20_rate,
            args.min_market_above_ma60_rate,
            args.min_market_advance_rate,
            args.allow_structural_bull,
            args.min_structural_ret120_p95,
            args.min_structural_return_spread,
        )
        structural_match, stock_concepts, matched_boards = stock_matches_structural_theme(
            args,
            code,
            pd.Timestamp(signal.signal_date),
            getattr(args, "structural_code_pool", set()),
        )
        if not signal_quality_passes_filter(
            signal.metadata,
            args.min_signal_platform_gain_pct,
            args.max_signal_pullback_volume_ratio,
            matched_mainline_boards=matched_boards,
            semiconductor_exception_min_breakout_volume_ratio=args.semiconductor_exception_min_breakout_volume_ratio,
            semiconductor_exception_max_pullback_volume_ratio=args.semiconductor_exception_max_pullback_volume_ratio,
        ):
            return None
        if not args.disable_market_filter and not market_context_passes_filter(
            market_context,
            args.min_market_above_ma20_rate,
            args.min_market_above_ma60_rate,
            args.min_market_advance_rate,
            args.allow_structural_bull,
            args.min_structural_ret120_p95,
            args.min_structural_return_spread,
            code=code,
            structural_code_pool={str(code).zfill(6)} if structural_match else set(),
        ):
            return None
    else:
        market_regime = ""
        structural_match = str(code).zfill(6) in getattr(args, "structural_code_pool", set())
        stock_concepts = []
        matched_boards = []
        if not signal_quality_passes_filter(
            signal.metadata,
            args.min_signal_platform_gain_pct,
            args.max_signal_pullback_volume_ratio,
        ):
            return None

    metadata = signal.metadata
    breakout_index = int(metadata["breakout_index"])
    breakout_date = pd.Timestamp(daily.iloc[breakout_index]["date"])
    close = float(latest["close"])
    support_level = float(metadata.get("structure_high", metadata.get("prior_high")))
    return {
        "code": code,
        "name": name,
        "model": args.model,
        "pattern_subtype": str(metadata.get("pattern_subtype", "")),
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
        "breakout_volume_ratio": round(float(metadata.get("breakout_volume_ratio", 0.0)), 3),
        "pullback_volume_ratio": round(float(metadata.get("pullback_volume_ratio", 0.0)), 3),
        "similarity_score": round(float(metadata.get("similarity_score", 0.0)), 2),
        "signal_initial_stop_price": round(float(signal.initial_stop), 3),
        "trigger_price": round(float(signal.trigger_price), 3),
        "ma_fast": round(float(metadata["ma_fast"]), 3),
        "ma_slow": round(float(metadata["ma_slow"]), 3),
        "market_advance_rate": round(float((market_context or {}).get("advance_rate", 0.0)), 4),
        "market_regime": market_regime,
        "in_structural_code_pool": structural_match,
        "market_above_ma20_rate": round(float((market_context or {}).get("above_ma20_rate", 0.0)), 4),
        "market_above_ma60_rate": round(float((market_context or {}).get("above_ma60_rate", 0.0)), 4),
        "market_new_high_60_rate": round(float((market_context or {}).get("new_high_60_rate", 0.0)), 4),
        "market_ret120_median": round(float((market_context or {}).get("ret120_median", 0.0)), 4),
        "market_ret120_p95": round(float((market_context or {}).get("ret120_p95", 0.0)), 4),
        "market_ret120_spread_p95_median": round(
            float((market_context or {}).get("ret120_spread_p95_median", 0.0)),
            4,
        ),
        "data_source": daily.attrs.get("data_source", "unknown"),
        "concept_boards": "、".join(stock_concepts),
        "matched_mainline_boards": "、".join(matched_boards),
        "note": f"{args.model}_daily; watchlist_only",
    }


def stock_matches_structural_theme(
    args: argparse.Namespace,
    code: str,
    signal_date: pd.Timestamp,
    structural_code_pool: set[str],
) -> tuple[bool, list[str], list[str]]:
    normalized_code = str(code).zfill(6)
    if normalized_code in structural_code_pool:
        return True, [], []

    rank_by_board = getattr(args, "structural_board_rank", {})
    if not rank_by_board:
        mainline_boards = getattr(args, "structural_board_pool", set())
        rank_by_board = {board: 999 for board in mainline_boards}
    if not rank_by_board:
        return False, [], []

    concepts_by_code = getattr(args, "stock_concepts_by_code", {})
    if normalized_code in concepts_by_code:
        concepts = list(concepts_by_code[normalized_code])
    else:
        concepts = load_or_fetch_stock_concepts(Path(getattr(args, "dynamic_mainline_cache_dir", "data/cache/mainline_pools")), normalized_code)
        concepts_by_code[normalized_code] = concepts
        args.stock_concepts_by_code = concepts_by_code

    matched = core_mainline_concept_matches(
        concepts,
        rank_by_board,
        top_rank=int(getattr(args, "mainline_fallback_top_rank", 5)),
        min_matches_outside_top_rank=int(getattr(args, "mainline_fallback_min_matches_outside_top_rank", 2)),
    )
    return bool(matched), concepts, matched


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
