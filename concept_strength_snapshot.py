from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from daily_sector_report import Board, fetch_boards, fetch_top_fund_flow_boards, fetch_top_volume_boards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect current A-share concept strength snapshots for later model filters.")
    parser.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", default="data/cache/concept_strength")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--data-source", choices=["em", "ths", "both"], default="both")
    parser.add_argument("--fund-flow-limit", type=int, default=80)
    parser.add_argument("--volume-limit", type=int, default=80)
    parser.add_argument("--append-history", action="store_true")
    return parser


def main() -> None:
    paths = collect_concept_strength_snapshot(build_parser().parse_args())
    for path in paths:
        print(path)


def collect_concept_strength_snapshot(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    board_sources = ["em", "ths"] if args.data_source == "both" else [args.data_source]
    for source in board_sources:
        try:
            boards = fetch_boards("concept", source)[: args.limit]
            rows.extend(board_to_row(args.date, "concept_pct_chg", source, rank, board) for rank, board in enumerate(boards, 1))
        except Exception as exc:
            errors.append({"source": source, "stage": "concept_pct_chg", "reason": str(exc)[:500]})

    try:
        fund_flow_boards = fetch_top_fund_flow_boards(args.fund_flow_limit)
        rows.extend(
            board_to_row(args.date, "concept_main_fund_flow", "akshare_fund_flow", rank, board)
            for rank, board in enumerate(fund_flow_boards, 1)
        )
    except Exception as exc:
        errors.append({"source": "akshare_fund_flow", "stage": "concept_main_fund_flow", "reason": str(exc)[:500]})

    try:
        volume_boards = fetch_top_volume_boards(args.volume_limit)
        rows.extend(
            board_to_row(args.date, "concept_volume", "akshare_sector_spot", rank, board)
            for rank, board in enumerate(volume_boards, 1)
        )
    except Exception as exc:
        errors.append({"source": "akshare_sector_spot", "stage": "concept_volume", "reason": str(exc)[:500]})

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        raise RuntimeError(f"concept strength snapshot returned no rows; errors={errors}")
    snapshot_path = output_dir / f"concept_strength_{args.date}.csv"
    snapshot.to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    paths = [snapshot_path]
    if errors:
        errors_path = output_dir / f"concept_strength_errors_{args.date}.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")
        paths.append(errors_path)
    if args.append_history:
        history_path = output_dir / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path, dtype={"date": str})
            history = history[history["date"].astype(str) != str(args.date)]
            snapshot = pd.concat([history, snapshot], ignore_index=True)
        snapshot.to_csv(history_path, index=False, encoding="utf-8-sig")
        paths.append(history_path)
    return paths


def board_to_row(date: str, ranking_type: str, provider: str, rank: int, board: Board) -> dict[str, object]:
    raw = board.raw or {}
    return {
        "date": str(date),
        "ranking_type": ranking_type,
        "provider": provider,
        "rank": int(rank),
        "board_name": board.name,
        "board_source": board.source,
        "pct_chg": round(float(board.pct_chg), 4) if pd.notna(board.pct_chg) else None,
        "metric_label": board.metric_label or "",
        "metric_value": round(float(board.metric_value), 4) if board.metric_value is not None and pd.notna(board.metric_value) else None,
        "board_code": first_raw_value(raw, ["f12", "板块代码", "板块代码"]),
        "leader_name": first_raw_value(raw, ["f128", "领涨股", "领涨股"]),
        "leader_code": first_raw_value(raw, ["f140", "领涨股代码", "领涨股代码"]),
        "raw_provider": str(raw.get("_provider", raw.get("_ranking_provider", ""))),
    }


def first_raw_value(raw: dict, keys: list[str]) -> object:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return value
    return ""


if __name__ == "__main__":
    main()
