from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import time

import akshare as ak
import pandas as pd

from daily_sector_report import Board, fetch_board_cons
from kline_model_research import find_signals_for_model
from scan_kline_candidates import fetch_weekly_bars


AI_KEYWORDS = [
    "AI",
    "AIGC",
    "ChatGPT",
    "CPO",
    "\u4eba\u5de5\u667a\u80fd",
    "\u5927\u6a21\u578b",
    "\u7b97\u529b",
    "\u8bed\u6599",
    "\u591a\u6a21\u6001",
    "\u667a\u80fd\u4f53",
    "\u6570\u636e\u4e2d\u5fc3",
    "\u5149\u6a21\u5757",
    "\u82f1\u4f1f\u8fbe",
    "\u4e91\u8ba1\u7b97",
]


@dataclass(frozen=True)
class StockItem:
    code: str
    name: str
    board_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find less-crowded AI concept boards with recent K-line model signals.")
    parser.add_argument("--start-date", default="20250603")
    parser.add_argument("--end-date", default="20260603")
    parser.add_argument("--signal-start-date", default="20260503")
    parser.add_argument("--max-one-year-return", type=float, default=1.0)
    parser.add_argument("--max-amount-percentile", type=float, default=0.70)
    parser.add_argument("--output-dir", default="reports/ai_undertraded_signal_scan_20260603")
    parser.add_argument("--data-source", choices=["tx", "tdx", "auto"], default="auto")
    parser.add_argument("--tdx-vipdoc", default=r"C:\new_tdx\vipdoc")
    parser.add_argument("--max-workers", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = run(args)
    for path in output_paths:
        print(path)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boards = find_ai_boards(args.start_date, args.end_date)
    if boards.empty:
        raise RuntimeError("no AI concept boards found")

    boards = boards.sort_values(["amount_percentile", "one_year_return"], ascending=[True, True]).reset_index(drop=True)
    target_boards = boards[
        (boards["one_year_return"] <= args.max_one_year_return)
        & (boards["amount_percentile"] <= args.max_amount_percentile)
    ].reset_index(drop=True)

    items = collect_board_members(target_boards)
    signals, failures = scan_recent_signals(items, args)
    board_summary = summarize_boards(target_boards, signals)

    board_path = output_dir / "ai_boards.csv"
    target_path = output_dir / "target_boards.csv"
    signal_path = output_dir / "recent_signals.csv"
    summary_path = output_dir / "board_summary.csv"
    failures_path = output_dir / "failures.csv"

    boards.to_csv(board_path, index=False, encoding="utf-8-sig")
    target_boards.to_csv(target_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(signals).to_csv(signal_path, index=False, encoding="utf-8-sig")
    board_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(failures_path, index=False, encoding="utf-8-sig")
    return [board_path, target_path, summary_path, signal_path, failures_path]


def find_ai_boards(start_date: str, end_date: str) -> pd.DataFrame:
    names = ak.stock_board_concept_name_ths()
    mask = pd.Series(False, index=names.index)
    for keyword in AI_KEYWORDS:
        mask |= names["name"].astype(str).str.contains(keyword, case=False, regex=False, na=False)

    rows: list[dict[str, object]] = []
    for row in names[mask].itertuples(index=False):
        name = str(row.name)
        code = str(row.code)
        try:
            history = normalize_board_history(
                ak.stock_board_concept_index_ths(symbol=name, start_date=start_date, end_date=end_date)
            )
            if len(history) < 40 or float(history["close"].iloc[0]) <= 0:
                continue
            one_year_return = float(history["close"].iloc[-1]) / float(history["close"].iloc[0]) - 1
            month_return = float(history["close"].iloc[-1]) / float(history["close"].iloc[-min(20, len(history))]) - 1
            avg_amount_20 = float(history["amount"].tail(min(20, len(history))).mean()) if "amount" in history else 0.0
            rows.append(
                {
                    "board_name": name,
                    "board_code": code,
                    "one_year_return": one_year_return,
                    "month_return": month_return,
                    "avg_amount_20": avg_amount_20,
                    "latest_date": pd.Timestamp(history["date"].iloc[-1]).strftime("%Y-%m-%d"),
                    "latest_close": float(history["close"].iloc[-1]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "board_name": name,
                    "board_code": code,
                    "one_year_return": float("nan"),
                    "month_return": float("nan"),
                    "avg_amount_20": float("nan"),
                    "latest_date": "",
                    "latest_close": float("nan"),
                    "history_error": str(exc)[:220],
                }
            )
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["amount_percentile"] = df["avg_amount_20"].rank(pct=True, ascending=True)
    return df


def normalize_board_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["date", "close", "amount"])
    df = history.copy()
    columns = list(df.columns)
    mapped = pd.DataFrame(
        {
            "date": pd.to_datetime(df[columns[0]]),
            "close": pd.to_numeric(df[columns[4]], errors="coerce"),
            "amount": pd.to_numeric(df[columns[6]], errors="coerce") if len(columns) > 6 else 0.0,
        }
    )
    return mapped.dropna(subset=["date", "close"]).reset_index(drop=True)


def collect_board_members(boards: pd.DataFrame) -> list[StockItem]:
    items: dict[tuple[str, str], StockItem] = {}
    for row in boards.itertuples(index=False):
        board = Board(
            row.board_name,
            "concept",
            0.0,
            {"_provider": "ths", "\u677f\u5757\u4ee3\u7801": str(row.board_code)},
        )
        members = fetch_board_cons(board)
        if members.empty:
            continue
        code_col = "\u4ee3\u7801" if "\u4ee3\u7801" in members else members.columns[0]
        name_col = "\u540d\u79f0" if "\u540d\u79f0" in members else members.columns[1]
        for _, member in members.iterrows():
            code_text = str(member.get(code_col, ""))
            match = re.search(r"\b\d{6}\b", code_text)
            if not match:
                match = re.search(r"\b\d{6}\b", " ".join(member.astype(str).tolist()))
            if not match:
                continue
            code = match.group(0)
            name = str(member.get(name_col, code))
            if code and not name.upper().startswith("ST") and "ST" not in name.upper():
                items[(code, row.board_name)] = StockItem(code, name, row.board_name)
        time.sleep(0.05)
    return list(items.values())


def scan_recent_signals(items: list[StockItem], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in items:
        entry = grouped.setdefault(item.code, {"name": item.name, "boards": []})
        entry["boards"].append(item.board_name)

    signals: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    scan_args = argparse.Namespace(
        start_date="20240101",
        end_date=args.end_date,
        data_source=args.data_source,
        tdx_vipdoc=args.tdx_vipdoc,
    )
    signal_start = pd.Timestamp(args.signal_start_date)
    signal_end = pd.Timestamp(args.end_date)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(scan_one_stock, code, str(info["name"]), list(info["boards"]), scan_args, signal_start, signal_end): code
            for code, info in grouped.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                rows = future.result()
                signals.extend(rows)
            except Exception as exc:
                failures.append({"code": code, "reason": str(exc)[:220]})
            if done % 100 == 0:
                print(f"scan_done {done}/{len(futures)} signals={len(signals)} failures={len(failures)}", flush=True)
    return signals, failures


def scan_one_stock(
    code: str,
    name: str,
    boards: list[str],
    args: argparse.Namespace,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> list[dict[str, object]]:
    weekly = fetch_weekly_bars(code, args)
    if len(weekly) < 60:
        return []
    signals = find_signals_for_model(
        "trend_pullback_restart",
        weekly,
        {"ma_fast": 10, "ma_slow": 30, "pullback_tolerance": 0.02},
    )
    rows: list[dict[str, object]] = []
    for signal in signals:
        signal_date = pd.Timestamp(signal.signal_date)
        if signal_start <= signal_date <= signal_end and strength_ok_at_index(weekly, signal.signal_index):
            latest = weekly.iloc[signal.signal_index]
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "signal_close": float(latest["close"]),
                    "initial_stop": round(float(signal.initial_stop), 3),
                    "trigger_price": round(float(signal.trigger_price), 3),
                    "boards": "；".join(sorted(set(boards))),
                    "data_source": weekly.attrs.get("data_source", "unknown"),
                }
            )
    return rows


def strength_ok_at_index(weekly: pd.DataFrame, index: int) -> bool:
    close = weekly["close"].astype(float)
    if index < 59 or close.iloc[index - 25] <= 0:
        return False
    ma20 = close.rolling(20).mean().iloc[index]
    ma40 = close.rolling(40).mean().iloc[index]
    ma60 = close.rolling(60).mean().iloc[index]
    return bool(
        close.iloc[index] > ma20 > ma40 > ma60
        and close.iloc[index] >= 0.95 * close.iloc[max(0, index - 51) : index + 1].max()
        and close.iloc[index] / close.iloc[index - 25] - 1 > 0.35
    )


def summarize_boards(boards: pd.DataFrame, signals: list[dict[str, object]]) -> pd.DataFrame:
    signal_df = pd.DataFrame(signals)
    rows: list[dict[str, object]] = []
    for board in boards.itertuples(index=False):
        if signal_df.empty:
            board_signals = pd.DataFrame()
        else:
            board_signals = signal_df[signal_df["boards"].astype(str).str.contains(str(board.board_name), regex=False)]
        rows.append(
            {
                "board_name": board.board_name,
                "board_code": board.board_code,
                "one_year_return": board.one_year_return,
                "month_return": board.month_return,
                "avg_amount_20": board.avg_amount_20,
                "amount_percentile": board.amount_percentile,
                "recent_signal_count": len(board_signals),
                "recent_signal_stocks": "；".join(
                    board_signals.sort_values("signal_date", ascending=False)["code"].astype(str).drop_duplicates().head(12).tolist()
                )
                if not board_signals.empty
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["recent_signal_count", "amount_percentile", "one_year_return"],
        ascending=[False, True, True],
        ignore_index=True,
    )


if __name__ == "__main__":
    main()
