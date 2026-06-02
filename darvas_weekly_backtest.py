from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except ImportError:  # pragma: no cover - handled at CLI runtime
    ak = None

try:
    import tushare as ts
except ImportError:  # pragma: no cover - handled at CLI runtime
    ts = None


@dataclass(frozen=True)
class Signal:
    signal_index: int
    signal_date: pd.Timestamp
    entry_price: float
    box_top: float
    box_bottom: float
    volume_ratio: float


@dataclass(frozen=True)
class Trade:
    code: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    box_top: float
    volume_ratio: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    realized_return: float
    best_high_return: float
    max_high_5w: float


def normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    normalized = df.rename(columns=rename_map).copy()
    if "volume" not in normalized.columns and "amount" in normalized.columns:
        normalized["volume"] = normalized["amount"]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    normalized = normalized[required]
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=required).sort_values("date")
    return normalized.reset_index(drop=True)


def normalize_tushare_daily(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"missing required tushare columns: {', '.join(missing)}")
    normalized = normalized[required]
    normalized["date"] = pd.to_datetime(normalized["date"], format="%Y%m%d")
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=required).sort_values("date")
    return normalized.reset_index(drop=True)


def normalize_tx_weekly_records(records: list[list[object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
    frame = frame.iloc[:, :6]
    frame.columns = ["date", "open", "close", "high", "low", "volume"]
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["open", "close", "high", "low", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    return frame.sort_values("date").reset_index(drop=True)


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_daily_columns(daily)
    weekly = (
        normalized.set_index("date")
        .resample("W-FRI")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return weekly.reset_index(drop=True)


def find_signals(
    weekly: pd.DataFrame,
    box_weeks: int = 20,
    volume_multiplier: float = 1.5,
    ma_fast: int = 20,
    ma_slow: int = 40,
    hold_weeks: int = 5,
) -> list[Signal]:
    df = weekly.copy().reset_index(drop=True)
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()

    signals: list[Signal] = []
    next_allowed_index = 0
    first_index = max(box_weeks, ma_slow)

    for index in range(first_index, len(df)):
        if index < next_allowed_index:
            continue

        prior_box = df.iloc[index - box_weeks : index]
        box_top = float(prior_box["high"].max())
        box_bottom = float(prior_box["low"].min())
        avg_volume = float(prior_box["volume"].mean())
        close = float(df.at[index, "close"])
        volume = float(df.at[index, "volume"])
        ma_fast_value = float(df.at[index, "ma_fast"])
        ma_slow_value = float(df.at[index, "ma_slow"])

        if avg_volume <= 0 or math.isnan(ma_fast_value) or math.isnan(ma_slow_value):
            continue

        volume_ratio = volume / avg_volume
        is_breakout = close > box_top
        has_volume = volume_ratio > volume_multiplier
        has_trend = close > ma_fast_value and ma_fast_value > ma_slow_value
        if is_breakout and has_volume and has_trend:
            signals.append(
                Signal(
                    signal_index=index,
                    signal_date=pd.Timestamp(df.at[index, "date"]),
                    entry_price=close,
                    box_top=box_top,
                    box_bottom=box_bottom,
                    volume_ratio=volume_ratio,
                )
            )
            next_allowed_index = index + hold_weeks + 1

    return signals


def simulate_trade(
    code: str,
    name: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    box_top: float,
    volume_ratio: float,
    future_weekly: pd.DataFrame,
    hold_weeks: int = 5,
    stop_loss: float = 0.10,
    drawdown: float = 0.20,
) -> Trade:
    future = future_weekly.head(hold_weeks).reset_index(drop=True)
    if future.empty:
        raise ValueError("future_weekly must contain at least one row")

    fixed_stop_price = entry_price * (1 - stop_loss)
    tracked_high = entry_price
    max_high_5w = float(future["high"].max())

    for _, row in future.iterrows():
        tracked_high = max(tracked_high, float(row["high"]))
        drawdown_stop_price = tracked_high * (1 - drawdown)
        low = float(row["low"])
        candidates: list[tuple[str, float]] = []

        if low <= fixed_stop_price:
            candidates.append(("fixed_stop", fixed_stop_price))
        if low <= drawdown_stop_price:
            candidates.append(("trailing_drawdown", drawdown_stop_price))

        if candidates:
            exit_reason, exit_price = min(candidates, key=lambda item: item[1])
            realized_return = exit_price / entry_price - 1
            return Trade(
                code=code,
                name=name,
                entry_date=pd.Timestamp(entry_date),
                entry_price=entry_price,
                box_top=box_top,
                volume_ratio=volume_ratio,
                exit_date=pd.Timestamp(row["date"]),
                exit_price=exit_price,
                exit_reason=exit_reason,
                realized_return=realized_return,
                best_high_return=max_high_5w / entry_price - 1,
                max_high_5w=max_high_5w,
            )

    exit_row = future.iloc[-1]
    exit_price = float(exit_row["close"])
    return Trade(
        code=code,
        name=name,
        entry_date=pd.Timestamp(entry_date),
        entry_price=entry_price,
        box_top=box_top,
        volume_ratio=volume_ratio,
        exit_date=pd.Timestamp(exit_row["date"]),
        exit_price=exit_price,
        exit_reason="time_exit",
        realized_return=exit_price / entry_price - 1,
        best_high_return=max_high_5w / entry_price - 1,
        max_high_5w=max_high_5w,
    )


def backtest_stock(
    code: str,
    name: str,
    daily: pd.DataFrame,
    signal_start: pd.Timestamp,
    box_weeks: int = 20,
    volume_multiplier: float = 1.5,
    hold_weeks: int = 5,
    stop_loss: float = 0.10,
    drawdown: float = 0.20,
    bar_frequency: str = "weekly",
) -> list[Trade]:
    bars = normalize_daily_columns(daily) if bar_frequency == "daily" else aggregate_weekly(daily)
    signals = find_signals(
        bars,
        box_weeks=box_weeks,
        volume_multiplier=volume_multiplier,
        hold_weeks=hold_weeks,
    )
    trades: list[Trade] = []
    for signal in signals:
        if signal.signal_date < signal_start:
            continue
        future = bars.iloc[signal.signal_index + 1 : signal.signal_index + 1 + hold_weeks]
        if len(future) < hold_weeks:
            continue
        trades.append(
            simulate_trade(
                code=code,
                name=name,
                entry_date=signal.signal_date,
                entry_price=signal.entry_price,
                box_top=signal.box_top,
                volume_ratio=signal.volume_ratio,
                future_weekly=future,
                hold_weeks=hold_weeks,
                stop_loss=stop_loss,
                drawdown=drawdown,
            )
        )
    return trades


def retry_call(func, attempts: int = 3, sleep_seconds: float = 1.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise last_error


def load_stock_universe_akshare() -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed")
    try:
        spot = retry_call(ak.stock_zh_a_spot_em)
        stocks = spot[["代码", "名称"]].dropna().drop_duplicates()
        stocks = stocks.rename(columns={"代码": "code", "名称": "name"})
    except Exception:
        stocks = retry_call(ak.stock_info_a_code_name)
        stocks = stocks[["code", "name"]].dropna().drop_duplicates()
    stocks["code"] = stocks["code"].astype(str).str.zfill(6)
    stocks["name"] = stocks["name"].astype(str)
    return stocks.sort_values("code").reset_index(drop=True)


def tushare_code(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def market_code(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def get_tushare_token(env_name: str = "TUSHARE_TOKEN") -> str | None:
    return os.environ.get(env_name) or os.environ.get("TUSHARE_TOKEN") or os.environ.get("TS_TOKEN")


def load_stock_universe_tushare(token: str | None = None, env_name: str = "TUSHARE_TOKEN") -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("tushare is not installed; install tushare and set TUSHARE_TOKEN")
    token = token or get_tushare_token(env_name)
    if not token:
        raise RuntimeError("Tushare token is not set; set TUSHARE_TOKEN or TS_TOKEN")
    pro = ts.pro_api(token)
    stocks = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name")
    stocks = stocks.rename(columns={"symbol": "code"})[["code", "name"]]
    stocks["code"] = stocks["code"].astype(str).str.zfill(6)
    stocks["name"] = stocks["name"].astype(str)
    return stocks.sort_values("code").reset_index(drop=True)


def load_stock_universe(provider: str, args: argparse.Namespace) -> pd.DataFrame:
    if provider == "tushare":
        try:
            return load_stock_universe_tushare(env_name=args.tushare_token_env)
        except Exception as exc:
            print(f"Tushare stock_basic failed: {exc}. Using AkShare stock universe.", flush=True)
            return load_stock_universe_akshare()
    return load_stock_universe_akshare()


def fetch_daily_history_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed")
    return retry_call(
        lambda: ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    )


def fetch_daily_history_akshare_tx(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed")
    start_text = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
    end_text = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
    return retry_call(
        lambda: ak.stock_zh_a_hist_tx(
            symbol=market_code(code),
            start_date=start_text,
            end_date=end_text,
            adjust="qfq",
        )
    )


def fetch_weekly_history_tx(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = market_code(code)
    end_year = datetime.strptime(end_date, "%Y%m%d").year
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    def request_weekly():
        response = requests.get(
            url,
            params={
                "_var": f"kline_weekqfq{end_year}",
                "param": f"{symbol},week,{start_date[:4]}-01-01,{end_year + 1}-12-31,640,qfq",
                "r": "0.1",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.text[response.text.find("={") + 1 :]
        data = json.loads(payload)["data"][symbol]
        records = data.get("qfqweek") or data.get("week") or []
        weekly = normalize_tx_weekly_records(records)
        start_ts = pd.Timestamp(datetime.strptime(start_date, "%Y%m%d"))
        end_ts = pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))
        return weekly[(weekly["date"] >= start_ts) & (weekly["date"] <= end_ts)].reset_index(drop=True)

    return retry_call(request_weekly)


def fetch_daily_history_tushare(
    code: str,
    start_date: str,
    end_date: str,
    env_name: str = "TUSHARE_TOKEN",
) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("tushare is not installed; install tushare and set TUSHARE_TOKEN")
    if args_tushare_mode(env_name) == "legacy":
        return fetch_daily_history_tushare_legacy(code, start_date, end_date)
    token = get_tushare_token(env_name)
    if not token:
        return fetch_daily_history_tushare_legacy(code, start_date, end_date)
    try:
        pro = ts.pro_api(token)
        raw = retry_call(
            lambda: ts.pro_bar(
                ts_code=tushare_code(code),
                api=pro,
                start_date=start_date,
                end_date=end_date,
                adj="qfq",
                asset="E",
            ),
            attempts=1,
        )
        return normalize_tushare_daily(raw)
    except Exception:
        return fetch_daily_history_tushare_legacy(code, start_date, end_date)


def args_tushare_mode(env_name: str) -> str:
    return os.environ.get("DARVAS_TUSHARE_MODE", "legacy").lower()


def fetch_daily_history_tushare_legacy(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("tushare is not installed")
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, ignore_index=False, **kwargs: pd.concat(  # type: ignore[attr-defined]
            [self, other],
            ignore_index=ignore_index,
        )
    start_text = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
    end_text = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
    def get_k_data_quietly():
        with contextlib.redirect_stdout(io.StringIO()):
            return ts.get_k_data(
                code,
                start=start_text,
                end=end_text,
                ktype="D",
                autype="qfq",
            )

    raw = retry_call(get_k_data_quietly)
    return normalize_daily_columns(raw)


def fetch_daily_history(
    provider: str,
    code: str,
    start_date: str,
    end_date: str,
    args: argparse.Namespace | None = None,
) -> pd.DataFrame:
    if provider == "tushare":
        env_name = args.tushare_token_env if args else "TUSHARE_TOKEN"
        return fetch_daily_history_tushare(code, start_date, end_date, env_name=env_name)
    if provider == "akshare_tx":
        return fetch_daily_history_akshare_tx(code, start_date, end_date)
    return fetch_daily_history_akshare(code, start_date, end_date)


def backtest_stock_weekly(
    code: str,
    name: str,
    weekly: pd.DataFrame,
    signal_start: pd.Timestamp,
    args: argparse.Namespace,
) -> list[Trade]:
    signals = find_signals(
        weekly,
        box_weeks=args.box_weeks,
        volume_multiplier=args.volume_multiplier,
        hold_weeks=args.hold_weeks,
    )
    trades: list[Trade] = []
    for signal in signals:
        if signal.signal_date < signal_start:
            continue
        future = weekly.iloc[signal.signal_index + 1 : signal.signal_index + 1 + args.hold_weeks]
        if len(future) < args.hold_weeks:
            continue
        trades.append(
            simulate_trade(
                code=code,
                name=name,
                entry_date=signal.signal_date,
                entry_price=signal.entry_price,
                box_top=signal.box_top,
                volume_ratio=signal.volume_ratio,
                future_weekly=future,
                hold_weeks=args.hold_weeks,
                stop_loss=args.stop_loss,
                drawdown=args.drawdown,
            )
        )
    return trades


def process_stock(
    code: str,
    name: str,
    provider: str,
    start_text: str,
    end_text: str,
    signal_start: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[list[Trade], dict[str, str] | None]:
    try:
        if provider == "tx_weekly":
            if args.bar_frequency != "weekly":
                return [], {"code": code, "name": name, "reason": "tx_weekly provider only supports weekly bars"}
            weekly = fetch_weekly_history_tx(code, start_text, end_text)
            if weekly.empty:
                return [], {"code": code, "name": name, "reason": "empty weekly history"}
            return backtest_stock_weekly(code, name, weekly, signal_start, args), None
        daily = fetch_daily_history(provider, code, start_text, end_text, args)
        if daily.empty:
            return [], {"code": code, "name": name, "reason": "empty history"}
        trades = backtest_stock(
            code=code,
            name=name,
            daily=daily,
            signal_start=signal_start,
            box_weeks=args.box_weeks,
            volume_multiplier=args.volume_multiplier,
            hold_weeks=args.hold_weeks,
            stop_loss=args.stop_loss,
            drawdown=args.drawdown,
            bar_frequency=args.bar_frequency,
        )
        return trades, None
    except Exception as exc:
        if provider == "tushare" and args.fallback_provider:
            fallback_trades, fallback_skip = process_stock(
                code,
                name,
                args.fallback_provider,
                start_text,
                end_text,
                signal_start,
                args,
            )
            if fallback_skip is None:
                return fallback_trades, None
            return fallback_trades, {
                "code": code,
                "name": name,
                "reason": f"tushare failed: {exc}; fallback failed: {fallback_skip['reason']}",
            }
        return [], {"code": code, "name": name, "reason": str(exc)}


def summarize_trades(trades: Iterable[Trade]) -> dict[str, object]:
    trade_list = list(trades)
    if not trade_list:
        return {
            "total_trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "avg_best_high_return_time_exit": np.nan,
            "median_best_high_return_time_exit": np.nan,
            "exit_counts": {},
            "yearly": pd.DataFrame(),
        }

    frame = pd.DataFrame([asdict(trade) for trade in trade_list])
    frame["entry_year"] = pd.to_datetime(frame["entry_date"]).dt.year
    frame["is_win"] = frame["realized_return"] > 0
    time_exit = frame[frame["exit_reason"] == "time_exit"]
    yearly = (
        frame.groupby("entry_year")
        .agg(
            trades=("code", "count"),
            win_rate=("is_win", "mean"),
            avg_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
        )
        .reset_index()
    )
    return {
        "total_trades": int(len(frame)),
        "win_rate": float(frame["is_win"].mean()),
        "avg_return": float(frame["realized_return"].mean()),
        "median_return": float(frame["realized_return"].median()),
        "avg_best_high_return_time_exit": float(time_exit["best_high_return"].mean()) if not time_exit.empty else np.nan,
        "median_best_high_return_time_exit": float(time_exit["best_high_return"].median()) if not time_exit.empty else np.nan,
        "exit_counts": frame["exit_reason"].value_counts().to_dict(),
        "yearly": yearly,
    }


def format_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def write_reports(
    trades: list[Trade],
    skipped: list[dict[str, str]],
    output_dir: Path,
    run_label: str,
    params: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_trades(trades)
    trades_path = output_dir / f"darvas_weekly_trades_{run_label}.csv"
    summary_path = output_dir / f"darvas_weekly_summary_{run_label}.md"

    trade_frame = pd.DataFrame([asdict(trade) for trade in trades])
    if not trade_frame.empty:
        trade_frame.to_csv(trades_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(trades_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Darvas Weekly Backtest Summary",
        "",
        "## Parameters",
        "",
    ]
    for key, value in params.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Total trades: {summary['total_trades']}",
            f"- Win rate: {format_pct(summary['win_rate'])}",
            f"- Average realized return: {format_pct(summary['avg_return'])}",
            f"- Median realized return: {format_pct(summary['median_return'])}",
            f"- Time-exit average best high return: {format_pct(summary['avg_best_high_return_time_exit'])}",
            f"- Time-exit median best high return: {format_pct(summary['median_best_high_return_time_exit'])}",
            "",
            "## Exit Reasons",
            "",
        ]
    )
    for reason, count in summary["exit_counts"].items():
        lines.append(f"- {reason}: {count}")

    yearly = summary["yearly"]
    lines.extend(["", "## Yearly", ""])
    if isinstance(yearly, pd.DataFrame) and not yearly.empty:
        lines.append("| Entry Year | Trades | Win Rate | Avg Return | Median Return |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, row in yearly.iterrows():
            lines.append(
                f"| {int(row['entry_year'])} | {int(row['trades'])} | "
                f"{format_pct(row['win_rate'])} | {format_pct(row['avg_return'])} | {format_pct(row['median_return'])} |"
            )
    else:
        lines.append("No trades.")

    lines.extend(["", "## Data Issues", "", f"- Skipped stocks: {len(skipped)}"])
    for item in skipped[:50]:
        lines.append(f"- {item['code']} {item['name']}: {item['reason']}")
    if len(skipped) > 50:
        lines.append(f"- ... {len(skipped) - 50} more skipped stocks")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return trades_path, summary_path


def run_backtest(args: argparse.Namespace) -> tuple[list[Trade], list[dict[str, str]], Path, Path]:
    end = datetime.strptime(args.end_date, "%Y%m%d").date() if args.end_date else date.today()
    signal_start = pd.Timestamp(end - timedelta(days=365 * args.years))
    if args.bar_frequency == "daily":
        fetch_start = signal_start - pd.Timedelta(days=(max(args.box_weeks, 40) + args.hold_weeks + 20) * 2)
    else:
        fetch_start = signal_start - pd.Timedelta(weeks=max(args.box_weeks, 40) + args.hold_weeks + 5)
    start_text = fetch_start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    scope_label = f"limit{args.limit}" if args.limit else "all"
    provider = args.data_provider

    try:
        stocks = load_stock_universe(provider, args)
    except Exception as exc:
        if provider == "tushare" and args.fallback_provider:
            print(f"Tushare universe failed: {exc}. Falling back to {args.fallback_provider}.", flush=True)
            provider = args.fallback_provider
            stocks = load_stock_universe(provider, args)
        else:
            raise

    run_label = (
        f"{signal_start.strftime('%Y%m%d')}_{end_text}_{provider}_{args.bar_frequency}"
        f"_box{args.box_weeks}_hold{args.hold_weeks}_{scope_label}"
        f"_dd{int(args.drawdown * 100)}"
    )
    if args.limit:
        stocks = stocks.head(args.limit)

    trades: list[Trade] = []
    skipped: list[dict[str, str]] = []
    total = len(stocks)
    rows = list(stocks.itertuples(index=False))
    if args.workers <= 1:
        for position, row in enumerate(rows, start=1):
            code = row.code
            name = row.name
            if args.verbose or position % 100 == 0:
                print(f"[{position}/{total}] {code} {name}", flush=True)
            stock_trades, skip = process_stock(code, name, provider, start_text, end_text, signal_start, args)
            trades.extend(stock_trades)
            if skip:
                skipped.append(skip)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_stock,
                    row.code,
                    row.name,
                    provider,
                    start_text,
                    end_text,
                    signal_start,
                    args,
                ): row
                for row in rows
            }
            for position, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                if args.verbose or position % 100 == 0:
                    print(f"[{position}/{total}] {row.code} {row.name}", flush=True)
                stock_trades, skip = future.result()
                trades.extend(stock_trades)
                if skip:
                    skipped.append(skip)

    params = {
        "signal_start": signal_start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "box_weeks": args.box_weeks,
        "volume_multiplier": args.volume_multiplier,
        "hold_weeks": args.hold_weeks,
        "bar_frequency": args.bar_frequency,
        "stop_loss": args.stop_loss,
        "drawdown": args.drawdown,
        "limit": args.limit or "all",
        "data_provider": provider,
        "requested_data_provider": args.data_provider,
        "workers": args.workers,
    }
    trades_path, summary_path = write_reports(trades, skipped, Path(args.output_dir), run_label, params)
    return trades, skipped, trades_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly Darvas breakout backtest for A-shares.")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--end-date", default=None, help="YYYYMMDD, default today")
    parser.add_argument("--box-weeks", type=int, default=20)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)
    parser.add_argument("--hold-weeks", type=int, default=5)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--drawdown", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports/darvas")
    parser.add_argument("--limit", type=int, default=None, help="Limit stock count for smoke tests")
    parser.add_argument("--data-provider", choices=["tx_weekly", "akshare", "akshare_tx", "tushare"], default="tushare")
    parser.add_argument("--fallback-provider", choices=["tx_weekly", "akshare", "akshare_tx"], default="tx_weekly")
    parser.add_argument("--tushare-token-env", default="TUSHARE_TOKEN")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bar-frequency", choices=["weekly", "daily"], default="weekly")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    trades, skipped, trades_path, summary_path = run_backtest(args)
    summary = summarize_trades(trades)
    print(f"Trades: {summary['total_trades']}")
    print(f"Win rate: {format_pct(summary['win_rate'])}")
    print(f"Average return: {format_pct(summary['avg_return'])}")
    print(f"Median return: {format_pct(summary['median_return'])}")
    print(f"Skipped stocks: {len(skipped)}")
    print(f"Trades CSV: {trades_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
