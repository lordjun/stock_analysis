from __future__ import annotations

from pathlib import Path
import struct

import pandas as pd


TDX_RECORD_SIZE = 32


def market_prefix(code: str) -> str:
    text = str(code).strip().zfill(6)
    if text.startswith(("5", "6", "9")) and not text.startswith("92"):
        return "sh"
    if text.startswith(("8", "92")):
        return "bj"
    return "sz"


def tdx_day_path(vipdoc_root: Path | str, code: str) -> Path:
    root = Path(vipdoc_root)
    prefix = market_prefix(code)
    text = str(code).strip().zfill(6)
    return root / prefix / "lday" / f"{prefix}{text}.day"


def read_tdx_daily_bars(
    vipdoc_root: Path | str,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    return read_tdx_day_file(tdx_day_path(vipdoc_root, code), start_date, end_date)


def read_tdx_day_file(
    path: Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    data = file_path.read_bytes()
    rows: list[dict[str, object]] = []
    for offset in range(0, len(data), TDX_RECORD_SIZE):
        chunk = data[offset : offset + TDX_RECORD_SIZE]
        if len(chunk) != TDX_RECORD_SIZE:
            continue
        raw_date, open_, high, low, close, amount, volume, _reserved = struct.unpack("<IIIIIfII", chunk)
        rows.append(
            {
                "date": pd.to_datetime(str(raw_date)),
                "open": open_ / 100.0,
                "high": high / 100.0,
                "low": low / 100.0,
                "close": close / 100.0,
                "amount": float(amount),
                "volume": float(volume),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def aggregate_tdx_weekly_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    daily = daily_bars.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["_actual_date"] = daily["date"]
    weekly = (
        daily.set_index("date")
        .resample("W-FRI")
        .agg(
            {
                "_actual_date": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .rename(columns={"_actual_date": "date"})
        .reset_index(drop=True)
    )
    return weekly[["date", "open", "high", "low", "close", "volume"]]


def read_tdx_weekly_bars(
    vipdoc_root: Path | str,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    return aggregate_tdx_weekly_bars(read_tdx_daily_bars(vipdoc_root, code, start_date, end_date))
