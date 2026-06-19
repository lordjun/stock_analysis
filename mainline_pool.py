from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import StringIO
from pathlib import Path
import re
from typing import Callable

import pandas as pd
import requests

from kline_model_research import normalize_price_bars
from tdx_data import read_tdx_daily_bars


STRUCTURAL_POOL_COLUMNS = ["code", "name", "board_name", "board_rank", "pool_source"]


@dataclass(frozen=True)
class MainlinePoolConfig:
    cache_dir: Path
    tdx_vipdoc: str
    cache_version: str = "v3"
    lookback_days: int = 540
    top_n: int = 10
    min_ret60: float = 0.05
    min_ret120: float = 0.10
    min_ret250: float = 0.20
    min_breadth120: float = 0.55
    theme_keywords: tuple[str, ...] = ()


class DynamicMainlinePoolProvider:
    def __init__(
        self,
        config: MainlinePoolConfig,
        builder: Callable[[str, MainlinePoolConfig], pd.DataFrame] | None = None,
    ) -> None:
        self.config = config
        self.builder = builder or build_dynamic_mainline_pool
        self._pool_by_date: dict[str, set[str]] = {}
        self._boards_by_date: dict[str, set[str]] = {}
        self._board_rank_by_date: dict[str, dict[str, int]] = {}

    def prepare(self, dates: list[str]) -> None:
        for date in sorted({normalize_date(date) for date in dates}):
            self.codes_for_date(date)

    def codes_for_date(self, date: str) -> set[str]:
        normalized = normalize_date(date)
        if normalized not in self._pool_by_date:
            frame = load_or_build_mainline_frame(normalized, self.config, self.builder)
            self._pool_by_date[normalized] = codes_from_pool_frame(frame)
            self._boards_by_date[normalized] = board_names_from_pool_frame(frame)
            self._board_rank_by_date[normalized] = board_rank_map_from_pool_frame(frame)
        return self._pool_by_date[normalized]

    def boards_for_date(self, date: str) -> set[str]:
        normalized = normalize_date(date)
        if normalized not in self._boards_by_date:
            self.codes_for_date(normalized)
        return self._boards_by_date[normalized]

    def union_codes(self) -> set[str]:
        output: set[str] = set()
        for codes in self._pool_by_date.values():
            output.update(codes)
        return output

    def codes_asof(self, date: str) -> set[str]:
        return codes_asof_date(date, self._pool_by_date)

    def boards_asof(self, date: str) -> set[str]:
        return names_asof_date(date, self._boards_by_date)

    def board_ranks_for_date(self, date: str) -> dict[str, int]:
        normalized = normalize_date(date)
        if normalized not in self._board_rank_by_date:
            self.codes_for_date(normalized)
        return self._board_rank_by_date[normalized]

    def board_ranks_asof(self, date: str) -> dict[str, int]:
        return rank_map_asof_date(date, self._board_rank_by_date)


def normalize_date(date: str | pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y%m%d")


def weekly_snapshot_dates(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = [day.strftime("%Y%m%d") for day in pd.date_range(start, end, freq="W-FRI")]
    end_raw = end.strftime("%Y%m%d")
    if end_raw not in dates:
        dates.append(end_raw)
    return sorted(set(dates))


def business_day_snapshot_dates(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return [day.strftime("%Y%m%d") for day in pd.bdate_range(start, end)]


def load_or_build_mainline_codes(
    end_date: str,
    config: MainlinePoolConfig,
    builder: Callable[[str, MainlinePoolConfig], pd.DataFrame],
) -> set[str]:
    frame = load_or_build_mainline_frame(end_date, config, builder)
    return codes_from_pool_frame(frame)


def load_or_build_mainline_frame(
    end_date: str,
    config: MainlinePoolConfig,
    builder: Callable[[str, MainlinePoolConfig], pd.DataFrame],
) -> pd.DataFrame:
    cache_path = mainline_cache_path(config.cache_dir, end_date, config)
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={"code": str})
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame = builder(end_date, config)
        frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return frame


def mainline_cache_path(cache_dir: Path, end_date: str, config: MainlinePoolConfig | None = None) -> Path:
    suffix = f"_{mainline_config_key(config)}" if config else ""
    return Path(cache_dir) / f"mainline_code_pool_{normalize_date(end_date)}{suffix}.csv"


def mainline_config_key(config: MainlinePoolConfig) -> str:
    raw = "|".join(
        [
            str(config.lookback_days),
            config.cache_version,
            str(config.top_n),
            f"{config.min_ret60:.4f}",
            f"{config.min_ret120:.4f}",
            f"{config.min_ret250:.4f}",
            f"{config.min_breadth120:.4f}",
            ",".join(config.theme_keywords),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def codes_from_pool_frame(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "code" not in frame:
        return set()
    return {str(code).zfill(6) for code in frame["code"].dropna().astype(str)}


def board_names_from_pool_frame(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "board_name" not in frame:
        return set()
    return {str(name).strip() for name in frame["board_name"].dropna().astype(str) if str(name).strip()}


def board_rank_map_from_pool_frame(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "board_name" not in frame or "board_rank" not in frame:
        return {}
    output: dict[str, int] = {}
    for _, row in frame.dropna(subset=["board_name", "board_rank"]).iterrows():
        name = str(row["board_name"]).strip()
        if not name:
            continue
        rank = int(float(row["board_rank"]))
        output[name] = min(rank, output.get(name, rank))
    return output


def codes_asof_date(date: str, pool_by_date: dict[str, set[str]]) -> set[str]:
    normalized = normalize_date(date)
    eligible = [pool_date for pool_date in pool_by_date if pool_date <= normalized]
    if not eligible:
        return set()
    return pool_by_date[max(eligible)]


def names_asof_date(date: str, names_by_date: dict[str, set[str]]) -> set[str]:
    normalized = normalize_date(date)
    eligible = [pool_date for pool_date in names_by_date if pool_date <= normalized]
    if not eligible:
        return set()
    return names_by_date[max(eligible)]


def rank_map_asof_date(date: str, ranks_by_date: dict[str, dict[str, int]]) -> dict[str, int]:
    normalized = normalize_date(date)
    eligible = [pool_date for pool_date in ranks_by_date if pool_date <= normalized]
    if not eligible:
        return {}
    return ranks_by_date[max(eligible)]


def build_dynamic_mainline_pool(end_date: str, config: MainlinePoolConfig) -> pd.DataFrame:
    boards = load_theme_concept_boards(list(config.theme_keywords))
    board_constituents: dict[str, pd.DataFrame] = {}
    for board_name, board_code in boards:
        try:
            board_constituents[board_name] = load_or_fetch_ths_concept_constituents(config.cache_dir, board_code, board_name)
        except Exception as exc:
            print(f"mainline_board_cons_failed board={board_name} reason={str(exc)[:160]}", flush=True)

    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=config.lookback_days)).strftime("%Y%m%d")
    stock_returns: dict[str, dict[str, float]] = {}
    for constituents in board_constituents.values():
        for code, _name in extract_constituent_codes(constituents):
            if code not in stock_returns:
                stock_returns[code] = read_stock_return_features(config.tdx_vipdoc, code, start_date, end_date)

    strength = build_constituent_return_concept_strength(board_constituents, stock_returns)
    selected = select_dynamic_mainline_boards(strength, config)

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
                    "pool_source": "dynamic_constituent_return_strength",
                }
            )
    return pd.DataFrame(rows, columns=STRUCTURAL_POOL_COLUMNS)


def select_dynamic_mainline_boards(strength: pd.DataFrame, config: MainlinePoolConfig) -> pd.DataFrame:
    if strength.empty:
        return strength.head(0)

    strict = strength[
        (strength["ret60_median"] >= config.min_ret60)
        & (strength["ret120_median"] >= config.min_ret120)
        & (strength["ret250_median"] >= config.min_ret250)
        & (strength["breadth120"] >= config.min_breadth120)
    ]
    if not strict.empty:
        return strict.head(config.top_n)

    current_strength = strength[
        (strength["ret60_median"] >= config.min_ret60)
        & (strength["ret120_median"] >= config.min_ret120)
        & (strength["breadth120"] >= config.min_breadth120)
    ]
    if not current_strength.empty:
        return current_strength.head(config.top_n)

    fallback = strength[(strength["ret120_median"] > 0) & (strength["breadth120"] >= 0.50)]
    if not fallback.empty:
        return fallback.head(max(1, min(config.top_n, 5)))
    return strength.head(0)


def load_theme_concept_boards(theme_keywords: list[str]) -> list[tuple[str, str]]:
    if not theme_keywords:
        return []
    import akshare as ak

    names = ak.stock_board_concept_name_ths()
    boards: list[tuple[str, str]] = []
    for _, row in names.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip()
        if name and code and any(keyword.lower() in name.lower() for keyword in theme_keywords):
            boards.append((name, code))
    return boards


def fetch_ths_concept_constituents(board_code: str, board_name: str) -> pd.DataFrame:
    url = f"http://q.10jqka.com.cn/gn/detail/code/{board_code}/"
    response = requests.get(url, headers=ths_headers(), timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    frames = []
    try:
        frames.append(pd.read_html(StringIO(response.text))[0])
    except ValueError as exc:
        raise RuntimeError(f"THS concept constituents parse failed: {board_name}") from exc

    max_page = parse_ths_max_page(response.text)
    for page in range(2, max_page + 1):
        page_url = f"http://q.10jqka.com.cn/gn/detail/field/264648/order/desc/page/{page}/ajax/1/code/{board_code}/"
        try:
            page_response = requests.get(page_url, headers=ths_headers(), timeout=20)
            page_response.raise_for_status()
            page_response.encoding = page_response.apparent_encoding
        except requests.RequestException as exc:
            print(f"ths_concept_page_failed board={board_name} page={page} reason={str(exc)[:120]}", flush=True)
            break
        try:
            frames.append(pd.read_html(StringIO(page_response.text))[0])
        except ValueError:
            break
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_or_fetch_ths_concept_constituents(cache_dir: Path, board_code: str, board_name: str) -> pd.DataFrame:
    cache_path = Path(cache_dir) / "constituents" / f"ths_concept_{board_code}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype=str)
    frame = fetch_ths_concept_constituents(board_code, board_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return frame


def load_or_fetch_stock_concepts(cache_dir: Path, code: str) -> list[str]:
    normalized_code = str(code).zfill(6)
    cache_path = Path(cache_dir) / "stock_concepts" / f"ths_stock_concepts_{normalized_code}.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, dtype=str)
        if "concept" in frame:
            return [str(value).strip() for value in frame["concept"].dropna().astype(str) if str(value).strip()]
    concepts = fetch_ths_stock_concepts(normalized_code)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"concept": concepts}).to_csv(cache_path, index=False, encoding="utf-8-sig")
    return concepts


def fetch_ths_stock_concepts(code: str) -> list[str]:
    normalized_code = str(code).zfill(6)
    url = f"http://basic.10jqka.com.cn/{normalized_code}/concept.html"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "http://basic.10jqka.com.cn/"},
        timeout=20,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "gbk"
    return parse_ths_stock_concepts(response.text)


def parse_ths_stock_concepts(html: str) -> list[str]:
    concepts: list[str] = []
    for match in re.finditer(r'<td[^>]*class=["\']gnName["\'][^>]*>(.*?)</td>', html, flags=re.I | re.S):
        text = re.sub(r"<[^>]+>", "", match.group(1))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            concepts.append(text)
    return list(dict.fromkeys(concepts))


def concept_names_match_mainline(concepts: list[str] | set[str], mainline_boards: set[str]) -> bool:
    return bool(matched_mainline_concepts(concepts, mainline_boards))


def matched_mainline_concepts(concepts: list[str] | set[str], mainline_boards: set[str]) -> list[str]:
    output: list[str] = []
    normalized_boards = [(board, normalize_concept_name(board)) for board in mainline_boards]
    for concept in concepts:
        concept_text = str(concept).strip()
        normalized_concept = normalize_concept_name(concept_text)
        if not normalized_concept:
            continue
        for board, normalized_board in normalized_boards:
            if concept_name_matches(normalized_concept, normalized_board):
                output.append(concept_text)
                break
    return list(dict.fromkeys(output))


def core_mainline_concept_matches(
    concepts: list[str] | set[str],
    rank_by_board: dict[str, int],
    top_rank: int = 5,
    min_matches_outside_top_rank: int = 2,
) -> list[str]:
    matched: list[str] = []
    normalized_ranks = [(board, normalize_concept_name(board), int(rank)) for board, rank in rank_by_board.items()]
    for concept in concepts:
        concept_text = str(concept).strip()
        normalized_concept = normalize_concept_name(concept_text)
        if not normalized_concept:
            continue
        for _board, normalized_board, _rank in normalized_ranks:
            if concept_name_matches(normalized_concept, normalized_board):
                matched.append(concept_text)
                break
    matched = list(dict.fromkeys(matched))
    if any(_best_rank_for_concept(concept, normalized_ranks) <= top_rank for concept in matched):
        return matched
    if len(matched) >= min_matches_outside_top_rank:
        return matched
    return []


def _best_rank_for_concept(concept: str, normalized_ranks: list[tuple[str, str, int]]) -> int:
    normalized_concept = normalize_concept_name(concept)
    ranks = [rank for _board, normalized_board, rank in normalized_ranks if concept_name_matches(normalized_concept, normalized_board)]
    return min(ranks) if ranks else 999


def normalize_concept_name(name: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(name))
    text = re.sub(r"（[^）]*）", "", text)
    text = text.replace("概念", "")
    return re.sub(r"[\s_\-—/|、,，;；?？]+", "", text).upper()


def concept_name_matches(concept: str, board: str) -> bool:
    if not concept or not board:
        return False
    return concept == board or (len(concept) >= 2 and concept in board) or (len(board) >= 2 and board in concept)


def parse_ths_max_page(html: str, max_pages: int = 80) -> int:
    pages = [int(match) for match in re.findall(r'page=["\'](\d+)["\']', html)]
    if not pages:
        return 1
    return max(1, min(max(pages), max_pages))


def ths_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://q.10jqka.com.cn/",
    }


def extract_constituent_codes(frame: pd.DataFrame) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for _, row in frame.iterrows():
        code = ""
        name = ""
        for value in row.tolist():
            text = str(value).strip()
            match = re.fullmatch(r"\d{6}", text)
            if match:
                code = text
                break
        if not code:
            continue
        for value in row.tolist():
            text = str(value).strip()
            if text and text != code and not re.fullmatch(r"[-+]?\d+(\.\d+)?%?", text):
                if not re.search(r"\d{6}", text):
                    name = text
                    break
        rows.append((code, name))
    return rows


def read_stock_return_features(tdx_vipdoc: str, code: str, start_date: str, end_date: str) -> dict[str, float]:
    try:
        bars = normalize_price_bars(read_tdx_daily_bars(tdx_vipdoc, code, start_date, end_date))
    except Exception:
        return {"ret60": float("nan"), "ret120": float("nan"), "ret250": float("nan")}
    if bars.empty:
        return {"ret60": float("nan"), "ret120": float("nan"), "ret250": float("nan")}
    close = pd.to_numeric(bars["close"], errors="coerce").dropna().reset_index(drop=True)
    return {
        "ret60": window_return(close, 60),
        "ret120": window_return(close, 120),
        "ret250": window_return(close, 250),
    }


def window_return(close: pd.Series, bars: int) -> float:
    if len(close) <= bars or float(close.iloc[-bars - 1]) <= 0:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-bars - 1] - 1)


def build_constituent_return_concept_strength(
    board_constituents: dict[str, pd.DataFrame],
    stock_returns: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for board_name, constituents in board_constituents.items():
        codes = [code for code, _name in extract_constituent_codes(constituents)]
        frame = pd.DataFrame([stock_returns.get(code, {}) for code in codes])
        if frame.empty or "ret120" not in frame:
            continue
        frame = frame.apply(pd.to_numeric, errors="coerce")
        valid = frame.dropna(subset=["ret60", "ret120", "ret250"])
        if len(valid) < 3:
            continue
        ret60_median = float(valid["ret60"].median())
        ret120_median = float(valid["ret120"].median())
        ret250_median = float(valid["ret250"].median())
        ret120_p75 = float(valid["ret120"].quantile(0.75))
        breadth120 = float((valid["ret120"] > 0).mean())
        score = ret120_median * 0.35 + ret60_median * 0.25 + ret250_median * 0.20 + ret120_p75 * 0.10 + breadth120 * 0.10
        rows.append(
            {
                "board_name": board_name,
                "member_count": len(valid),
                "ret60_median": ret60_median,
                "ret120_median": ret120_median,
                "ret250_median": ret250_median,
                "ret120_p75": ret120_p75,
                "breadth120": breadth120,
                "mainline_score": score,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "board_name",
                "member_count",
                "ret60_median",
                "ret120_median",
                "ret250_median",
                "ret120_p75",
                "breadth120",
                "mainline_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["mainline_score", "ret120_median", "ret60_median"],
        ascending=False,
        ignore_index=True,
    )


def parse_theme_keywords(raw: str | list[str]) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(item.strip() for item in raw if str(item).strip())
    return tuple(item.strip() for item in re.split(r"[,;，；\s]+", str(raw)) if item.strip())
