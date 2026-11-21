"""Read delivered bundles (zips or extracted folders) into pandas.

Bundle layouts as delivered:
- Polymarket, one day: ``<base>/{l1,l2,trades}.parquet``, ``<base>/reference/*.csv``, ``<base>/known_gaps.csv``.
- Polymarket, a range: ``<base>/<slug>/date=D/`` holding that day's files as above
  (reference CSVs per day; known_gaps.csv only on a day with gaps; a day without
  data has no folder). This reader adds ``date`` (and ``slug`` for a range
  bundle), and merges the days' reference tables with the latest day winning on
  a duplicate id, since later days can list markets that earlier days do not.
- Kalshi: ``<base>/kalshi_<table>/date=D/<SERIES>/<table>.parquet``; adds ``date`` and ``series``.
- Hyperliquid: ``<base>/hl_<table>/date=D/<coin>/<table>.parquet`` and ``quality/date=D/<coin>.json``; adds ``date``.

Tables of different days are concatenated with missing columns filled with
nulls, so a range that starts before a column existed (Kalshi ``venue_clock``
from 2026-10-29) keeps the column instead of losing it to the first file's schema.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import TickFoundryError

_DATE_DIR = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
_PM_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-|$)")


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract a zip, refusing members that would land outside ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            target = (dest / m.filename).resolve()
            if root != target and root not in target.parents:
                raise TickFoundryError(f"refusing to extract {m.filename!r}: it points outside {dest}")
        zf.extractall(dest)
    return dest


@dataclass
class Data:
    """What ``pull`` and ``sample`` return. Each table is a pandas DataFrame (empty when not requested)."""

    venue: str
    l1: pd.DataFrame = field(default_factory=pd.DataFrame)
    l2: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)  # kalshi ticker / cfbenchmarks
    reference: dict[str, pd.DataFrame] = field(default_factory=dict)  # Polymarket reference/*.csv
    known_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    folders: list[Path] = field(default_factory=list)  # one folder per bundle day, each one LocalReplay.from_bundle

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name in ("l1", "l2", "trades"):
            return getattr(self, name)
        if name in self.extra:
            return self.extra[name]
        raise KeyError(name)

    def __repr__(self) -> str:
        parts = [f"{k}={len(getattr(self, k)):,} rows" for k in ("l1", "l2", "trades") if not getattr(self, k).empty]
        parts += [f"{k}={len(v):,} rows" for k, v in self.extra.items()]
        return f"Data({self.venue}: {', '.join(parts) or 'empty'}; {len(self.folders)} bundle(s))"


def _concat(tables: list[pa.Table]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pa.concat_tables(tables, promote_options="default").to_pandas()


def _with(table: pa.Table, **cols: str) -> pa.Table:
    for name, value in cols.items():
        if name not in table.column_names:
            table = table.append_column(name, pa.array([value] * table.num_rows, pa.string()))
    return table


def bundle_root(folder: Path) -> Path:
    """The bundle's top folder: every zip holds one ``<base>/`` directory."""
    subs = [p for p in folder.iterdir() if p.is_dir()]
    if len(subs) == 1 and not any(p.is_file() and p.suffix == ".parquet" for p in folder.iterdir()):
        return subs[0]
    return folder


def read_polymarket(folders: Iterable[Path], layers: Iterable[str], dates: Iterable[str | None] | None = None) -> Data:
    out = Data("polymarket")
    tables: dict[str, list[pa.Table]] = {k: [] for k in ("l1", "l2", "trades")}
    refs: dict[str, list[pd.DataFrame]] = {}
    gaps: list[pd.DataFrame] = []
    dates_list = list(dates) if dates is not None else None
    day_roots: list[tuple[Path, str, str | None]] = []  # (folder with the day's files, date, slug)
    for i, folder in enumerate(folders):
        root = bundle_root(folder)
        if (root / "l1.parquet").exists() or not any(root.glob("*/date=*")):
            day = dates_list[i] if dates_list and i < len(dates_list) else None
            if day is None:
                m = _PM_DATE.search(root.name)
                day = m.group(1) if m else ""
            day_roots.append((root, day, None))
        else:  # a range bundle: one folder per series and day
            for d in sorted(p for p in root.glob("*/date=*") if p.is_dir()):
                m = _DATE_DIR.match(d.name)
                day_roots.append((d, m.group(1) if m else "", d.parent.name))
    day_roots.sort(key=lambda r: (r[1], r[2] or ""))
    for root, day, slug in day_roots:
        out.folders.append(root)
        extra_cols = {"date": day, **({"slug": slug} if slug else {})}
        for layer in tables:
            f = root / f"{layer}.parquet"
            if layer in layers and f.exists():
                tables[layer].append(_with(pq.read_table(f), **extra_cols))
        refdir = root / "reference"
        if refdir.is_dir():
            for csv in sorted(refdir.glob("*.csv")):
                refs.setdefault(csv.stem, []).append(pd.read_csv(csv, dtype=str))
        g = root / "known_gaps.csv"
        if g.exists():
            df = pd.read_csv(g, dtype=str)
            df.insert(0, "date", day)
            if slug:
                df.insert(1, "slug", slug)
            gaps.append(df)
    out.l1, out.l2, out.trades = (_concat(tables[k]) for k in ("l1", "l2", "trades"))
    out.reference = {k: _merge_reference(k, v) for k, v in refs.items()}
    out.known_gaps = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame()
    return out


_REFERENCE_KEYS = {"markets": "condition_id", "tokens": "token_id", "events": "event_id", "series": "series_id"}


def _merge_reference(name: str, days: list[pd.DataFrame]) -> pd.DataFrame:
    """The days' copies of one reference table, in day order; the latest day wins on a duplicate id."""
    df = pd.concat(days, ignore_index=True)
    key = _REFERENCE_KEYS.get(name)
    if key and key in df.columns:
        return df.drop_duplicates(subset=[key], keep="last", ignore_index=True)
    return df.drop_duplicates(keep="last", ignore_index=True)


def _partitioned(root: Path, prefix: str, key_col: str) -> dict[str, list[pa.Table]]:
    """Read ``<prefix><table>/date=D/<KEY>/<table>.parquet`` files grouped by table."""
    found: dict[str, list[pa.Table]] = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        table_name = ds_dir.name[len(prefix):]
        for date_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            m = _DATE_DIR.match(date_dir.name)
            if not m:
                continue
            for key_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                for f in sorted(key_dir.glob("*.parquet")):
                    t = _with(pq.read_table(f), date=m.group(1), **{key_col: key_dir.name})
                    found.setdefault(table_name, []).append(t)
    return found


def read_kalshi(folders: Iterable[Path]) -> Data:
    out = Data("kalshi")
    found: dict[str, list[pa.Table]] = {}
    for folder in folders:
        root = bundle_root(folder)
        out.folders.append(root)
        for k, v in _partitioned(root, "kalshi_", "series").items():
            found.setdefault(k, []).extend(v)
    for k, v in found.items():
        df = _concat(v)
        if k in ("l1", "l2", "trades"):
            setattr(out, k, df)
        else:
            out.extra[k] = df
    return out


def read_hyperliquid(folders: Iterable[Path]) -> Data:
    out = Data("hyperliquid")
    found: dict[str, list[pa.Table]] = {}
    quality: list[dict] = []
    for folder in folders:
        root = bundle_root(folder)
        out.folders.append(root)
        for k, v in _partitioned(root, "hl_", "coin").items():
            found.setdefault(k, []).extend(v)
        qdir = root / "quality"
        if qdir.is_dir():
            for f in sorted(qdir.glob("date=*/*.json")):
                try:
                    q = json.loads(f.read_text())
                except ValueError:
                    continue
                m = _DATE_DIR.match(f.parent.name)
                quality.append({"date": m.group(1) if m else None, "coin": f.stem, **(q if isinstance(q, dict) else {"quality": q})})
    for k, v in found.items():
        df = _concat(v)
        if k in ("l1", "l2", "trades"):
            setattr(out, k, df)
        else:
            out.extra[k] = df
    out.known_gaps = pd.DataFrame(quality)
    return out


def cut_to_window(data: Data, start_ns: int | None, end_ns: int | None) -> Data:
    """Keep rows whose time is in [start_ns, end_ns). Time = ``ts_recv_ns``, or ``ts_src_ms`` where the receive time is null."""
    if start_ns is None and end_ns is None:
        return data

    def cut(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "ts_recv_ns" in df.columns:
            t = df["ts_recv_ns"].astype("Int64")
            if "ts_src_ms" in df.columns:
                t = t.fillna(df["ts_src_ms"].astype("Int64") * 1_000_000)
        elif "ts_src_ms" in df.columns:
            t = df["ts_src_ms"].astype("Int64") * 1_000_000
        else:
            return df
        keep = t.notna()
        if start_ns is not None:
            keep &= t >= start_ns
        if end_ns is not None:
            keep &= t < end_ns
        out = df.loc[keep.fillna(False).astype(bool).to_numpy()]
        return cast(pd.DataFrame, out).reset_index(drop=True)

    data.l1, data.l2, data.trades = cut(data.l1), cut(data.l2), cut(data.trades)
    data.extra = {k: cut(v) for k, v in data.extra.items()}
    return data
