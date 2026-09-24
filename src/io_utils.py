"""Loading, memory reduction, caching. Built for files that don't fit comfortably."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, DATA_RAW


def list_data(root: Path | str = DATA_RAW) -> pd.DataFrame:
    """Every file under the data root with size — first thing to run on new data."""
    root = Path(root)
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "mb", "suffix"])
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(root)),
                "mb": round(p.stat().st_size / 1024**2, 2),
                "suffix": p.suffix.lower(),
            })
    return pd.DataFrame(rows).sort_values("mb", ascending=False).reset_index(drop=True)


def peek(path: Path | str, n: int = 5) -> pd.DataFrame:
    """Read only the first n rows — instant schema check on a multi-GB file."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(n)
    return pd.read_csv(path, nrows=n)


def count_rows(path: Path | str) -> int:
    """Row count without loading the file into memory."""
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    with open(path, "rb") as fh:
        return sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1024 * 1024), b"")) - 1


def reduce_mem(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numerics in place-ish. Typically cuts a wide frame by 50-70%.

    Floats go to float32, never float16 — float16 silently destroys precision
    on money/measurement columns and has cost people real leaderboard points.
    """
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min):
                continue
            for cand in (np.int8, np.int16, np.int32, np.int64):
                info = np.iinfo(cand)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(cand)
                    break
        elif pd.api.types.is_float_dtype(dt):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"reduce_mem: {start:.1f} MB -> {end:.1f} MB ({100 * (1 - end / max(start, 1e-9)):.0f}% saved)")
    return df


def load(path: Path | str, reduce: bool = True, **kwargs) -> pd.DataFrame:
    """Load csv/tsv/parquet/json-lines with a timing line and optional downcast."""
    path = Path(path)
    t0 = time.time()
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        df = pd.read_parquet(path, **kwargs)
    elif suf in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True, **kwargs)
    elif suf == ".json":
        df = pd.read_json(path, **kwargs)
    elif suf in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t", **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    print(f"loaded {path.name}: {df.shape[0]:,} x {df.shape[1]} in {time.time() - t0:.1f}s")
    return reduce_mem(df) if reduce else df


def cache_frame(key: str, builder, force: bool = False) -> pd.DataFrame:
    """Memoise an expensive frame to parquet in data/interim.

    Feature engineering that takes minutes should run once per session, not
    once per cell execution.
    """
    path = DATA_INTERIM / f"{key}.parquet"
    if path.exists() and not force:
        print(f"cache hit: {path.name}")
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    print(f"cache write: {path.name} ({path.stat().st_size / 1024**2:.1f} MB)")
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Short stable hash of a frame — catches accidental data changes between runs."""
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.head(1000), index=False).values.tobytes())
    return h.hexdigest()[:12]
