"""Normalise every record once and cache it as parquet.

Normalisation costs about six minutes per full pass over train, which is
dead weight on every experiment. Paying it once turns the blocking
iteration loop from minutes into seconds.

Two forms are stored per field:
  *_norm  -- transliterated, suffix-stripped. What Stage 2 should score.
  *_key   -- the above plus phonetic folding. Recall-first, blocking only.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalize import blocking_key, normalize, script_of  # noqa: E402

D = Path("/home/user/Amazon-ML/data/raw/student_resource/dataset")
OUT = Path("/home/user/Amazon-ML/data/interim")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 100_000


def process(rows: list[tuple[str, str, str, str]]) -> dict:
    eid, cty = [], []
    nn, an, nk, ak, sc = [], [], [], [], []
    for e, name, addr, c in rows:
        eid.append(e)
        cty.append(c)
        n_norm = normalize(name)
        a_norm = normalize(addr)
        nn.append(n_norm)
        an.append(a_norm)
        nk.append(blocking_key(name))
        ak.append(blocking_key(addr))
        sc.append(script_of(name))
    return {"entity_id": eid, "country": cty, "name_norm": nn, "addr_norm": an,
            "name_key": nk, "addr_key": ak, "script": sc}


def chunks(path: Path):
    buf = []
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                buf.append((p[0], p[1], p[2], p[3]))
                if len(buf) >= CHUNK:
                    yield buf
                    buf = []
    if buf:
        yield buf


SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("country", pa.string()),
    ("name_norm", pa.string()), ("addr_norm", pa.string()),
    ("name_key", pa.string()), ("addr_key", pa.string()),
    ("script", pa.string()),
])


def build(split: str, src: int, pool: Pool) -> None:
    src_path = D / split / f"{split}_source{src}.tsv"
    out_path = OUT / f"{split}_source{src}.parquet"
    if out_path.exists():
        print(f"  skip {out_path.name} (exists)", flush=True)
        return
    t = time.time()
    n = 0
    # write to a temp path and rename on success, so an interrupted run
    # cannot leave a truncated file that the next run mistakes for complete
    tmp_path = out_path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
    try:
        for res in pool.imap(process, chunks(src_path), chunksize=1):
            writer.write_table(pa.Table.from_pydict(res, schema=SCHEMA))
            n += len(res["entity_id"])
        writer.close()
        tmp_path.rename(out_path)
    except BaseException:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    mb = out_path.stat().st_size / 1024**2
    print(f"  {out_path.name}: {n:,} rows, {mb:.0f} MB, {time.time()-t:.0f}s", flush=True)


def main() -> None:
    t0 = time.time()
    with Pool(4) as pool:
        for split in ("train", "test"):
            print(f"{split}:", flush=True)
            for src in (1, 2, 3):
                build(split, src, pool)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
