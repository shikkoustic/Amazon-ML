"""Stage 1: candidate generation.

Blocking cuts the search space from every possible pair down to a shortlist
the matcher can afford to score. Two properties govern the design:

- Anything dropped here is unrecoverable. No downstream model can score a
  pair it never sees, so recall is the objective and precision is not.
- Matched records always share a country (verified on all 7,638,365 true
  pairs, zero exceptions), so the search partitions by country for free.

Candidates come from the union of independent passes. Names and addresses
fail on different records -- a name in Devanagari against a Latin query
scores zero, while its address is often near-identical -- so a pass that
looks weak alone can still be the only thing covering a slice of the data.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


@dataclass
class BlockingConfig:
    """Knobs, with the defaults measurement settled on."""

    # (2,4) char n-grams survive typos and word transposition better than
    # word tokens, and work unchanged on transliterated Indic text.
    ngram_range: tuple[int, int] = (2, 4)
    analyzer: str = "char_wb"
    min_df: int = 2
    max_features: int | None = 400_000

    # True matches per entity top out at 5 from S2 and 6 from S3, so 25 per
    # source per signal leaves roughly 4x headroom for ranking error.
    top_k: int = 25
    # Kept deliberately low: blocking should over-collect and let Stage 2
    # discriminate. Raising this trades recall for a smaller candidate set.
    min_sim: float = 0.02
    # Rows of Source 1 scored against the index at once. Bounds peak memory
    # without changing results.
    chunk_size: int = 20_000


@dataclass
class PassStats:
    name: str
    pairs: int = 0
    seconds: float = 0.0


@dataclass
class BlockingResult:
    """Candidates as a mapping from Source 1 id to the set of ids found."""

    candidates: dict[str, set[str]] = field(default_factory=dict)
    stats: list[PassStats] = field(default_factory=list)

    def add(self, s1_id: str, other_id: str) -> None:
        self.candidates.setdefault(s1_id, set()).add(other_id)

    @property
    def n_pairs(self) -> int:
        return sum(len(v) for v in self.candidates.values())


def build_index(texts: list[str], cfg: BlockingConfig):
    """Fit TF-IDF over the index side and return (vectorizer, matrix.T).

    The transpose is returned because sp_matmul_topn wants the index
    oriented features-by-documents, and transposing once per index beats
    transposing once per query chunk.
    """
    vec = TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts)
    return vec, X.T.tocsr()


def topk_neighbours(query_texts: list[str], vec, index_T, cfg: BlockingConfig):
    """Yield (query_row, index_col, similarity) for the top matches.

    Queries run in chunks so peak memory stays flat regardless of how many
    Source 1 records a country holds.
    """
    n = len(query_texts)
    for start in range(0, n, cfg.chunk_size):
        stop = min(start + cfg.chunk_size, n)
        Q = vec.transform(query_texts[start:stop])
        C = sp_matmul_topn(Q, index_T, top_n=cfg.top_k,
                           threshold=cfg.min_sim, sort=False, n_threads=4)
        C = C.tocoo()
        yield start + C.row, C.col, C.data
        del Q, C
        gc.collect()
