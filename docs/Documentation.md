# ML Challenge 2026: Business Entity Resolution

**Team Name:** _[fill in]_
**Team Members:** _[fill in]_
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

Two-stage entity resolution: word-level TF-IDF blocking narrows 22.7 trillion
possible pairs to 100 candidates per Source-1 entity, then a gradient-boosted
classifier over 79 similarity features decides each pair. Two things
distinguish it. First, the strongest features are *relative* — a candidate's
rank, its gap to the best candidate and its z-score within that entity's own
candidate set — because under a per-entity metric the question is never "is
this pair similar" but "is this the best explanation available for this
record". Second, a cross-encoder reads only the ~1% of pairs the classifier is
genuinely unsure about, which is where every error lives.

---

## 2. Methodology

### 2.1 Problem Analysis

The three sources describe the same businesses with independent noise. What the
data actually contains, measured rather than assumed:

**Countries are not interchangeable.** Training covers the US and India only.
The test set is 15% France. Any normalisation tuned on training and applied
globally is unvalidated on a sixth of the score — we learned this the
expensive way (§5).

**Addresses are damaged in structured ways.** Street numbers drop digits
(`780` → `80`), ordinals alternate between forms (`11th` / `eleventh`),
directionals and unit designators appear and vanish (`ste 200`, `apt 4b`), and
letters are swapped for lookalike digits (`8est` for `best`). These are not
random edits; each has a rule.

**Names are abbreviated as subsequences, not as lookups.** `intl` for
`international`, `mgmt` for `management`, `bmw` for
`bayerische motoren werke`. A hand-written abbreviation table cannot cover
this and actively hurts (§5); testing whether the short form is a subsequence
of the long one does, and generalises to forms nobody wrote down.

**Transliteration dominates the Indian data.** The same street is spelled
`Nehru`, `Nehroo`, `Naehru`. Dropping vowels and folding
consonant classes to a skeleton makes these identical without touching
anything that should stay distinct.

**5.58% of Source-1 entities have no match at all.** Under this metric those
score 1.0 for silence and 0.0 for any guess, so the model must be willing to
return nothing.

**The hardest class is same-name-different-address.** It is simultaneously 31%
of the missed matches and 38% of the false positives: two branches of one
chain, and one record with a corrupted address, look identical to a feature
set. This single class is the ceiling on the current approach.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier, with a transformer cascade on the
uncertain band.

**Core Innovation:** Deciding *where* the model needs help before building
anything to help it. The classifier's scores are extremely concentrated —
pairs scoring below 0.05 are 0.1% true and pairs above 0.95 are 99.2% true, so
every mistake in the submission sits inside 0.42% of pairs. That measurement
turns an open-ended modelling problem into a bounded one: resolving the
0.05–0.95 band perfectly is worth 0.9319 → 0.9753 on held-out data, and the
first-stage score identifies those pairs for free. A cross-encoder that would
be far too slow to run on 173 million pairs runs comfortably on 1% of them.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:** word-level TF-IDF over normalised business name and
address, on each (source, country) partition separately, retrieved by sparse
top-K cosine similarity (`sparse_dot_topn`). Three retrieval channels are
unioned: name-only, address-only, and name+address combined. Character
trigrams (`char_wb`) were built and measured as a fourth channel and did not
add recall once normalisation was repaired, so they are not in the final set.

Normalisation before blocking does the real work: Unicode folding, legal-form
removal (`inc`, `ltd`, `pvt`, `sarl`), country-aware abbreviation expansion,
consonant-skeleton transliteration, lookalike-digit repair and singularisation.

**Candidate pairs generated:** 50 per Source-1 entity from Source 2 and 50
from Source 3 — 100 per entity, 173,254,400 pairs across the 1,732,544 test
entities.

**How we ensured true matches were not lost:** measured, not asserted. With a
perfect matcher on this candidate set the metric reaches **0.9886**, so
blocking gives up 1.14%. That number was the basis for stopping work on
blocking: a full loss decomposition (§5) attributed only 15.3% of the total
loss to candidates never proposed, against 57.6% to the matcher rejecting
pairs already in front of it. Widening the candidate set from 25 to 100 per
entity, and from one retrieval channel to three, moved recall by a fraction of
a point and the final score by nothing measurable.

---

## 4. Matching Model

**Features used (79 total):**

- **Name:** token Jaccard, character-trigram Jaccard, IDF-weighted overlap and
  containment, SoftTF-IDF with a subsequence-based abbreviation test,
  consonant-skeleton match, length ratio, initials match, rare-token
  agreement.
- **Address:** the same similarity family, plus house-number equality and a
  digit-subsequence test for dropped digits, PIN/ZIP agreement, ordinal-token
  normalisation, unit/suite-number compatibility, and directional agreement.
- **Cross-source:** agreement between what Source 2 and Source 3 candidates
  say about the same Source-1 record — a candidate corroborated from the other
  source is much likelier to be real.
- **Relative (within the entity's candidate set):** rank, gap to the best
  candidate, ratio to the best candidate, z-score, is-argmax, and whether this
  is the *only* candidate with a near-exact address. These matter because the
  metric is per-entity. A pair scoring 0.6 when the next best scores 0.59 is a
  different proposition from the same 0.6 when the next best scores 0.2.

**Model type:** LightGBM binary classifier (MIT licence), 79 features, trained
on 2,000,000 labelled candidate pairs with a per-entity 75/25 split — split by
entity, never by pair, or the relative features leak.

**Second stage:** a MiniLM cross-encoder (Apache-2.0, 22M parameters), fine
tuned on pairs the first stage scored between 0.05 and 0.95, reading
`name | address` for both records as one sequence. It lifts in-band AUC from
0.8043 to 0.8907. Its score is blended 0.6/0.4 with the first stage's inside
the band and ignored outside it.

**Threshold selection method:** F_0.5 maximisation on the held-out split.
First stage alone the optimum is **0.70** — far above 0.5, because precision
enters the metric at twice the weight of recall, so two correct matches plus
one wrong (0.714) scores worse than one correct match alone (0.833). With the
cascade the optimum moves to **0.55**, because the blend compresses scores
toward the middle.

Alternatives tested and rejected: per-entity expected-F_0.5 maximisation
(–0.0005 against a flat threshold), isotonic calibration, and a low-threshold
"rescue" rule for entities that would otherwise be empty (–0.0035).

---

## 5. Results & Error Analysis

**F_0.5 Score (macro), held-out validation**, 5,000 Source-1 entities, seed 42,
split by entity. Every number measured on the same split:

    0.9112   TF-IDF blocking + 27 features
    0.9143   word-level blocking + region features
    0.9164   normalisation rebuilt (transliteration, legal forms, variants)
    0.9256   cross-source agreement features
    0.9307   street-number and digit-subsequence repair
    0.9319   name containment + ordinal normalisation
    0.9437   cross-encoder cascade on the 0.05-0.95 band

**Where the loss is.** Decomposing before optimising was the single most
useful thing we did:

    perfect                                              1.0000
    ceiling: perfect matcher on our candidates           0.9886
    no false positives, recall unchanged                 0.9458
    actual                                               0.9319

    matcher rejects a true pair already in candidates      57.6% of loss
    matcher adds a false positive                          27.1%
    blocking never proposed the pair                       15.3%

**Common false positives (wrong merges):** two branches of the same chain in
the same city — identical names, addresses that differ only in a street number
we cannot trust because street numbers are also the field most often damaged.
38% of false positives are this one class. Franchise names shared across
unrelated owners are the second class.

**Common false negatives (missed matches):** the same class from the other
side (31%), then records where one side's address is empty (19%) and the name
alone cannot reach the confidence a double-weighted-precision metric demands,
then trade-name substitution where the business is listed under a name that
shares no tokens with the registered one (10%).

**Approaches measured and rejected**, recorded because each cost hours:
wider and additional blocking channels (recall flat across a 7× increase in
candidates); 7× more training data (flat); character-trigram name blocking as
a fourth channel (flat after normalisation was fixed); a second gradient
boosted model specialised on the uncertain band (0.7787 in-band against 0.8043
for the general model — the features are exhausted, which is what sent us to a
cross-encoder); hand-written abbreviation and region tables (actively harmful,
below).

---

## 6. Conclusion

A carefully normalised blocking stage plus a gradient-boosted matcher over 79
features — a third of them relative to the entity's own candidate set — reaches
0.9319 held out, and a cross-encoder over the 1% of pairs the matcher is unsure
about adds 0.0118 more. The lesson that transferred furthest was to decompose
the loss before optimising: it showed blocking was a sixth of the problem after
we had already spent a day on it, and it showed that every remaining error was
concentrated in a band small enough for a transformer to afford.

The lesson that cost the most was that hard-coded knowledge does not
generalise across countries. A US state-abbreviation table expanded `de` and
`la` inside French addresses, turning `rue de la Paix` into
`rue delaware louisiana paix` 119,000 times. France is 15% of the test set and
absent from training, so validation never saw it and the leaderboard score fell
from 0.9243 to 0.907. Every replacement afterwards had to be a rule with a
stated domain rather than an entry in a list.

---

## Appendix

### A. Code Artefacts

Everything ships under `code/business_entity_resolution/`, with a `README.md`
and `requirements.txt`. Dependencies are scikit-learn, LightGBM,
sparse_dot_topn, pyarrow, unidecode and transformers.

    src/normalize.py          country-aware normalisation and blocking keys
    src/variants.py           abbreviation tables, split global vs short-key
    src/features.py           the 79 pair features
    src/scoring.py            macro F_0.5, verified against the worked example
    scripts/build_cache.py    raw TSV -> normalised parquet
    scripts/word_candidates.py  TF-IDF blocking -> candidate_pairs.tsv
    scripts/union_pairs.py    candidates + ground truth -> labelled pairs
    scripts/build_relative.py  pair features + within-entity relative features
    scripts/train_v2.py       trains the matcher and sweeps the threshold
    scripts/predict_test.py   scores test candidates -> matching_results.tsv
    scripts/evaluate.py       scores any predictions file, standard library only
    kaggle_kernel/train_crossencoder.py  second-stage training (GPU)

To reproduce both outputs from the raw data:

    python scripts/build_cache.py --split test
    python scripts/word_candidates.py --split test --analyzer word \
        --out output/candidate_pairs.tsv
    python scripts/predict_test.py --candidates output/candidate_pairs.tsv \
        --split test --model matcher_v2.txt --threshold 0.55 --workers 2 \
        --crossencoder models/crossencoder_v1 --ce-band 0.05,0.95 \
        --ce-weight 0.6 --out output/matching_results.tsv

`predict_test.py` is resumable: each shard writes its own result and `--resume`
skips the ones already finished.

### B. Additional Results

**Score concentration**, held-out pairs by first-stage score:

    score band        share of pairs     share true
    < 0.05                    97.9%           0.1%
    0.05 - 0.95                1.0%          ~50%
    > 0.95                     1.1%          99.2%

**Cross-encoder band width**, held-out, blend weight and threshold swept:

    band            band pairs    F_0.5     delta
    none (baseline)          -   0.9319         -
    0.20 - 0.80          2,101   0.9388   +0.0068
    0.15 - 0.85          2,690   0.9415   +0.0095
    0.10 - 0.90          3,534   0.9427   +0.0108
    0.05 - 0.95          5,132   0.9437   +0.0117
    0.02 - 0.98          7,523   0.9440   +0.0121

Widening past 0.05–0.95 buys 0.0004 for 47% more transformer time, which is
why the submission uses 0.05–0.95.
