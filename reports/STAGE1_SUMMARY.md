# Stage 1 — Blocking / Candidate Generation

Status: **method validated and locked. Test-set generation is a long batch
job that has not yet been run to completion** (see Open items).

Everything below is measured against the training ground truth. Where a
number is an estimate or an assumption, it says so.

---

## 1. What the data actually is

| | Train | Test |
|---|---|---|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |

Exhaustive comparison on train is 2.2M × 10.3M ≈ **22.7 trillion pairs**.

**Countries.** Train is US 60% / India 40%. Test is India 46.8% / US 38.3% /
**France 15.0%**, a country absent from training.

**Scripts.** Source 1 is 100% Latin. Sources 2 and 3 carry **nine Indic
scripts** — Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali,
Malayalam, Oriya, Gurmukhi — at ~9.2% of names in train and **~11% in test**.

**Hygiene.** No empty names, no duplicate ids, country always present.
~3.3% of addresses are empty. The literal string `<NULL>` appears inside
address fields and must be treated as missing.

---

## 2. Facts verified exhaustively (not sampled)

Checked over all **7,638,365** true pairs, zero exceptions:

| Property | Result | Why it matters |
|---|---|---|
| Matched records share a country | **0 violations** | Partitioning by country is free |
| Referential integrity both ways | 0 dangling, 0 missing | Ground truth is trustworthy |
| No duplicate ids within a row | 0 | — |
| All matched ids are S2-/S3- | 0 bad | — |
| **Matches are one-to-many** | **7,638,365 slots = 7,638,365 distinct ids** | See below |

### The exclusivity finding

No Source 2 or Source 3 record is *ever* claimed by two Source 1 entities.
The task is a **one-to-many assignment problem**, not independent pair
classification.

If Stage 2 predicts the same S2 record for two different S1 entities, at
least one is provably wrong without any model. Resolving those conflicts in
favour of the higher-confidence claim is free precision — and F_0.5 weights
precision twice as heavily as recall. **This is a Stage 2 action item.**

### Match-count distribution

Singletons 5.58%; mean 3.46 matches; max **11** total, **5** from S2, **6**
from S3. An all-empty submission scores **0.0558** macro F_0.5.

---

## 3. The metric, and why pair recall misleads

The score is F_0.5 computed **per Source 1 entity, then averaged**. Two
consequences that drive every design decision:

- **Precision counts double.** For an entity with 2 true matches, finding
  both plus one wrong (0.71) scores *worse* than finding only one and
  stopping (0.83). Greed is punished.
- **Partial recall is cheap.** Missing one of three matches still scores
  0.909 for that entity, and the 5.58% singletons score 1.0 with no
  candidates at all.

So every configuration below reports **max macro F_0.5** — the best score
achievable assuming a perfect Stage 2 that predicts exactly the true
matches present among the candidates. That is the real cap on the score;
pair recall only proxies it, and proxies it badly.

---

## 4. Reachability ceiling vs realised recall

**Ceiling** (can a signal see the pair at all, given exhaustive search),
measured on 439,565 true pairs, stable across 5 seeds (σ = 0.002%):

| Signal | India | US |
|---|---|---|
| Raw name | 81.4% | 98.5% |
| Transliterated name | 95.6% | 98.5% |
| + phonetic folding | 98.2% | 98.7% |
| Address | 96.1% | 95.2% |
| **name ∪ address** | **99.998%** | **100.000%** |

Raw names fail on India because Source 1 is Latin and Sources 2/3 are not.
**Transliteration is load-bearing, not decoration.**

**But the ceiling assumes exhaustive comparison.** Under actual top-K
ranking over the full same-country index, name-only collapses from a 91.5%
ceiling to **64.8% realised** at K=25. The gap is entirely ranking: a true
match is visible but buried under businesses with similar names.

---

## 5. The combined field

Indexing `name + " " + address` as a single string was the largest single
improvement. Separate passes cannot express that a record agreeing on
*both* fields outranks one agreeing on either alone — which is precisely
what top-K was discarding.

Combined field alone reaches **93.4% at 20 candidates/entity**, beating the
name∪address union at the same K while producing *half* the candidates.

### Pass contribution (ablation at K=50)

| Pass | Recall contributed | Cost (cand/entity) |
|---|---|---|
| **combined field** | **+1.115%** | 57.9 |
| address | +0.445% | 65.4 |
| name | +0.365% | 70.3 |
| rare-token index | +0.065% | 3.5 |
| sorted-token key | +0.020% | 13.2 |

The sorted-token key earns its place least and can be dropped.

---

## 6. Throughput

Query rate is governed by **n-grams per document**, not by K or by the
similarity floor — varying either left the rate unchanged at ~33 q/s,
because the cost is in the sparse dot products, not top-K selection.

Pruning high-document-frequency n-grams is the real lever:

| max_df | n-grams/doc | q/s | F_0.5 ceiling @K=25 |
|---|---|---|---|
| 1.0 | 40.3 | 35 | 0.9890 |
| **0.05** | ~25 | 98 | **0.9885** |
| 0.005 | 6.3 | 1055 | 0.9216 |

**0.05 is near-free** — 0.0005 of ceiling for a ~2.3× speedup. 0.005 is
dramatic but costs seven points of ceiling.

---

## 7. Chosen configuration

```
partition   by country (dynamic; never separates a true match)
normalise   transliterate -> collapse doubled letters -> strip legal
            suffixes -> phonetic fold   (ordering matters, see §9)
signals     combined name+address, plus name, plus address
ngrams      char_wb (3,3)
max_df      0.05
K           25 per source per signal, applied to S2 and S3 INDEPENDENTLY
```

### Operating points

| Config | recall | **max macro F_0.5** | cand/entity | est. test runtime |
|---|---|---|---|---|
| combo only, K=25 | 94.6% | 0.9798 | 50 | ~5 h |
| combo only, K=50 | 95.5% | 0.9835 | 100 | ~5 h |
| **combo+name+addr, K=25** | **96.8%** | **0.9885** | **128** | **~15 h** |
| combo+name+addr, K=50 | 97.5% | 0.9907 | 260 | ~15 h |
| all passes, K=100 | 98.2% | 0.9937 | 523 | ~30 h |

Going from 20 to 523 candidates/entity (26×) buys **+0.018** of ceiling.
Since Stage 2 will fall well short of its ceiling regardless, the smaller
set is worth more than the last fraction of recall — it makes Stage 2
tractable and fast to iterate on.

---

## 8. France

France has no labels, so it cannot be measured directly. Three pieces of
indirect evidence:

1. **Config choice transfers.** Configurations rank almost identically by
   US and by India recall — 86.8% pairwise ordering agreement, identical
   top four. A configuration chosen on labelled countries is not
   country-specific.
2. **The hard case is India, not France.** The failure mode we feared —
   an unreadable writing system — is an *India* problem that transliteration
   solves. France is Latin-script, and the US column is healthier than the
   India column on every name signal.
3. **Candidate counts are normal.** On real test data France gets **127.3
   candidates/entity** vs India 124.7 and US 129.0, with zero empty
   normalised keys. Nothing is silently failing.

Country is used only as a partition key, never one-hot encoded, so a new
label requires no code change.

---

## 9. Bugs found and fixed

- **Pooled top-K across sources.** S2 and S3 triples were truncated
  together, letting one source absorb a query's entire budget. Since 80% of
  entities match records in *both*, this silently capped recall.
- **Asymmetric normalisation.** Suffixes were stripped before
  transliteration artefacts were collapsed, so the Latin record lost
  "private limited" while its Indic counterpart kept "praaivett limittedd".
  Reordering lifted cross-script similarity from 0.000 to 0.16–0.27.
- **Normalisation could empty a name.** "SARL" reduced to the empty string,
  which matches nothing. Now falls back to the pre-removal form.
- **Truncated cache files** looked complete to the next run.
- **Absent configs reported as 0% recall** rather than skipped.

---

## 10. Where the test run has to happen

Measured throughput is **~90 queries/sec** against a 2.3M-document index.
The test set needs 1,732,544 queries per signal per source, so:

| Config | query-passes | compute |
|---|---|---|
| combo only, K=50 | 3.5M | **~11 h** |
| combo+name+addr, K=25 | 10.4M | ~32 h |

**This container cannot do it.** It suspends between turns, yielding about
seven usable minutes per turn; eleven hours would take roughly ninety
turns and thirty-two hours nearly three hundred.

**Run it on Kaggle instead.** A Kaggle session runs 12 hours uninterrupted,
which fits the combo-only configuration in a single session.
`scripts/kaggle_generate_candidates.py` is self-contained for that purpose
— its inlined normaliser is verified to produce byte-identical output to
`src/normalize.py` on cross-script and edge-case inputs. It checkpoints per
country and source, so a session cut short resumes.

```bash
!pip -q install unidecode sparse_dot_topn
!python kaggle_generate_candidates.py     --data /kaggle/input/<slug>/student_resource/dataset --out /kaggle/working
```

Given the compute constraint, **combo-only at K=50 is the recommended test
configuration**: 0.9835 ceiling against 0.9885 for all three signals, at
one third the cost. Stage 2 will not approach either ceiling, so the
0.005 difference is not worth tripling the runtime.

`scripts/generate_candidates.py` remains the local equivalent, chunk-
checkpointed, and has accumulated partial progress under
`data/interim/candgen/`.

## 11. Open items for Stage 2
1. **Exploit exclusivity** (§2) as a post-processing assignment step.
2. **Tune the decision threshold for F_0.5, not F1 — and make it
   rank-aware.** `src/scoring.py` derives the break-even confidence for
   adding one more match, and it is not a single number. It depends on how
   many matches the entity already has:

   | entity truth size | matches already predicted | add only above |
   |---|---|---|
   | 5 | 1 | 0.444 |
   | 3 | 1 | 0.571 |
   | 2 | 1 | 0.667 |
   | 3 | 2 | 0.727 |
   | any | all of them | 0.800 |

   So be liberal on an entity's first match and progressively stricter
   after. The 0.800 asymptote means adding to an already-complete
   prediction is almost never worth it. A flat cut-off — and certainly the
   0.5 a classifier defaults to — leaves points on the table.
3. **Build a singleton detector** — 5.58% of entities, each worth a full
   1.0, and any false positive on one costs the entire point.
4. **Score the un-phonetically-folded forms.** The folding trades precision
   for recall, which is right for blocking and wrong for scoring.

## 12. Untested

- char n-gram range (2,4) — ~3× the cost of (3,3); the arm was dropped
  before a clean comparison, so (3,3) is chosen on cost, not on measured
  superiority
- dense/multilingual embeddings as an additional union pass
- bounded one-hop S2↔S3 bridge expansion
- numeric-address-only blocking key

---

## Reproducing

```bash
scripts/ensure_data.sh                  # restore data if container recycled
python3 scripts/build_cache.py          # normalise once -> parquet
python3 scripts/profile_data.py         # dataset profile
python3 scripts/verify_assumptions.py   # exhaustive structural checks
python3 scripts/stage1_experiments.py --per-country 10000 --ngrams 33 \
        --max-df 0.05 --tag d05         # the sweep
python3 scripts/generate_candidates.py --split test   # production output
```

All scripts checkpoint and resume; the container suspends between turns.
