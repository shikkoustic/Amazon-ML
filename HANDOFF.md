# Amazon ML Challenge 2026 — Full Handoff

Everything needed to continue cold. Repo: `shikkoustic/Amazon-ML`,
branch `claude/magical-bell-viflxk`.

---

## 1. THE PROBLEM

**Business Entity Resolution.** Three independent sources list the same
businesses with different IDs and noisy text. For every Source-1 entity,
find all matching records in Source 2 and Source 3. An entity may match
zero, one, or many.

**Fields (all files, tab-separated):** `entity_id` (prefix S1-/S2-/S3-
identifies the source), `business_name`, `business_address`, `country`.

**Files:** `dataset/train/train_source{1,2,3}.tsv`,
`dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`.
Ground truth: `source1_entity_id` + comma-separated `matched_entity_ids`
(empty if none). Read with `sep="\t"` — without it pandas silently returns
one column.

**Noise:** abbreviations (Pvt/Private, Rd/Road), legal-suffix mismatch,
typos, word-order swaps, DBA/trade names replacing the name entirely,
landmark addresses ("Near SBI ATM"), missing pincodes, `<NULL>` as literal
text in address fields.

### Scale
| | train | test |
|---|---|---|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |

Exhaustive comparison on train = 22.7 trillion pairs.

### Metric — macro F_0.5
`F_0.5 = 1.25 P R / (0.25 P + R)`, computed **per Source-1 entity then
averaged**. Consequences that drive every decision:
- Precision counts **2x**. Two true matches plus one wrong scores 0.714 —
  worse than finding one and stopping (0.833). Greed is penalised.
- Partial recall is cheap: 2 of 3 matches still scores 0.909.
- **Singletons score 1.0 for an empty prediction, 0.0 for any guess.**
- Break-even confidence for adding another match runs 0.44 (first of five)
  to 0.80 (already complete). Optimal flat threshold measured = **0.67**.

### Submission
Two files in `output/`:
1. `matching_results.tsv` — `source1_entity_id \t matched_entity_ids`.
   **The only file scored.** One row per test S1 entity (all 1,732,544),
   empty for singletons, no duplicate IDs, only S2-/S3- IDs that exist in
   the test set, no self-matches.
2. `candidate_pairs.tsv` — same format, the blocking output. Not scored but
   audited; matches must be a subset of candidates.

Final zip also needs `code/business_entity_resolution/` (runnable, README,
requirements.txt) and a filled `Documentation_template.md`.
Validate with `student_resource/utils/validate_submission.py` before every
upload — it checks format only, never quality.

**Rules:** no external data lookup of any kind (ER APIs, government
registries, geocoding, internet augmentation) — instant disqualification.
Final model must be MIT/Apache-2.0 and under 8B params. LightGBM is fine.

---

## 2. VERIFIED FACTS (exhaustive, all 7,638,365 true pairs, zero exceptions)

- **Matched records always share a country** → partitioning is free, and it
  generalises to unseen countries because the rule is "same country", never
  a hard-coded list.
- **Matches are strictly one-to-many**: 7,638,365 slots = 7,638,365 distinct
  IDs. No S2/S3 record is ever claimed by two S1 entities.
- Singletons **5.58%**. Mean 3.46 matches. Max 11 (max 5 from S2, 6 from S3).
- Source 1 is 100% Latin. S2/S3 carry **nine Indic scripts** (Devanagari,
  Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya, Gurmukhi) —
  9.2% of names in train, **11% in test**.
- No empty names, no duplicate IDs, country always present, 3.3% empty
  addresses.
- Test country mix: India 46.8%, US 38.3%, **France 15.0%** (absent from
  training — its recall is inferred, never measured).

---

## 3. CURRENT STATE

| | |
|---|---|
| Blocking recall (combo K=50, char-trigram) | 95.5% |
| F_0.5 ceiling that implies | 0.9835 |
| Matcher, measured honestly | **0.9145** |
| Leaderboard leaders | **0.980** |
| All-empty baseline | 0.056 |

**The gap is blocking recall, not the model.** Our ceiling (0.9835) barely
exceeds their score.

### THE KEY FINDING — not yet deployed
**99.93% of true pairs share at least one exact whole token.**

| blocking | India recall | speed | full test run |
|---|---|---|---|
| char_wb (3,3) — what Kaggle is running | 91.3% @K=50 | 33 q/s | 14.4 h |
| **word (1,1)** | **94.5% @K=100** | **107 q/s** | **4.5 h** |

Char n-grams carry 31 non-zeros/doc vs 11 for words, so the sparse product
touches 3x more index. Choosing them for typo tolerance was wrong: the noise
rarely destroys whole tokens. **This is the highest-value fix available.**

---

## 4. WHAT EXISTS IN THE REPO

| file | purpose |
|---|---|
| `src/normalize.py` | transliterate → collapse doubles → strip suffixes (order matters, see §6) |
| `src/features.py` | 27 pair features |
| `src/scoring.py` | macro F_0.5, verified against organisers' worked example (0.714) |
| `scripts/build_cache.py` | normalise all records once → parquet |
| `scripts/profile_data.py`, `verify_assumptions.py` | the §2 measurements |
| `scripts/stage1_experiments.py` | blocking sweep harness |
| `scripts/generate_candidates.py` | blocking for a split (local) |
| `scripts/aws_blocking.py` | **word-level blocking, many-core — use this** |
| `kaggle_kernel/run_blocking.py` | char-trigram version running on Kaggle |
| `scripts/export_training_pairs.py` | candidates + labels for Stage 2 |
| `scripts/build_relative.py` | adds within-entity relative features |
| `scripts/train_v2.py` | trains the matcher, sweeps threshold |
| `reports/STAGE1_SUMMARY.md` | full Stage 1 write-up |

### ⚠️ WHAT DOES NOT EXIST YET
**There is no test-inference script.** Nothing takes test candidates →
features → model → `matching_results.tsv`. This must be written. It needs
to stream (1.7M entities x ~100 candidates = ~170M pairs) rather than load
everything into memory.

---

## 5. IN FLIGHT

**Kaggle — 6 notebooks, char-trigram, started ~11:00 25 Sep, ~8h each.**
Your account: `AmznChlng` (combo s2), `Amzn-ML-2` (combo s3), plus addr s2
and addr s3. Teammate's account: name s2, name s3.
Outputs: `candidate_pairs_{signal}_s{2|3}.tsv` in each notebook's Output tab.

**EC2** — instance `i-0ccb9aeb7d7ab959c`, **c7i.8xlarge, 32 vCPU**,
public IP **98.80.138.181**, us-east-1, **Amazon Linux 2023** (user
`ec2-user`, package manager `dnf`). ~$1.43/h — **TERMINATE WHEN DONE**.

Setup (already run, or re-run if needed):
```bash
sudo dnf install -y -q python3-pip unzip
pip3 install --user -q numpy scipy scikit-learn sparse_dot_topn unidecode gdown
mkdir -p ~/work && cd ~/work
python3 -m gdown "1xrbNNUwuVk_GwfLJ8cw-3ZSmzQ9FPCZg" -O dataset.zip
unzip -q dataset.zip && rm dataset.zip
```
Then copy `scripts/aws_blocking.py` to `~/work/run.py` and:
```bash
cd ~/work
DATA=/home/ec2-user/work/student_resource/dataset OUT=/home/ec2-user/work/output \
  THREADS=32 nohup python3 -u run.py > run.log 2>&1 &
tail -f run.log
```
Expect ~600 q/s. Combined field ~50 min, all three signals ~2.5 h.
Retrieve results: `cd ~/work/output && python3 -m http.server 8000`
then browse `http://98.80.138.181:8000` (port 8000 already open).

---

## 6. THINGS THAT WILL BITE YOU

- **Normalisation order matters.** Strip legal suffixes AFTER collapsing
  doubled letters, or the Latin record loses "private limited" while its
  Indic counterpart keeps "praaivett limittedd" — asymmetric, and it
  destroys exactly the pairs transliteration exists to rescue. Fixing the
  order lifted cross-script similarity from 0.000 to 0.16–0.27.
- **Guard against normalisation emptying a string.** "SARL" → "" matches
  nothing.
- **Top-K must be applied per source.** Pooling S2 and S3 lets one source
  absorb the whole budget; 80% of entities match records in both.
- **Downsampling negatives inflates the score.** Reporting 0.9468 that way
  was wrong; honest was 0.9024. It also breaks calibration — train at 16.8%
  positives against a true 2.6% and a score of 0.6 means a 16% chance of
  being right.
- **Relative features need the FULL candidate set.** Rank within a sampled
  set is a different quantity.
- Stage 2 must be trained on the **same blocking config** that inference
  uses, since the strongest features are within-candidate-set z-scores.

---

## 7. TESTED AND REJECTED — don't redo

| idea | result |
|---|---|
| Exploiting one-to-many constraint | real but **non-binding**, 0 conflicts at any useful threshold |
| Rank-aware thresholds | +0.0014, not worth complexity |
| Stacking on out-of-fold scores | 0.9210 vs 0.9217, no gain |
| "Rescue" (take top-1 when nothing clears threshold) | −0.005, wrecks singletons |
| Short-circuit routing on exact match | loses recall; entities average 3.46 matches |
| Intersecting two retrieval methods | backwards — blocking takes the union |
| Union-Find transitive closure | unbounded chaining, worst failure mode under F_0.5 |
| max_df=0.005 pruning | 30x faster but costs 7 points of ceiling |

**What DID work:** combined name+address field (+1.115% recall), relative
within-entity features (**+0.0146 ablated**), max_df=0.05 (free 2x speedup),
threshold 0.67.

---

## 8. PLAN — in order

1. **Run word-level blocking on EC2** (`scripts/aws_blocking.py`). Biggest
   available win. ~50 min for the combined field at 32 cores.
2. Collect Kaggle outputs as they land; union with the EC2 output. Union
   only raises recall.
3. Measure each config's recall on train, pick the best.
4. Rebuild Stage 2 training pairs on the **chosen** config
   (`export_training_pairs.py` → `build_relative.py` → `train_v2.py`).
5. **Write the test-inference script** (§4) → `matching_results.tsv`.
6. Run `validate_submission.py`, then submit and read the real score.
7. Iterate against leaderboard feedback.

### Targets
- Realistic with word-level blocking: **0.92–0.94**
- Needed to be competitive: **0.97+**
- Current projection without the fix: 0.9145

Getting *any* leaderboard reading is urgent — every number so far is from
our own validation split, and we have had zero external feedback.
