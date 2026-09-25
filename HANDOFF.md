# Amazon ML Challenge 2026 — Handoff

Paste this into a fresh Claude session to continue.

## Task
Business Entity Resolution. For each Source-1 business, find matching records
in Source 2 and 3. Metric: **macro F_0.5** (per-entity, then averaged).
Deadline: ~25 Sep 2026 + 2 days. Leaderboard leaders at **0.980**.

## Repo
`shikkoustic/Amazon-ML`, branch `claude/magical-bell-viflxk`.
Full detail in `reports/STAGE1_SUMMARY.md` and `reports/pipeline_record.html`.

## Verified facts (exhaustive, all 7,638,365 true pairs, zero exceptions)
- Matched records **always share a country** → partition by country is free
- Matches are **strictly one-to-many** (no S2/S3 record claimed by two S1 entities)
- Singletons: **5.58%**; max matches per entity 11 (max 5 from S2, 6 from S3)
- Source 1 is 100% Latin; S2/S3 carry **nine Indic scripts** (~9% train, ~11% test)
- Test = India 46.8% / US 38.3% / **France 15%** (France absent from training)

## Current numbers
| | |
|---|---|
| Blocking recall (combo K=50) | 95.5% |
| Blocking ceiling that implies | 0.9835 |
| Matcher, measured honestly | **0.9145** |
| Leaderboard leaders | 0.980 |

## KEY FINDING (acted on, not yet deployed)
**99.93% of true pairs share at least one exact whole token.**
Word-level TF-IDF beats char-trigrams on BOTH axes:

| blocking | India recall | speed | full test run |
|---|---|---|---|
| char_wb (3,3) — what Kaggle is running | 91.3% @K=50 | 33 q/s | 14.4 h |
| **word (1,1)** | **94.5% @K=100** | **107 q/s** | **4.5 h** |

Char n-grams were the wrong choice: 31 non-zeros/doc vs 11, so the sparse
product touches 3x more index. This is why competitors are fast AND accurate.

## In flight
**Kaggle** (6 notebooks, ~8h, char-trigram, started ~11:00 25 Sep):
combo/name/addr x source2/source3. Outputs named
`candidate_pairs_{signal}_s{2|3}.tsv`. Download and send when done.

**EC2** — instance `i-0ccb9aeb7d7ab959c`, **c7i.8xlarge, 32 vCPU**,
public IP **98.80.138.181**, us-east-1. Terminal open via EC2 Instance Connect.
⚠️ **It is Amazon Linux 2023, not Ubuntu** — user is `ec2-user`, package
manager is `dnf`. Costs ~$1.43/h — TERMINATE when done.

### Setup commands for Amazon Linux (corrected)
```bash
sudo dnf install -y -q python3-pip unzip
pip3 install --user -q numpy scipy scikit-learn sparse_dot_topn unidecode gdown
mkdir -p ~/work && cd ~/work
python3 -m gdown "1xrbNNUwuVk_GwfLJ8cw-3ZSmzQ9FPCZg" -O dataset.zip
unzip -q dataset.zip && rm dataset.zip
find . -name "test_source1.tsv"; nproc
```
Then create `~/work/run.py` from `scripts/aws_blocking.py` in the repo and run:
```bash
cd ~/work
DATA=/home/ec2-user/work/student_resource/dataset OUT=/home/ec2-user/work/output \
  THREADS=32 nohup python3 -u run.py > run.log 2>&1 &
tail -f run.log
```
Expect ~600 q/s. Combined-field output in ~50 min, all three signals ~2.5 h.

### Getting results off the box
```bash
cd ~/work/output && python3 -m http.server 8000
```
Then browse to `http://98.80.138.181:8000` (port 8000 is already open).

## Stage 2 (done, works)
- `src/features.py` — 27 pair features. Best: `addr_idf_overlap` AUC 0.936
- **Relative within-entity features are the biggest win** (+0.0146 ablated).
  `addr_idf_overlap_z` AUC 0.946, 7x the model gain of anything else.
- `src/scoring.py` — metric verified against organisers' worked example (0.714)
- Optimal threshold **0.67** (derived from metric, confirmed empirically)
- Scripts: `build_relative.py` → `train_v2.py`

## Tested and REJECTED (don't redo these)
- One-to-many constraint as a filter — real but **non-binding**, 0 conflicts
- Rank-aware thresholds — +0.0014, not worth it
- Stacking on OOF scores — 0.9210 vs 0.9217, no gain
- "Rescue" rule (take top-1 when nothing clears threshold) — costs 0.005
- Short-circuit routing on exact match — loses recall
- Intersecting retrieval methods — backwards, union is correct

## Next steps, in order
1. Run word-level blocking on EC2 (the big win, 4.5h → ~50 min at 32 cores)
2. Collect Kaggle outputs, union everything, measure best config on train
3. Rebuild Stage 2 features on the **matched** blocking config, retrain
4. Inference on test → `matching_results.tsv` → submit
5. Validate with `student_resource/utils/validate_submission.py` before submitting

## Known gap
We are at 0.914 projected; leaders at 0.980. Our blocking ceiling (0.9835)
is barely above their score, so blocking recall is the binding constraint.
Word-level blocking is the identified fix and is not yet deployed to test.
