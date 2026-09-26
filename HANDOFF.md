# Amazon ML Challenge 2026 — Handoff

Everything needed to continue cold. Repo `shikkoustic/Amazon-ML`, branch
`claude/magical-bell-viflxk`. Read §5 before changing anything: it lists the
mistakes that already cost real score.

---

## 1. THE PROBLEM

**Business Entity Resolution.** Three independent sources list the same
businesses with different IDs and noisy text. For every Source-1 entity, find
all matching records in Source 2 and Source 3. An entity may match zero, one
or many.

**Fields, all files, tab-separated:** `entity_id` (the `S1-`/`S2-`/`S3-`
prefix identifies the source), `business_name`, `business_address`,
`country`. There is no separate source column.

**Read every file with `sep="\t"`.** Without it pandas silently returns one
column, because addresses and ID lists both contain commas.

### Scale

|          | train     | test      |
|----------|-----------|-----------|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |

Exhaustive comparison on train is 22.7 trillion pairs, hence blocking.

### The metric

`F_0.5 = 1.25·P·R / (0.25·P + R)`, computed **per Source-1 entity then
averaged**. Three consequences drive every design decision:

- **Precision counts twice.** Two correct matches plus one wrong scores
  0.714; finding one and stopping scores 0.833. Greed is punished.
- **Singletons are all-or-nothing.** An entity with no true matches scores
  1.0 for an empty prediction and 0.0 for any guess. They are 5.58% of
  entities.
- **Partial recall is cheap.** Two of three true matches with no false
  positives still scores 0.909.

Optimal threshold measured at **0.70** for the current model. It is far above
0.5 because of the first point.

### Submission

**During the challenge:** upload `matching_results.tsv` only. That is the
only file scored on the leaderboard.

**Final package:** one zip with `output/matching_results.tsv`,
`output/candidate_pairs.tsv`, `code/business_entity_resolution/` (runnable,
with README and requirements.txt) and a filled `Documentation_template.md`.

Rejection rules, all verified against our output: every Source-1 entity must
have exactly one row; matched IDs must be `S2-`/`S3-` only and must exist in
the test set; no duplicate IDs within a list; no duplicate entity rows.
Validate with `student_resource/utils/validate_submission.py` before every
upload — it caught nothing on ours, but it is free.

**Rules:** no external data lookup of any kind (entity-resolution APIs,
government registries, geocoding, internet augmentation) — immediate
disqualification. Final model MIT or Apache-2.0 and under 8B parameters.
LightGBM and MiniLM both qualify.

**Late rule change, watch this:** `candidate_pairs.tsv` counts toward the
final ranking, and *a smaller candidate set per entity ranks higher*. Ours is
100/entity. Worth considering before the final package.

---

## 2. WHERE THINGS STAND

Leaderboard, in order:

    0.907   v1   combo char-trigram candidates, 69-feature model
    0.908   v2   France normalisation fix
    0.912   v3p  France+India re-scored, matched candidates, 79 features

Leaders are at **0.988–0.99**. A teammate's other team reports 0.925.

Held-out validation (5,000 entities, seed 42, 75/25 by entity):

    0.9112  starting point
    0.9143  word-level blocking + region features
    0.9164  full normalisation rebuild
    0.9256  cross-source agreement
    0.9307  street-number repair           <- +0.0051
    0.9319  name containment + ordinals
    0.9427  cross-encoder cascade          <- +0.0108

Validation runs about 0.017 above the board. That gap was a train/inference
blocking mismatch and should be smaller now that candidates match.

---

## 3. THE DECOMPOSITION THAT MATTERS

Do not optimise anything before reading this. Measured on validation:

    perfect                                           1.0000
    ceiling (perfect matcher on our candidates)       0.9886
    no false positives                                0.9458
    ACTUAL                                            0.9256

    matcher rejects a true pair already in candidates   57.6% of loss
    matcher adds a false positive                       27.1%
    blocking never proposed the pair                    15.3%

**Blocking is a sixth of the problem.** A day went into it for +0.000.

Ranking the rejected true pairs by recoverable macro-F:

    name identical, address differs      31%
    both partly agree                    20%
    address empty on one side            19%
    both differ substantially            13%
    address agrees, name replaced        10%
    name agrees, address differs          7%

The first class is *also* 38% of the false positives. Same-name-different-
address is simultaneously the largest recall loss and the largest precision
loss: the model cannot tell a damaged address from a different branch.

**The score is concentrated in a tiny band.** The model is near-certain about
almost everything: pairs below 0.05 are 0.1% true, pairs above 0.95 are 99.2%
true. Every mistake sits in 0.42% of pairs. Resolving 0.20–0.80 perfectly is
worth 0.9319 → 0.9569; resolving 0.05–0.95 is worth 0.9753. At test scale
that is 1.7M pairs to re-score, not 173M, and the first-stage score names
them.

---

## 4. WHAT EXISTS

### Data

    data/raw/student_resource/dataset/{train,test}/    raw TSVs
    data/interim/{train,test}_source{1,2,3}.parquet    normalised cache
    data/interim/matcher_v2.txt                        current model, 79 features
    data/interim/specialist.txt                        band specialist (failed, see §6)

The parquet cache is normalised text plus blocking keys. Rebuild with
`scripts/build_cache.py` after any normalisation change — it skips files that
already exist, so move them aside first.

### Candidates

    kaggle_outputs/wordblock-v3/candidate_pairs_combo_s{2,3}.tsv   TEST, matched
    data/interim/wordcand/candidate_pairs_combo_train.tsv          20k entities
    data/interim/wordcand/candidate_pairs_combo_big.tsv            200k entities

The test candidates are word-level TF-IDF with the repaired normalisation,
100/entity, 1,732,544 rows each, all countries. **These match what the model
was trained on. Use these.**

### Pipeline

    src/normalize.py         country-aware normalisation
    src/variants.py          GLOBAL + SHORT variant tables (see §5)
    src/features.py          79 pair features
    src/scoring.py           macro F_0.5, verified against the worked example
    scripts/build_cache.py   raw -> normalised parquet
    scripts/word_candidates.py   blocking, resumable, --analyzer word|char_wb
    scripts/union_pairs.py       candidates + labels -> parquet
    scripts/build_relative.py    pair features + within-entity relative features
    scripts/train_v2.py          trains the matcher, sweeps the threshold
    scripts/predict_test.py      scores test candidates -> matching_results.tsv
    scripts/evaluate.py          scores any predictions file, stdlib only
    kaggle_kernel/train_crossencoder.py   cross-encoder on Kaggle GPU

`predict_test.py` is resumable: each shard writes its result and `--resume`
skips finished ones. It shards by country first (safe — matched records always
share a country) then by entity hash, so an entity's candidates stay together,
which the relative features require.

### Infrastructure

Kaggle CLI is configured for `shikkoustic` (`~/.kaggle/`). Kernels are pushed
by API rather than pasted — a 20KB paste silently truncated once and died on a
syntax error mid-table.

    shikkoustic/wordblock-v3     word-level test blocking, COMPLETE
    shikkoustic/ce-matcher       cross-encoder, GPU, COMPLETE
    shikkoustic/amazon-ml-ce-pairs   dataset: cross-encoder training pairs

Kaggle's *script* images are leaner than its *notebook* images — `unidecode`
is absent from one. The kernel installs what it needs rather than dying on
the import.

No AWS credentials in this container; EC2 was a dead end (the instance was a
t3.micro the free plan could not resize).

---

## 5. TRAPS THAT ALREADY COST SCORE

**Hard-coded abbreviations destroyed a country.** A US-state table mapped
`de`→delaware and `la`→louisiana, which are the commonest words in a French
address: `rue de la Paix` became `rue delaware louisiana paix`, 119,000 times
across French addresses. France is 15% of test and absent from training, so
validation never saw it. Fixed by a rule, not a list: keys of three
characters or fewer apply only to the countries they were derived from.
Three entries also merged distinct words — court/connecticut, mount/montana,
saint/street — because `ct`, `mt` and `st` are each two different English
words.

**Train and inference must use the same blocking configuration.** The
strongest features are relative: rank, gap-to-best and z-score *within the
entity's candidate set*. Score against a differently-built candidate pool and
every one of them shifts. Done by accident in v1; cost about 0.017.

**Validation covered only US and India.** Those are the only countries in
training. France is 15% of test and was never measured until the board told
us. Any future change should be checked for country-specific damage.

**Downsampling negatives inflates the score.** Reporting 0.9468 that way was
wrong; honest was 0.9024. It also breaks calibration.

**Sorting matters.** The candidate list came from iterating a Python set, and
string hashing is randomised per process, so two identical runs disagreed on
~3% of entities. Relative features break ties by position. Sort the list.

**Forking after loading a LightGBM model deadlocks.** Loading a Booster
initialises OpenMP; the forked children hang with the parent at 100% CPU and
workers at 0%. The feature count is read from the model file header instead,
so the parent never imports lightgbm before forking.

**Two workers beat four.** Each worker holds the country's text map (India is
~5M entities) and copy-on-write stops helping once they touch those pages.
Four workers ran at under half a core each and the pass slowed from an
estimated 2h toward 5h. Two workers: memory 13GB → 5GB, and *faster*.

**Background jobs die when the container suspends.** Anything long must
checkpoint. `word_candidates.py` and `predict_test.py` both do;
`build_relative.py` does not and has been killed mid-run.

---

## 6. TESTED AND REJECTED — do not redo

    raising blocking recall, three separate times   +0.000 to +0.002 each
    training data 7x larger                         recall flat at 0.856
    exclusivity constraint (one-to-many)            zero conflicts at any threshold
    top-k and rank-aware decision rules             all below a flat threshold
    expected-F_0.5 decision rule with calibration   -0.0005 against flat
    scaling the variant table 286 -> 6,042          touches ~5% of pairs
    character trigrams vs word tokens               worse recall, 6.7x slower
    a second GBM on the uncertain band              0.7787 in-band vs 0.8043
    "rescue" rule (take top-1 when nothing clears)  -0.005, wrecks singletons
    Union-Find transitive closure                   unbounded chaining

The specialist result is the important one: a GBM on the same features does
*worse* than the general model inside the band. The features are spent, which
is the whole argument for the cross-encoder.

---

## 7. THE CROSS-ENCODER CASCADE

The one lever with real magnitude left.

**Why.** The band the first stage cannot decide is 0.42% of pairs and holds
all the remaining loss. Hand-built features cannot separate it (§6). A model
reading the raw text can: MiniLM reached 0.8907 in-band AUC against the first
stage's 0.8043.

**Measured.** Blending the cross-encoder 60/40 with the first stage inside the
band: validation 0.9319 → **0.9427**, +0.0108. Weight 1.0 gives +0.0095, so
the first stage still carries signal where it is unsure.

**Known limit.** That prototype trained 6 minutes on 25,868 in-band pairs.
A perfect band resolution is worth +0.043, so it captured about a quarter.
`candidate_pairs_combo_big.tsv` (200,000 entities, 20M pairs) exists to give
roughly 250,000 in-band pairs — 10× the training data — for a larger model.
That is the open work.

**To apply it to test you need first-stage scores for every test pair**, to
know which fall in the band. `predict_test.py --dump-scores` does this but
requires `--workers 1`. The current v3 run does not dump them, so a second
pass is needed. Plan for that before starting a scoring run.

**Licensing.** MiniLM is Apache-2.0 and 22M parameters, inside the rules. The
pretrained weights are a general language model, not a lookup of any business,
and only the provided training data is used to fine-tune.

---

## 8. IN FLIGHT AND NEXT

**Running now:** `predict_test.py` producing `output/matching_results_v3.tsv`
with matched candidates, 79 features, threshold 0.70, two workers. 40/48
shards. France and India complete; US in progress.

**Already submitted from it:** `output/matching_results_v3partial.tsv` — v3
rows for France and India, v2 rows for US — scored 0.912.

**Next, in order:**

1. When US finishes, write the full v3 and submit. Expect ~0.9145.
2. Second scoring pass with `--dump-scores` to get first-stage scores for all
   test pairs, so the band can be identified.
3. Build features on `candidate_pairs_combo_big.tsv`, extract in-band pairs,
   retrain the cross-encoder larger on Kaggle GPU, re-measure the cascade.
4. Apply the cascade to the test band, submit.
5. Final package: both TSVs, `code/business_entity_resolution/`, and a filled
   `Documentation_template.md`. **The methodology document is not written
   yet** and is a hard requirement.

**Realistic expectation.** Full v3 ~0.9145, plus the current cascade ~0.925,
plus a scaled cross-encoder perhaps 0.93–0.94. 0.95 has not been shown to be
reachable from this architecture; every other lever has been measured and
exhausted.

---

## 9. HOW TO MAKE PROGRESS HERE

The method that worked, three times out of three:

1. **Decompose the loss.** Blocking, matcher recall, matcher precision,
   singletons. Fix the biggest bucket, not the most interesting one.
2. **Read the actual failures.** Pull the true pairs the model rejects most
   confidently and look at them. Rank the classes by recoverable macro-F.
3. **Fix the systematic class with a general rule**, then measure on held-out
   before believing it.

Guessing at features from first principles returned +0.002 every time.
Reading failures returned +0.005 (street numbers) and +0.009
(transliteration). The cascade came from decomposing the score distribution
rather than the feature space.

One more caution. A signal that separates well *in isolation* is not
necessarily new information. Cross-source agreement measured 0.628 vs 0.182
standalone and returned +0.0013, because it restated what the model already
had. Test marginally, against everything else the model sees.
