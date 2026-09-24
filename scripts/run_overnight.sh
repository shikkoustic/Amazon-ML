#!/usr/bin/env bash
# Unattended Stage 1 pipeline. Each step gates the next.
set -uo pipefail
cd /home/user/Amazon-ML
R=reports; mkdir -p $R

echo "=== $(date -u) waiting for normalisation cache ==="
while pgrep -f build_cache.py >/dev/null; do sleep 20; done
ls -lh data/interim/

echo; echo "=== $(date -u) SMOKE TEST (subsampled index - code check only) ==="
timeout 1800 python3 scripts/stage1_experiments.py \
  --per-country 300 --index-sample 40000 --ngrams 33 \
  --out smoke.json > $R/stage1_smoke.log 2>&1
if [ $? -ne 0 ]; then
  echo "SMOKE TEST FAILED - aborting"; tail -30 $R/stage1_smoke.log; exit 1
fi
echo "smoke test passed"; tail -5 $R/stage1_smoke.log

echo; echo "=== $(date -u) REAL RUN (full index, 10k queries/country) ==="
timeout 21600 python3 scripts/stage1_experiments.py \
  --per-country 10000 --ngrams 33,24 \
  --out stage1_experiments.json > $R/stage1_main.log 2>&1
echo "exit=$?"
echo "=== $(date -u) done ==="
