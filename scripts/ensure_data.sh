#!/usr/bin/env bash
# Re-fetch the dataset if the container was reclaimed. Idempotent.
set -euo pipefail
D=/home/user/Amazon-ML/data/raw
GDRIVE_ID=1xrbNNUwuVk_GwfLJ8cw-3ZSmzQ9FPCZg

if [ -f "$D/student_resource/dataset/train/train_ground_truth.tsv" ]; then
  echo "data present"; exit 0
fi
echo "data missing - re-downloading"
mkdir -p "$D" && cd "$D"
[ -f dataset.zip ] || python3 -m gdown "$GDRIVE_ID" -O dataset.zip
unzip -q -o dataset.zip -x "__MACOSX/*"
rm -rf __MACOSX; find student_resource -name ".DS_Store" -delete
echo "data restored"
