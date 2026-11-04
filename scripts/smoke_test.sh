#!/usr/bin/env bash
# End-to-end check on synthetic data (CPU, ~1 minute). Not a result — just proves the pipeline runs.
set -euo pipefail
T=${1:-/tmp/floodshift_smoke}
rm -rf "$T"
python scripts/make_synthetic.py --out "$T/data" --size 128
python scripts/train.py --data "$T/data/sen1floods11" --out "$T/runs" --members 2 --epochs 4 --val-every 2 \
    --batch 8 --crop 64 --weights none --encoder resnet18 --workers 0
python scripts/evaluate.py --data "$T/data/sen1floods11" --runs "$T/runs" --out "$T/results" --mc-samples 3
python scripts/pakistan_case_study.py predict --runs "$T/runs" --data "$T/data/pakistan" \
    --out "$T/results/pakistan" --sen1floods11-metrics "$T/results/metrics.json" --tile 128 --overlap 32
echo "smoke test passed: $T/results"
