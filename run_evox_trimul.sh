#!/usr/bin/env bash
# Run SkyDiscover/EvoX on the TriMul problem for 25 iterations.
#
# Prerequisites:
#   1. Deploy the Modal evaluator first (once):
#      uv run modal deploy eval_modal_trimul.py
#   2. Ensure ANTHROPIC_API_KEY and MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are in .env

set -uo pipefail

RUN_BASE="trimul/skydiscover_runs"
PYTHON=".venv/bin/python"

set -a
source .env
set +a

mkdir -p "$RUN_BASE"

echo ""
echo "=== Measuring baseline (starting_point.py) ==="

BASELINE_JSON=$(mktemp /tmp/baseline_trimul_XXXXXX.json)

if ! $PYTHON trimul/run_eval.py trimul/starting_point.py \
        -o "$BASELINE_JSON" --mode leaderboard; then
    echo "  Baseline eval failed — continuing anyway"
    rm -f "$BASELINE_JSON"
else
    PARSE=$($PYTHON -c "
import json, re
md = json.load(open('$BASELINE_JSON'))
gm = re.search(r'Geometric mean: ⏱ ([\d.]+)', md)
gpu = re.search(r'GPU: \`([^\`]+)\`', md)
print(gm.group(1) if gm else '0')
print(gpu.group(1) if gpu else 'unknown')
")
    rm -f "$BASELINE_JSON"

    GEOMEAN=$(echo "$PARSE" | sed -n '1p')
    GPU_NAME=$(echo "$PARSE" | sed -n '2p')

    echo "  GPU         : $GPU_NAME"
    echo "  Geomean     : ${GEOMEAN} µs (baseline)"
fi

RUN_NUM=$(ls -d "$RUN_BASE"/run* 2>/dev/null | wc -l)
RUN_NUM=$((RUN_NUM + 1))
RUN_OUT="$RUN_BASE/run$RUN_NUM"
mkdir -p "$RUN_OUT"

echo ""
echo "=== Starting 25-iteration EvoX run (run${RUN_NUM}) ==="

tmux kill-session -t evox-trimul 2>/dev/null || true
tmux new-session -d -s evox-trimul

tmux send-keys -t evox-trimul \
    "cd /workspace/trimul-evox && set -a && source .env && set +a && \
.venv/bin/skydiscover-run \
  trimul/starting_point.py \
  trimul/skydiscover_evaluator.py \
  --config trimul/skydiscover_config.yaml \
  --search evox \
  --iterations 25 \
  --output $RUN_OUT \
2>&1 | tee ${RUN_OUT}.log" Enter

echo ""
echo "  tmux session : evox-trimul"
echo "  Output dir   : $RUN_OUT"
echo "  Log          : ${RUN_OUT}.log"
echo ""
echo "  Monitor : tmux attach -t evox-trimul"
