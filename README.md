# Grayscale Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for RGB-to-grayscale conversion on NVIDIA A100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on an A100 via Modal, and logs the result.

## Task

Convert a square RGB image to grayscale using the standard luminance coefficients:

```
Y = 0.2989 R + 0.5870 G + 0.1140 B
```

`custom_kernel` receives an RGB tensor and returns a grayscale tensor:

| Argument | Shape | Dtype |
|---|---|---|
| input | `H × W × 3` | `float32` |
| output | `H × W` | `float32` |

**Benchmark shapes:**

| Size | Image |
|---|---|
| 512 | 512 × 512 × 3 |
| 1024 | 1024 × 1024 × 3 |
| 2048 | 2048 × 2048 × 3 |
| 4096 | 4096 × 4096 × 3 |
| 8192 | 8192 × 8192 × 3 |
| 16384 | 16384 × 16384 × 3 |

Ranked by geometric mean latency across all six shapes (lower is better).

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

Deploy the A100 evaluator (once, before any agent runs):

```bash
uv run modal deploy eval_modal_grayscale.py
```

## Running the agent

```bash
uv run grayscale/agent.py --iterations 20
```

Start from a specific baseline file:

```bash
uv run grayscale/agent.py --baseline grayscale/submission.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run grayscale/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

In tmux (recommended for long runs):

```bash
tmux new-session -d -s agent "set -a && source .env && set +a && uv run grayscale/agent.py --iterations 25 2>&1 | tee grayscale/agent_run.log"
tmux attach -t agent
```

Evaluate a kernel file without running the agent:

```bash
cd grayscale && python run_eval.py submission.py -o results.json
python run_eval.py submission.py -o results.json --mode test   # correctness only
```

## Structure

```
eval_modal_grayscale.py   — deployable Modal A100 evaluator
grayscale/
├── agent.py              — advisor-worker agentic loop
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── submission.py         — the kernel file the worker edits each iteration
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
