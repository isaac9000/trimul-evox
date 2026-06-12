#!/usr/bin/env python3
"""Count exact LLM API calls in a skydiscover run log."""

import re
import sys
from pathlib import Path
from collections import defaultdict

def count_calls(log_path: Path) -> dict:
    text = log_path.read_text()
    lines = text.splitlines()

    api_calls = [l for l in lines if "HTTP Request: POST" in l and "chat/completions" in l]
    retries   = [l for l in lines if re.search(r"Error on attempt [0-9]+/[0-9]+", l)]
    timeouts  = [l for l in lines if "Timeout on attempt" in l]

    solution_iters  = len(re.findall(r"Iteration \d+:", text))
    meta_evolutions = len(re.findall(r"evolving search strategy", text))
    label_calls     = len(re.findall(r"Generated variation operator labels", text))
    failed_evals    = len(re.findall(r"validity=0\.0000", text))

    return {
        "total_api_calls":    len(api_calls),
        "retry_calls":        len(retries),
        "timeout_calls":      len(timeouts),
        "solution_iterations": solution_iters,
        "meta_evolutions":    meta_evolutions,
        "label_gen_calls":    label_calls,
        "failed_meta_evals":  failed_evals,
    }


def main():
    if len(sys.argv) < 2:
        # Default: find the latest log in the most recent run
        runs = sorted(Path("grayscale/skydiscover_runs").glob("run*/logs/*.log"))
        if not runs:
            print("No run logs found. Pass a log path as argument.")
            sys.exit(1)
        log_path = runs[-1]
    else:
        log_path = Path(sys.argv[1])

    print(f"Log: {log_path}")
    stats = count_calls(log_path)
    print(f"\n{'─'*40}")
    print(f"  Total API calls (exact):   {stats['total_api_calls']}")
    print(f"{'─'*40}")
    print(f"  Solution iterations:       {stats['solution_iterations']}")
    print(f"  Meta search evolutions:    {stats['meta_evolutions']}")
    print(f"  Label generation calls:    {stats['label_gen_calls']}")
    print(f"  Retry calls (failed+retry):{stats['retry_calls']}")
    print(f"  Timeout calls:             {stats['timeout_calls']}")
    print(f"  Failed meta evals:         {stats['failed_meta_evals']}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    main()
