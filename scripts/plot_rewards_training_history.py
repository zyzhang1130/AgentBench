#!/usr/bin/env python3
"""Plot averaged env_reward per batch from training logs.

Supports two input formats under AgentBench/grpo_logs:
- training_history_*.jsonl: stream of JSON objects including
  {"type": "candidate_evaluation", "env_reward": ...} and
  {"type": "commit", "best_candidate_reward": ...}.
  The script computes the mean env_reward for each block of candidate_evaluation
  events preceding a commit and (optionally) overlays best_candidate_reward.
- reward_history_*.json: {"task_name": ..., "entries": [[...], [...], ...]}
  where each inner list contains candidate rewards for a batch; the batch mean
  is the average of that list. No best series is available in this format.

Usage examples:
- python AgentBench/scripts/plot_rewards_training_history.py \
    --input /home/zy1130/AgentBench/grpo_logs/20250828_181729/training_history_20250828_181729.jsonl
- python AgentBench/scripts/plot_rewards_training_history.py \
    --input /home/zy1130/AgentBench/grpo_logs/20250829_101125/reward_history_20250829_101125.json
- python AgentBench/scripts/plot_rewards_training_history.py --show --overlay-best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import math


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(float(x) for x in xs)) / float(len(xs))


def moving_average(values: List[float], window: int) -> List[float]:
    n = len(values)
    if window <= 1 or n == 0:
        return [float(v) for v in values]
    window = min(window, n)
    pref = [0.0]
    for v in values:
        pref.append(pref[-1] + float(v))
    out: List[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        s = pref[i + 1] - pref[start]
        out.append(s / count)
    return out


def load_batch_means_and_bests(path: Path) -> Tuple[List[float], List[float], Optional[float]]:
    """Return (batch_means, bests) from training_history JSONL.

    - batch_means[i] is the average of `env_reward` values between commits i-1 and i.
    - bests[i] is the `best_candidate_reward` value at commit i.
    """
    batch_means: List[float] = []
    bests: List[float] = []
    cur: List[float] = []
    top_entropy_quantile: Optional[float] = None

    # Try parsing as training_history_*.jsonl (stream of JSON objects).
    try:
        content = path.read_text()
    except FileNotFoundError:
        return [], []

    decoder = json.JSONDecoder()
    i = 0
    n = len(content)
    while i < n:
        while i < n and content[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            i += 1
            continue
        i = end

        if not isinstance(obj, dict):
            continue
        t = obj.get("type")
        # Capture config fields if present
        if t == "config":
            teq = obj.get("top_entropy_quantile")
            if isinstance(teq, (int, float)):
                top_entropy_quantile = float(teq)
            # continue scanning other records
        if t == "candidate_evaluation":
            r = obj.get("env_reward")
            if isinstance(r, (int, float)):
                cur.append(float(r))
        elif t == "commit":
            batch_means.append(mean(cur))
            cur = []
            br = obj.get("best_candidate_reward")
            bests.append(float(br) if isinstance(br, (int, float)) else float("nan"))
        else:
            # ignore other record types
            pass

    if batch_means:
        return batch_means, bests, top_entropy_quantile

    # Fallback: try parsing as reward_history_*.json
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
            entries = obj["entries"]
            # entries is List[List[number]]; compute per-batch mean
            rh_means: List[float] = []
            for row in entries:
                if isinstance(row, list) and row:
                    rh_means.append(mean([float(x) for x in row if isinstance(x, (int, float))]))
                elif isinstance(row, list):
                    rh_means.append(0.0)
            teq = obj.get("top_entropy_quantile")
            if isinstance(teq, (int, float)):
                top_entropy_quantile = float(teq)
            return rh_means, [], top_entropy_quantile
    except Exception:
        pass

    # No recognizable format
    return [], [], None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot averaged env_reward per batch from training logs (training_history_*.jsonl or reward_history_*.json)")
    parser.add_argument(
        "--input",
        default="/home/zy1130/AgentBench/grpo_logs/20250830_023136/reward_history_20250830_023136.json",
        help="Path to training_history_*.jsonl or reward_history_*.json (under AgentBench/grpo_logs/<timestamp>/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: next to input, suffix _trainhist_avg.png)",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Trailing moving-average window (0/1 disables smoothing)",
    )
    parser.add_argument(
        "--overlay-best",
        action="store_true",
        help="Overlay the best_candidate_reward series at commit boundaries",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    batch_means, bests, top_entropy_quantile = load_batch_means_and_bests(inp)
    if not batch_means:
        print("No batch means could be computed. Ensure the input is a training_history_*.jsonl or reward_history_*.json file.")
        return

    try:
        import matplotlib.pyplot as plt
        import math
        import numpy as np
    except Exception:
        print("matplotlib (and numpy) are required. Install via: pip install matplotlib numpy")
        raise

    if args.smooth_window and args.smooth_window > 1:
        y = moving_average(batch_means, args.smooth_window)
        label = f"Batch mean env_reward (SMA w={args.smooth_window})"
    else:
        y = batch_means
        label = "Batch mean env_reward"

    # Append top_entropy_quantile info to legend label if available
    if top_entropy_quantile is not None:
        try:
            label += f"; q={top_entropy_quantile:.3f}"
        except Exception:
            label += f"; q={top_entropy_quantile}"

    x = list(range(1, len(y) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", linewidth=1.5, label=label)

    if args.overlay_best and bests:
        # Align bests length to number of batches (they should match if every batch ended with commit)
        n = min(len(bests), len(x))
        xs = x[:n]
        ys = bests[:n]
        # Replace NaNs for plotting transparency
        ys_clean = [v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None for v in ys]
        plt.plot(xs, ys_clean, color="green", alpha=0.6, linestyle=":", linewidth=1.2, label="best_candidate_reward")

    overall = mean(batch_means)
    plt.axhline(overall, color="orange", linestyle="--", linewidth=1, label=f"Overall mean (raw) = {overall:.3f}")
    title_suffix = f" — SMA w={args.smooth_window}" if args.smooth_window and args.smooth_window > 1 else ""
    plt.title(f"Avg env_reward per candidate batch{title_suffix} (n={len(batch_means)})")
    plt.xlabel("Batch index (per commit)")
    plt.ylabel("Mean env_reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = inp.with_name(inp.stem + "_trainhist_avg.png")

    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
