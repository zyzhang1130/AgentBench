#!/usr/bin/env python3
"""Plot averaged env_reward per batch for multiple logs on one figure.

Supports two input formats under AgentBench/grpo_logs:
- training_history_*.jsonl: stream of JSON objects including
  {"type": "candidate_evaluation", "env_reward": ...} and
  {"type": "commit", "best_candidate_reward": ...}.
  For each block of candidate_evaluation events preceding a commit, the script
  computes the mean env_reward and (optionally) overlays best_candidate_reward.
- reward_history_*.json: {"task_name": ..., "entries": [[...], [...], ...],
  "top_entropy_quantile": <float>?}. Each inner list contains candidate
  rewards for a batch; the batch mean is the average of that list.

Usage examples:
- python AgentBench/scripts/plot_rewards_compare.py \
    AgentBench/grpo_logs/20250829_101125/reward_history_20250829_101125.json \
    AgentBench/grpo_logs/20250830_023136/reward_history_20250830_023136.json
- python AgentBench/scripts/plot_rewards_compare.py \
    AgentBench/grpo_logs/20250829_101125/training_history_20250829_101125.jsonl \
    --overlay-best --smooth-window 10 --show
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import glob
from typing import List, Tuple, Optional


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
    """Return (batch_means, bests, top_entropy_quantile) from a log file.

    - batch_means[i] is the average of `env_reward` values between commits i-1 and i.
    - bests[i] is the `best_candidate_reward` value at commit i (may be empty for
      reward_history files).
    - top_entropy_quantile if present (float), else None.
    """
    batch_means: List[float] = []
    bests: List[float] = []
    cur: List[float] = []
    top_entropy_quantile: Optional[float] = None

    try:
        content = path.read_text()
    except FileNotFoundError:
        return [], [], None

    # First try to parse as training_history_*.jsonl
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
        if t == "config":
            teq = obj.get("top_entropy_quantile")
            if isinstance(teq, (int, float)):
                top_entropy_quantile = float(teq)
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

    # Fallback: reward_history_*.json
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
            entries = obj["entries"]
            teq = obj.get("top_entropy_quantile")
            if isinstance(teq, (int, float)):
                top_entropy_quantile = float(teq)
            rh_means: List[float] = []
            for row in entries:
                if isinstance(row, list) and row:
                    rh_means.append(mean([float(x) for x in row if isinstance(x, (int, float))]))
                elif isinstance(row, list):
                    rh_means.append(0.0)
            return rh_means, [], top_entropy_quantile
    except Exception:
        pass

    return [], [], None


# Optional: set default inputs here if you prefer editing the script
# Example:
# DEFAULT_INPUTS = [
#     "AgentBench/grpo_logs/20250829_101125/reward_history_20250829_101125.json",
#     "AgentBench/grpo_logs/20250830_023136/reward_history_20250830_023136.json",
# ]
DEFAULT_INPUTS: List[str] = ['/home/zy1130/AgentBench/grpo_logs/20250829_003421/reward_history_20250829_003421.json',
                             '/home/zy1130/AgentBench/grpo_logs/20250829_101125/reward_history_20250829_101125.json',
                             '/home/zy1130/AgentBench/grpo_logs/20250830_023136/reward_history_20250830_023136.json',
                             '/home/zy1130/AgentBench/grpo_logs/20250903_171524/reward_history_20250903_171524.json']


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Plot averaged env_reward per batch for multiple logs (training_history_*.jsonl or reward_history_*.json)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Paths to one or more training_history_*.jsonl or reward_history_*.json files",
    )
    parser.add_argument(
        "--inputs-file",
        default=None,
        help="Path to a text file with one input path or glob per line (blank lines and # comments ignored)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: next to first input, suffix _compare_avg.png)",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Trailing moving-average window (0/1 disables smoothing)",
    )
    parser.add_argument(
        "--overlay-best",
        action="store_true",
        help="Overlay the best_candidate_reward series at commit boundaries for training_history inputs",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install via: pip install matplotlib")
        raise

    plt.figure(figsize=(11, 6))

    # Build input list: file, CLI args, or DEFAULT_INPUTS
    input_list: List[str] = []
    if args.inputs_file:
        try:
            lines = Path(args.inputs_file).read_text().splitlines()
            for ln in lines:
                s = ln.strip()
                if not s or s.startswith('#'):
                    continue
                # Expand globs (may yield zero, handle later)
                matched = glob.glob(s)
                if matched:
                    input_list.extend(matched)
                else:
                    input_list.append(s)
        except Exception as e:
            print(f"Warning: could not read --inputs-file: {e}")
    input_list.extend(args.inputs)
    if not input_list and DEFAULT_INPUTS:
        input_list.extend(DEFAULT_INPUTS)

    any_series = False
    handles = []
    labels = []

    for idx, path_str in enumerate(input_list):
        inp = Path(path_str)
        batch_means, bests, teq = load_batch_means_and_bests(inp)
        if not batch_means:
            print(f"Warning: {inp} produced no batch means; skipping.")
            continue

        any_series = True
        if args.smooth_window and args.smooth_window > 1:
            y = moving_average(batch_means, args.smooth_window)
            base_label = f"{inp.stem} (SMA w={args.smooth_window})"
        else:
            y = batch_means
            base_label = inp.stem
        if teq is not None:
            try:
                base_label += f"; q={teq:.3f}"
            except Exception:
                base_label += f"; q={teq}"

        x = list(range(1, len(y) + 1))
        line = plt.plot(x, y, marker="o", linewidth=1.6, label=base_label)[0]
        handles.append(line)
        labels.append(base_label)

        if args.overlay_best and bests:
            n = min(len(bests), len(x))
            xs = x[:n]
            ys = bests[:n]
            ys_clean = [v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None for v in ys]
            plt.plot(
                xs,
                ys_clean,
                color=line.get_color(),
                alpha=0.5,
                linestyle=":",
                linewidth=1.2,
                label=f"{inp.stem} best_candidate_reward",
            )

    if not any_series:
        print("No series to plot. Provide inputs as positional args, via --inputs-file, or set DEFAULT_INPUTS in the script.")
        return

    plt.title("Avg env_reward per candidate batch (multi)")
    plt.xlabel("Batch index (per commit)")
    plt.ylabel("Mean env_reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        # Choose an output path based on first resolved input (if any)
        first_str = input_list[0]
        first = Path(first_str)
        out_path = first.with_name(first.stem + "__compare_avg.png")

    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
