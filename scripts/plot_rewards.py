#!/usr/bin/env python3
"""Plot per-step average rewards from reward_history JSON.

Supports both formats:
- Old: [[...], [...], ...]
- New: {"task_name": str, "entries": [[...], ...]}

Usage examples:
- python scripts/plot_rewards.py \
    --input /home/zy1130/AgentBench/src/grpo_logs/reward_history_20250828_123601.json
- python scripts/plot_rewards.py --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union


def load_reward_entries(path: Union[str, Path]) -> Tuple[str, List[List[float]]]:
    """Load reward entries and task name from a reward_history file.

    Returns (task_name, entries) where entries is a list of lists of numbers.
    """
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)

    task_name = "unknown-task"
    if isinstance(data, dict):
        task_name = str(data.get("task_name", task_name))
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("Invalid reward file: 'entries' is not a list")
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Invalid reward file: expected dict or list at top level")
    return task_name, entries


def mean_of_list(xs: List[float]) -> float:
    nums = [x for x in xs if isinstance(x, (int, float))]
    return float(sum(nums)) / float(len(nums)) if nums else 0.0


def moving_average(values: List[float], window: int) -> List[float]:
    """Trailing moving average with same-length output.

    For index i, averages values[max(0, i-window+1): i+1].
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average candidate rewards per step.")
    parser.add_argument(
        "--input",
        default="/home/zy1130/AgentBench/grpo_logs/20250829_003421/reward_history_20250829_003421.json",
        help="Path to reward_history_*.json (under AgentBench/grpo_logs/<timestamp>/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: next to input, suffix _avg.png)",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Trailing moving-average window (0/1 disables smoothing)",
    )
    parser.add_argument(
        "--overlay-raw",
        action="store_true",
        help="Overlay the raw per-step means when smoothing",
    )
    args = parser.parse_args()

    task_name, entries = load_reward_entries(args.input)
    step_means = [mean_of_list(step) for step in entries]

    if not step_means:
        print("No data found to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install via: pip install matplotlib")
        raise

    # Apply smoothing if requested
    if args.smooth_window and args.smooth_window > 1:
        plot_means = moving_average(step_means, args.smooth_window)
        label = f"Per-step mean (SMA w={args.smooth_window})"
    else:
        plot_means = step_means
        label = "Per-step mean reward"

    xs = list(range(1, len(plot_means) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(xs, plot_means, marker="o", linewidth=1.5, label=label)

    if args.smooth_window and args.smooth_window > 1 and args.overlay_raw:
        plt.plot(
            list(range(1, len(step_means) + 1)),
            step_means,
            color="gray",
            alpha=0.35,
            linestyle="--",
            linewidth=1,
            label="Raw per-step mean",
        )

    overall = mean_of_list(step_means)
    plt.axhline(overall, color="orange", linestyle="--", linewidth=1, label=f"Overall mean (raw) = {overall:.3f}")
    title_suffix = f" — SMA w={args.smooth_window}" if args.smooth_window and args.smooth_window > 1 else ""
    plt.title(f"Average Candidate Reward per Step — {task_name}{title_suffix} (n={len(step_means)})")
    plt.xlabel("Training step")
    plt.ylabel("Average reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        inp = Path(args.input)
        out_path = inp.with_name(inp.stem + "_avg.png")

    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
