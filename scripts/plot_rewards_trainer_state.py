#!/usr/bin/env python3
"""Plot per-step rewards from trainer_state.json (TRL GRPOTrainer log_history).

Usage examples:
- python AgentBench/scripts/plot_rewards_trainer_state.py \
    --input /home/zy1130/AgentBench/grpo_logs/20250828_181729/trainer_state.json
- python AgentBench/scripts/plot_rewards_trainer_state.py --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def moving_average(values: List[float], window: int) -> List[float]:
    """Trailing moving average with same-length output."""
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


def load_trainer_rewards(path: Path) -> Tuple[List[int], List[float]]:
    """Return (steps, rewards) from trainer_state.json log_history."""
    with path.open("r") as f:
        data = json.load(f)

    logs = data.get("log_history", [])
    steps, rewards = [], []
    for idx, rec in enumerate(logs):
        if not isinstance(rec, dict):
            continue
        if "reward" not in rec:
            continue
        step = int(rec.get("step", idx + 1))
        r = float(rec["reward"])
        steps.append(step)
        rewards.append(r)

    # Ensure sorted by step in case out-of-order entries exist
    pairs = sorted(zip(steps, rewards), key=lambda x: x[0])
    if not pairs:
        return [], []
    s, r = zip(*pairs)
    return list(s), list(r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-step reward from trainer_state.json."
    )
    parser.add_argument(
        "--input",
        default="/home/zy1130/AgentBench/grpo_logs/20250829_003421/reward_history_20250829_003421.json",
        help="Path to trainer_state.json (under AgentBench/grpo_logs/<timestamp>/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: next to input, suffix _trainer_state.png)",
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
        help="Overlay the raw per-step rewards when smoothing",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    steps, rewards = load_trainer_rewards(inp)
    if not rewards:
        print("No reward entries found in trainer_state.json.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install via: pip install matplotlib")
        raise

    # Apply smoothing if requested
    if args.smooth_window and args.smooth_window > 1:
        plot_rewards = moving_average(rewards, args.smooth_window)
        label = f"Reward (SMA w={args.smooth_window})"
    else:
        plot_rewards = rewards
        label = "Reward"

    plt.figure(figsize=(10, 5))
    plt.plot(steps, plot_rewards, marker="o", linewidth=1.5, label=label)

    if args.smooth_window and args.smooth_window > 1 and args.overlay_raw:
        plt.plot(
            steps,
            rewards,
            color="gray",
            alpha=0.35,
            linestyle="--",
            linewidth=1,
            label="Raw reward",
        )

    overall = sum(rewards) / len(rewards)
    plt.axhline(
        overall,
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"Overall mean (raw) = {overall:.3f}",
    )

    title_suffix = (
        f" — SMA w={args.smooth_window}" if args.smooth_window and args.smooth_window > 1 else ""
    )
    plt.title(f"Trainer Reward per Step{title_suffix} (n={len(rewards)})")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = inp.with_name(inp.stem + "_trainer_state.png")

    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

