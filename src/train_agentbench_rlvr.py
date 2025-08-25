"""Simplified GRPO training script for AgentBench with RLVR.

This script adapts the LifelongAgentBench training script to the
stateless AgentBench environment.  Most of the session bookkeeping
present in the LifelongAgentBench version has been removed because
AgentBench tasks are treated as single-step problems.  The script still
uses the GRPO trainer from TRL and queries the server for verifiable
rewards via the ``/api/branch/complete`` endpoint.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import List

import GPUtil
import requests
import torch
from torch.utils.data import Dataset
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# GPU selection helpers
# ---------------------------------------------------------------------------

def wait_for_healthy_gpus(required: int = 1, max_load: float = 0.30, max_mem: float = 0.30, poll: int = 10) -> List[int]:
    """Return a list of available GPU ids."""
    while True:
        candidates = GPUtil.getAvailable(order="memory", limit=16, maxLoad=max_load, maxMemory=max_mem)
        if len(candidates) >= required:
            return candidates[:required]
        print(f"[GPU picker] Not enough available GPUs; retrying in {poll}s...")
        time.sleep(poll)


def pick_gpus_and_budgets(required: int = 1, safety_frac: float = 0.80):
    """Pick healthy GPUs and return a max_memory dict keyed by local ordinal."""
    abs_ids = wait_for_healthy_gpus(required=required)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, abs_ids))
    print("[GPU picker] Using GPUs:", abs_ids)

    totals_mib = []
    for abs_id in abs_ids:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(abs_id),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        ).strip()
        totals_mib.append(int(out))

    max_memory = {i: f"{int((tot_mib / 1024.0) * safety_frac)}GiB" for i, tot_mib in enumerate(totals_mib)}
    print("[GPU picker] max_memory per local device:", max_memory)
    return max_memory, len(abs_ids)


MAX_MEMORY, NUM_VISIBLE = pick_gpus_and_budgets(required=1, safety_frac=0.80)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.9"

# ---------------------------------------------------------------------------
# Model and tokenizer
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 512

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Avoid left-truncation which might remove instructions
print("📊 BEFORE fixes:")
print("   tok.truncation_side:", getattr(tok, "truncation_side", "NOT_SET"))
tok.truncation_side = "right"
print("📊 AFTER fixes:")
print("   tok.truncation_side:", tok.truncation_side)

print("Loading model...")
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory=MAX_MEMORY,
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

model = base_model

# ---------------------------------------------------------------------------
# Environment helper
# ---------------------------------------------------------------------------


class AgentBenchHTTPEnv:
    """Minimal HTTP environment wrapper for AgentBench."""

    def __init__(self, port: int = 8000) -> None:
        self.base = f"http://localhost:{port}"

    def reset(self):
        return requests.post(self.base + "/api/reset", timeout=30).json()

    def step(self, action: str):
        payload = {"action": action}
        return requests.post(self.base + "/api/step", json=payload, timeout=120).json()

    def snapshot(self):
        return requests.get(self.base + "/api/snapshot", timeout=30).json()

    def complete(self):
        return requests.post(self.base + "/api/complete", timeout=120).json()


env = AgentBenchHTTPEnv(port=8000)


# Extract observation from server responses -------------------------------------------------

def extract_observation(resp: dict) -> str:
    if not isinstance(resp, dict):
        return ""
    session = resp.get("session") or resp.get("info", {}).get("session", {})
    if session and "chat_history" in session:
        hist = session["chat_history"]["value"]
        conversation = []
        for msg in hist:
            role = "User" if msg["role"] == "user" else "Agent"
            conversation.append(f"{role}: {msg['content']}")
        return "\n\n".join(conversation)
    return resp.get("observation", "")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnvDataset(Dataset):
    """Dataset providing the latest environment observation."""

    def __init__(self, env: AgentBenchHTTPEnv):
        self.env = env
        self.resp = self.env.reset()
        self.obs = extract_observation(self.resp)

    def __len__(self) -> int:
        return 100  # arbitrary number of steps

    def __getitem__(self, idx):
        return {"prompt": self.obs}


dataset = EnvDataset(env)

# ---------------------------------------------------------------------------
# Helper functions for text processing
# ---------------------------------------------------------------------------

import re


def sanitize(s: str) -> str:
    t = s.strip()
    t = t.replace("``` sql", "```sql")
    t = t.replace("```sql ", "```sql\n")
    return t


def first_action_block(txt: str) -> str:
    if not txt:
        return txt
    lines = txt.splitlines()
    action_start = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*Action:\s*(Operation|Answer)", line):
            action_start = i
            break
    if action_start == -1:
        return txt
    block = [lines[action_start]]
    for line in lines[action_start + 1 :]:
        if line.strip().startswith("Action:"):
            break
        block.append(line)
    return "\n".join(block)


def is_valid_action(txt: str) -> bool:
    return bool(re.search(r"Action:\s*(Operation|Answer)", txt))


def select_best_idx(cands: List[str], cand_rewards: List[float]) -> int:
    max_reward = max(cand_rewards)
    tied = [i for i, r in enumerate(cand_rewards) if r == max_reward]
    for idx in tied:
        if re.search(r"Action:\s*Operation", cands[idx]):
            return idx
    for idx in tied:
        if is_valid_action(cands[idx]):
            return idx
    return tied[0]


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def ab_reward_function(completions, prompts, **kwargs):
    rewards = []
    n_gen = cfg.num_generations
    for i in range(0, len(completions), n_gen):
        cands = [first_action_block(sanitize(c)) for c in completions[i : i + n_gen]]
        base_snapshot = env.snapshot()
        cand_rewards = []
        for cand in cands:
            payload = {"session": base_snapshot.get("session"), "candidate": cand}
            resp = requests.post(env.base + "/api/branch/complete", json=payload, timeout=120)
            branch = resp.json()
            outcome = branch.get("session", {}).get("evaluation_record", {}).get("outcome")
            cand_rewards.append(1.0 if outcome == "correct" else 0.0)
        best_idx = select_best_idx(cands, cand_rewards)
        best_cand = cands[best_idx]
        commit_out = env.step(best_cand)
        rewards.extend(cand_rewards)
        dataset.resp = commit_out if not commit_out.get("done") else env.reset()
        dataset.obs = extract_observation(dataset.resp)
    return rewards


# ---------------------------------------------------------------------------
# GRPO configuration and trainer
# ---------------------------------------------------------------------------

cfg = GRPOConfig(
    output_dir="grpo_logs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    top_entropy_quantile=0.5,
    num_train_epochs=1,
    logging_steps=1,
    num_generations=4,
    generation_kwargs={
        "max_new_tokens": MAX_TOKENS,
        "do_sample": True,
        "temperature": 1.0,
        "pad_token_id": tok.eos_token_id,
        "use_cache": True,
    },
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[ab_reward_function],
    train_dataset=dataset,
    args=cfg,
)
trainer.processing_class = tok
trainer.tokenizer = tok


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting GRPO training on AgentBench...")
    trainer.train()
    print("Training complete. Saving model...")
    trainer.save_model("trained_model")
    tok.save_pretrained("trained_model")
    print("Model saved to ./trained_model")
