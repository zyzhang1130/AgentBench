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
import json
from pathlib import Path
from typing import List

import GPUtil
import requests
import torch
from torch.utils.data import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# GPU selection helpers
# ---------------------------------------------------------------------------

def wait_for_healthy_gpus(required: int = 1, max_load: float = 0.10, max_mem: float = 0.05, poll: int = 10) -> List[int]:
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
tok.pad_token_id = tok.eos_token_id

# Avoid left-truncation which might remove instructions
print("📊 BEFORE fixes:")
print("   tok.truncation_side:", getattr(tok, "truncation_side", "NOT_SET"))
tok.truncation_side = "right"
print("📊 AFTER fixes:")
print("   tok.truncation_side:", tok.truncation_side)

# Ensure the tokenizer can tokenize a simple prompt
_sample_chat = tok.apply_chat_template(
    [
        {"role": "user", "content": "Hello"},
    ],
    add_generation_prompt=True,
    tokenize=False,
)
assert len(tok(_sample_chat).input_ids) > 0, "Tokenizer produced no tokens"

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory=MAX_MEMORY,
)
base_model.config.use_cache = False
try:
    # Ensure the model returns ModelOutput instead of tuples
    base_model.config.return_dict = True
except Exception:
    pass
try:
    # Also disable cache at generation config level to avoid warnings
    base_model.generation_config.use_cache = False
except Exception:
    pass
base_model.gradient_checkpointing_enable()

model = base_model

from collections import defaultdict
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = defaultdict(bool)
import types
if not hasattr(model, "add_model_tags"):
    model.add_model_tags = types.MethodType(lambda self, tags: None, model)


# ---------------------------------------------------------------------------
# Environment helper
# ---------------------------------------------------------------------------


class AgentBenchHTTPEnv:
    """HTTP environment wrapper using the AgentBench TaskWorker routes."""

    def __init__(self, port: int = 8000) -> None:
        self.base = f"http://localhost:{port}"
        self.session_id = None
        self._next_session_id = 0

    def _new_session_id(self) -> int:
        sid = self._next_session_id
        self._next_session_id += 1
        return sid

    def reset(self):
        new_session_id = self._new_session_id()
        payload = {"index": 0, "session_id": new_session_id}
        resp = requests.post(self.base + "/api/start_sample", json=payload, timeout=30)
        if resp.status_code != 200:
            try:
                preview = resp.text[:400]
            except Exception:
                preview = "<no body>"
            print(f"[DEBUG] start_sample HTTP {resp.status_code}: {preview}")
        data = {}
        try:
            data = resp.json()
        except Exception:
            pass
        # Only update session_id on success and when field exists
        sid = data.get("session_id") if isinstance(data, dict) else None
        if resp.status_code == 200 and sid is not None:
            self.session_id = sid
        else:
            # keep previous session_id if any, otherwise clear
            if self.session_id is None:
                self.session_id = None
        return data

    def step(self, action: str):
        payload = {
            "session_id": self.session_id,
            "agent_response": {"content": action},
        }
        resp = requests.post(self.base + "/api/interact", json=payload, timeout=120)
        if resp.status_code != 200:
            try:
                preview = resp.text[:400]
            except Exception:
                preview = "<no body>"
            print(f"[DEBUG] interact HTTP {resp.status_code}: {preview}")
        try:
            data = resp.json()
        except Exception:
            data = {}
        self.session_id = data.get("session_id", self.session_id)
        return data

    def snapshot(self):
        payload = {"session_id": self.session_id}
        resp = requests.post(self.base + "/api/sample_status", json=payload, timeout=30)
        if resp.status_code != 200:
            try:
                preview = resp.text[:400]
            except Exception:
                preview = "<no body>"
            print(f"[DEBUG] sample_status HTTP {resp.status_code}: {preview}")
        try:
            return resp.json()
        except Exception:
            return {}

    def complete(self):
        if self.session_id is None:
            return {}
        payload = {"session_id": self.session_id}
        resp = requests.post(self.base + "/api/cancel", json=payload, timeout=120)
        self.session_id = None
        return resp.json()


env = AgentBenchHTTPEnv(port=8000)


# Extract observation from server responses -------------------------------------------------

def extract_observation(resp: dict) -> str:
    if not isinstance(resp, dict):
        return ""
    output = resp.get("output", {})
    history = output.get("history") or []
    # One-time debug when history is missing
    if not history:
        global _OBS_DEBUG_PRINTED
        if "_OBS_DEBUG_PRINTED" not in globals():
            _OBS_DEBUG_PRINTED = False
        if not _OBS_DEBUG_PRINTED:
            _OBS_DEBUG_PRINTED = True
            try:
                keys = list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__
                out_keys = list(output.keys()) if isinstance(output, dict) else type(output).__name__
                print(f"[DEBUG] No output.history found. resp keys={keys}; output keys={out_keys}")
                # Short preview for troubleshooting
                try:
                    print("[DEBUG] Response preview:", json.dumps(resp)[:400])
                except Exception:
                    pass
            except Exception:
                pass
    conversation = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        conversation.append(f"{role}: {msg.get('content', '')}")
    return "\n\n".join(conversation)


def build_prompt_from_resp(resp: dict) -> str:
    """Build a chat-style prompt from a server response."""
    obs = extract_observation(resp)
    user_content = obs if obs else "No observation provided. Begin."
    system_content = (
        "Valid actions:\n"
        "- Action: Operation[operation, input]\n"
        "- Action: Answer[answer]\n"
        "Respond with a single action in one of the above formats."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnvDataset(Dataset):
    """Dataset providing the latest environment observation."""

    def __init__(self, env: AgentBenchHTTPEnv):
        self.env = env
        self.resp = self.env.reset()
        # Debug if missing history at initialization
        try:
            out = self.resp.get("output", {}) if isinstance(self.resp, dict) else {}
            if not (isinstance(out, dict) and out.get("history")):
                print(
                    f"[DEBUG] EnvDataset init: missing output.history; resp keys={list(self.resp.keys()) if isinstance(self.resp, dict) else type(self.resp).__name__}"
                )
        except Exception:
            pass
        self.prompt = build_prompt_from_resp(self.resp)

    def __len__(self) -> int:
        return 100  # arbitrary number of steps

    def __getitem__(self, idx):
        return {"prompt": self.prompt}


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

_BRANCH_AVAILABLE = True


def ab_reward_function(completions, prompts, **kwargs):
    rewards = []
    n_gen = cfg.num_generations
    for i in range(0, len(completions), n_gen):
        cands = [first_action_block(sanitize(c)) for c in completions[i : i + n_gen]]
        base_snapshot = env.snapshot()
        cand_rewards = []
        # Determine prompt text for logging (fall back to dataset.prompt)
        try:
            prompt_text = prompts[i // n_gen]
        except Exception:
            prompt_text = dataset.prompt

        global _BRANCH_AVAILABLE
        for j, cand in enumerate(cands):
            outcome = None
            r = 0.0
            branch_status = None
            branch_error = None
            if _BRANCH_AVAILABLE:
                payload = {"session_id": base_snapshot.get("session_id"), "candidate": cand}
                resp = requests.post(env.base + "/api/branch/complete", json=payload, timeout=120)
                branch_status = resp.status_code
                if resp.status_code != 200:
                    try:
                        branch_error = resp.text[:200]
                    except Exception:
                        branch_error = "<no body>"
                    print(f"[DEBUG] branch/complete HTTP {resp.status_code}: {branch_error}")
                    # Disable further branch attempts to avoid spamming
                    _BRANCH_AVAILABLE = False
                try:
                    branch = resp.json()
                    outcome = branch.get("session", {}).get("evaluation_record", {}).get("outcome")
                    r = 1.0 if outcome == "correct" else 0.0
                except Exception:
                    pass
            cand_rewards.append(r)

            # Append detailed branch evaluation record to training log
            try:
                with open(training_log_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "type": "branch_evaluation",
                                "prompt": prompt_text,
                                "completion_idx": j,
                                "completion": cand,
                                "branch_outcome": outcome,
                                "branch_reward": r,
                                "branch_http_status": branch_status if branch_status is not None else "disabled",
                                **({"branch_http_error": branch_error} if branch_error else {}),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            except Exception:
                pass
        best_idx = select_best_idx(cands, cand_rewards)
        best_cand = cands[best_idx]
        commit_out = env.step(best_cand)
        rewards.extend(cand_rewards)

        # Log commit record
        status = commit_out.get("output", {}).get("status") if isinstance(commit_out, dict) else None
        commit_done = isinstance(status, str) and status.lower() != "running"
        try:
            with open(training_log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "commit",
                            "prompt": prompt_text,
                            "best_completion_idx": int(best_idx),
                            "committed_completion": best_cand,
                            "done": commit_done,
                            "commit_status": status,
                            "best_candidate_reward": cand_rewards[best_idx],
                        },
                        indent=2,
                    )
                    + "\n"
                )
        except Exception:
            pass

        # Persist reward history incrementally
        try:
            try:
                with open(reward_log_file, "r") as f:
                    reward_list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                reward_list = []
            reward_list.append(cand_rewards)
            with open(reward_log_file, "w") as f:
                json.dump(reward_list, f, indent=2)
        except Exception:
            pass

        if commit_done:
            dataset.resp = env.reset()
        else:
            dataset.resp = commit_out
        dataset.prompt = build_prompt_from_resp(dataset.resp)
    return rewards


# ---------------------------------------------------------------------------
# GRPO configuration and trainer
# ---------------------------------------------------------------------------

cfg = GRPOConfig(
    output_dir="grpo_logs",  # will be redirected to timestamped folder under repo root
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    top_entropy_quantile=0.5,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no",  # disable checkpoint saves
    num_generations=4,
    max_prompt_length=None,
    generation_kwargs={
        "max_new_tokens": MAX_TOKENS,
        "do_sample": True,
        "temperature": 1.0,
        "pad_token_id": tok.eos_token_id,
        "use_cache": False,
    },
)

# ---------------------------------------------------------------------------
# Logging files (reward history + per-step records)
# ---------------------------------------------------------------------------
from datetime import datetime

# Create logs under AgentBench/grpo_logs/<timestamp>
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Robust repo-root detection: supports running from root, src/, or scripts/
here = Path(__file__).resolve().parent
REPO_ROOT = here if (here / "src").exists() else here.parent
LOGS_BASE = REPO_ROOT / "grpo_logs"
OUTPUT_DIR = LOGS_BASE / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure Trainer writes its artifacts inside the same timestamped folder
cfg.output_dir = str(OUTPUT_DIR)

training_log_file = OUTPUT_DIR / f"training_history_{timestamp}.jsonl"
reward_log_file = OUTPUT_DIR / f"reward_history_{timestamp}.json"

# Initialize files
try:
    with open(reward_log_file, "w") as f:
        json.dump([], f)
    with open(training_log_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "config",
                    "model_name": MODEL_NAME,
                    "learning_rate": cfg.learning_rate,
                    "num_generations": cfg.num_generations,
                    "timestamp": timestamp,
                },
                indent=2,
            )
            + "\n"
        )
except Exception:
    pass

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[ab_reward_function],
    train_dataset=dataset,
    args=cfg,
    processing_class=tok,
)
trainer.processing_class = tok
trainer.tokenizer = tok


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting GRPO training on AgentBench...")
    # Run training and capture metrics
    result = trainer.train()

    # Persist trainer state and metrics to output_dir (grpo_logs)
    trainer.save_state()  # writes trainer_state.json
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)  # writes train_results.json
    print("Training complete. Skipping model checkpoint save per request.")
