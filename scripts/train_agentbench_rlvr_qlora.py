"""Simplified GRPO training script for AgentBench with RLVR.

This script communicates with the AgentBench TaskController (not the
TaskWorker directly) to manage sessions and interactions. It no longer
uses any non-existent "branch" endpoints and only records rewards that
are actually produced by the environment in its TaskOutput result (if
present), avoiding proxy rewards.
"""

import os
import time
import subprocess
import json
from pathlib import Path
from typing import List, Optional, Union

import GPUtil
import requests
import torch
from torch.utils.data import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# GPU selection helpers
# ---------------------------------------------------------------------------

def wait_for_healthy_gpus(required: int = 1, max_load: float = 0.10, max_mem: float = 0.01, poll: int = 10) -> List[int]:
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


MAX_MEMORY, NUM_VISIBLE = pick_gpus_and_budgets(required=3, safety_frac=0.90)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.9"

# ---------------------------------------------------------------------------
# Model and tokenizer
# ---------------------------------------------------------------------------
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
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
USE_QLORA = os.getenv("QLORA", "1").lower() in ("1", "true", "yes", "y")
if USE_QLORA:
    print("Using QLoRA (4-bit) base via bitsandbytes…")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
else:
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
if USE_QLORA:
    # Cast norms to fp32, enable grad inputs, etc., for k-bit training
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

# Configure and apply LoRA adapters
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))

# Common target modules for LLaMA/Qwen-style architectures
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

print(
    f"Enabling {'QLoRA' if USE_QLORA else 'LoRA'}: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}; targets={LORA_TARGET_MODULES}"
)
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

model = get_peft_model(base_model, lora_config)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
try:
    model.print_trainable_parameters()
except Exception:
    pass

from collections import defaultdict
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = defaultdict(bool)
import types
if not hasattr(model, "add_model_tags"):
    model.add_model_tags = types.MethodType(lambda self, tags: None, model)


# ---------------------------------------------------------------------------
# Environment helper (via TaskController)
# ---------------------------------------------------------------------------


class AgentBenchHTTPEnv:
    """HTTP environment wrapper using the AgentBench TaskController routes."""

    def __init__(self, controller_port: int = 5000, task_name: str = "dbbench-std", index: int = 0) -> None:
        # Controller exposes endpoints under /api
        self.base = f"http://localhost:{controller_port}"
        self.task_name = task_name
        self.index = index
        self.session_id: Optional[int] = None

    def reset(self):
        # Ask controller to start a new sample for the task; controller allocates session_id
        payload = {"name": self.task_name, "index": int(self.index)}
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
        # Update controller-assigned session_id
        sid = data.get("session_id") if isinstance(data, dict) else None
        if resp.status_code == 200 and sid is not None:
            try:
                self.session_id = int(sid)
            except Exception:
                self.session_id = sid
        return data

    def wait_for_worker(self, timeout_sec: float = 30.0, poll_sec: float = 1.0) -> bool:
        """Wait until the desired task has at least one ALIVE worker.

        Returns True if available; False if timed out.
        """
        import time as _time
        deadline = _time.time() + timeout_sec
        while _time.time() < deadline:
            try:
                r = requests.get(self.base + "/api/list_workers", timeout=5)
                info = r.json() if r.status_code == 200 else {}
                task = info.get(self.task_name)
                if not task:
                    _time.sleep(poll_sec)
                    continue
                workers = task.get("workers", {})
                for _, w in workers.items():
                    # WorkerStatus.ALIVE == 0
                    if w.get("status") == 0:
                        return True
            except Exception:
                pass
            _time.sleep(poll_sec)
        return False

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

    def complete(self):
        # Cancel current session via controller (optional cleanup)
        if self.session_id is None:
            return {}
        payload = {"session_id": self.session_id}
        try:
            resp = requests.post(self.base + "/api/cancel", json=payload, timeout=120)
            return resp.json()
        except Exception:
            return {}
        finally:
            self.session_id = None


# Task selection (default to dbbench-std). Override with env var TASK_NAME and START_INDEX.
TASK_NAME = os.getenv("TASK_NAME", "dbbench-std")
START_INDEX = int(os.getenv("START_INDEX", "0"))

# Epoch-style: cycle over N prompts per epoch (default 1 = legacy behavior)
N_PROMPTS = int(os.getenv("N_PROMPTS", "10"))

env = AgentBenchHTTPEnv(controller_port=5000, task_name=TASK_NAME, index=START_INDEX)


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
        "You are interacting with an environment. Respond with a single action.\n"
        "Examples:\n"
        "- DBBench: 'Action: Operation' with a SQL block, or 'Action: Answer' with Final Answer.\n"
        "- WebShop: 'Thought:' then 'Action:' lines like search[...] or click[...]."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate_action_from_prompt(prompt_text: str) -> str:
    """Use the current model to generate the next action text from a prompt."""
    import torch
    with torch.no_grad():
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=cfg.generation_kwargs.get("max_new_tokens", MAX_TOKENS),
            do_sample=cfg.generation_kwargs.get("do_sample", True),
            temperature=cfg.generation_kwargs.get("temperature", 1.0),
            pad_token_id=cfg.generation_kwargs.get("pad_token_id", tok.eos_token_id),
            use_cache=False,
        )
        out = tok.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return first_action_block(sanitize(out))


def rollout_to_termination(eval_env: "AgentBenchHTTPEnv", first_action: str, max_steps: int = 6) -> float:
    """Send first_action, then iteratively generate and send actions until done or step limit.

    Returns the terminal reward if available, else 0.0.
    """
    try:
        resp = eval_env.step(first_action)
        out = resp.get("output", {}) if isinstance(resp, dict) else {}
        status = out.get("status")
        done = isinstance(status, str) and status.lower() != "running"
        if done:
            return _parse_env_reward(out.get("result"))
        steps = 0
        while steps < max_steps:
            # Build prompt from the latest observation/history and generate the next action
            prompt_text = build_prompt_from_resp(resp)
            action = generate_action_from_prompt(prompt_text)
            resp = eval_env.step(action)
            out = resp.get("output", {}) if isinstance(resp, dict) else {}
            status = out.get("status")
            done = isinstance(status, str) and status.lower() != "running"
            if done:
                return _parse_env_reward(out.get("result"))
            steps += 1
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnvDataset(Dataset):
    """Dataset providing a single, evolving environment observation (legacy mode)."""

    def __init__(self, env: AgentBenchHTTPEnv):
        self.env = env
        # Wait for task worker to be registered and alive on controller
        if not self.env.wait_for_worker(timeout_sec=60.0, poll_sec=1.0):
            print("[WARN] No ALIVE worker for task; start_sample likely to fail.")
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
        # Immediately release the session to avoid occupying the single-worker
        try:
            self.env.complete()
        except Exception:
            pass

    def __len__(self) -> int:
        return 100  # arbitrary number of steps

    def __getitem__(self, idx):
        return {"prompt": self.prompt}


class MultiPromptDataset(Dataset):
    """Dataset that cycles through a fixed set of N prompts each epoch.

    Prompts are collected from the controller at indices [START_INDEX .. START_INDEX + N_PROMPTS - 1].
    __len__ returns N_PROMPTS so a single epoch corresponds to exactly one pass over the set.
    """

    def __init__(self, task_name: str, start_index: int, n_prompts: int, controller_port: int = 5000):
        assert n_prompts >= 1
        self.task_name = task_name
        self.prompts: List[str] = []
        # Ensure at least one worker is alive
        tmp_env = AgentBenchHTTPEnv(controller_port=controller_port, task_name=task_name, index=start_index)
        if not tmp_env.wait_for_worker(timeout_sec=60.0, poll_sec=1.0):
            print("[WARN] No ALIVE worker for task; collecting prompts may fail.")
        # Collect initial prompts for the N indices
        for k in range(n_prompts):
            idx = start_index + k
            e = AgentBenchHTTPEnv(controller_port=controller_port, task_name=task_name, index=idx)
            try:
                resp = e.reset()
                prompt = build_prompt_from_resp(resp)
                self.prompts.append(prompt)
            except Exception:
                self.prompts.append("No observation provided. Begin.")
            finally:
                try:
                    e.complete()
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx % len(self.prompts)]}


dataset: Dataset
if N_PROMPTS > 1:
    print(f"[Dataset] Multi-prompt mode: N_PROMPTS={N_PROMPTS}, START_INDEX={START_INDEX}")
    dataset = MultiPromptDataset(task_name=TASK_NAME, start_index=START_INDEX, n_prompts=N_PROMPTS, controller_port=5000)
else:
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

def _parse_env_reward(result: Union[dict, None]) -> float:
    """Extract a numeric reward from TaskOutput.result if present.

    Supported patterns:
    - WebShop: {"reward": float, ...}
    - ALFWorld: {"result": 0/1 or True/False}
    Returns 0.0 if not available.
    """
    try:
        if not isinstance(result, dict):
            return 0.0
        if "reward" in result and isinstance(result["reward"], (int, float)):
            return float(result["reward"])
        if "result" in result:
            val = result["result"]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, bool):
                return 1.0 if val else 0.0
        return 0.0
    except Exception:
        return 0.0


def ab_reward_function(completions, prompts, **kwargs):
    """Compute rewards for each candidate directly from env results.

    For each completion candidate, we evaluate it in a fresh session for the
    same task/index and record the environment's terminal reward if available.
    Then we commit the best candidate into the main ongoing session to advance
    training context.
    """
    rewards = []
    n_gen = cfg.num_generations
    for i in range(0, len(completions), n_gen):
        # Normalize candidates
        cands = [first_action_block(sanitize(c)) for c in completions[i : i + n_gen]]

        # Determine prompt text for logging (fall back to dataset.prompt)
        try:
            prompt_text = prompts[i // n_gen]
        except Exception:
            prompt_text = dataset.prompt

        cand_rewards = []
        # Evaluate each candidate in an isolated session to get actual env reward
        for j, cand in enumerate(cands):
            r = 0.0
            eval_env = AgentBenchHTTPEnv(controller_port=5000, task_name=TASK_NAME, index=START_INDEX)
            try:
                eval_env.wait_for_worker(timeout_sec=15.0, poll_sec=1.0)
                _ = eval_env.reset()
                if eval_env.session_id is None:
                    raise RuntimeError("start_sample failed (no session)")
                # For multi-step environments (e.g., ALFWorld), roll out to termination using the model
                r = rollout_to_termination(eval_env, cand, max_steps=6)
            except Exception:
                r = 0.0
            finally:
                try:
                    eval_env.complete()
                except Exception:
                    pass

            cand_rewards.append(r)

            # Log per-candidate evaluation
            try:
                with open(training_log_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "type": "candidate_evaluation",
                                "prompt": prompt_text,
                                "completion_idx": j,
                                "completion": cand,
                                "env_reward": r,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            except Exception:
                pass

        # Select best candidate by reward, with simple tie-breakers
        best_idx = select_best_idx(cands, cand_rewards)
        best_cand = cands[best_idx]

        # Commit best candidate to the main session to advance context (optional in multi-prompt mode)
        COMMIT_BEST = os.getenv("COMMIT_BEST", "1" if N_PROMPTS <= 1 else "0").lower() in ("1", "true", "yes", "y")
        commit_out = {}
        if COMMIT_BEST:
            if env.session_id is None:
                # Try to (re)start main session if previous reset failed
                env.wait_for_worker(timeout_sec=15.0, poll_sec=1.0)
                env.reset()
            if env.session_id is None:
                # Still no session; skip commit for this round
                try:
                    with open(training_log_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "type": "commit_skipped",
                                    "prompt": prompt_text,
                                    "reason": "no active session",
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            else:
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

        # Persist reward history incrementally (include task_name metadata)
        try:
            try:
                with open(reward_log_file, "r") as f:
                    reward_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                reward_data = {
                    "task_name": TASK_NAME,
                    "top_entropy_quantile": cfg.top_entropy_quantile,
                    "entries": [],
                }

            # Normalize structure if older list-style file is encountered
            if isinstance(reward_data, list):
                reward_data = {
                    "task_name": TASK_NAME,
                    "top_entropy_quantile": cfg.top_entropy_quantile,
                    "entries": reward_data,
                }
            if not isinstance(reward_data, dict):
                reward_data = {
                    "task_name": TASK_NAME,
                    "top_entropy_quantile": cfg.top_entropy_quantile,
                    "entries": [],
                }
            if "entries" not in reward_data or not isinstance(reward_data.get("entries"), list):
                reward_data["entries"] = []
            if "task_name" not in reward_data:
                reward_data["task_name"] = TASK_NAME
            if "top_entropy_quantile" not in reward_data:
                reward_data["top_entropy_quantile"] = cfg.top_entropy_quantile

            reward_data["entries"].append(cand_rewards)

            with open(reward_log_file, "w") as f:
                json.dump(reward_data, f, indent=2)
        except Exception:
            pass

        # Refresh dataset only in legacy single-prompt mode; multi-prompt stays fixed per epoch
        if N_PROMPTS <= 1:
            try:
                if commit_done:
                    dataset.resp = env.reset()
                else:
                    dataset.resp = commit_out
                dataset.prompt = build_prompt_from_resp(dataset.resp)
            except Exception:
                pass
            # Free main session to allow subsequent candidate evaluations
            try:
                env.complete()
            except Exception:
                pass
    return rewards


# ---------------------------------------------------------------------------
# GRPO configuration and trainer
# ---------------------------------------------------------------------------

# Allow epochs to be overridden to enable epoch-style cycling over prompts
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "10"))

cfg = GRPOConfig(
    output_dir="grpo_logs",  # will be moved under AgentBench root with timestamp below
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    top_entropy_quantile=1.0,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=1,
    save_strategy="no",  # disable checkpoint saves
    num_generations=8,
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

# Robust repo-root detection: works from repo root, src/, or scripts/
here = Path(__file__).resolve().parent
REPO_ROOT = here if (here / "src").exists() else here.parent
LOGS_BASE = REPO_ROOT / "grpo_logs"
OUTPUT_DIR = LOGS_BASE / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure Trainer also writes its artifacts (trainer_state.json, results) into the same timestamped folder
cfg.output_dir = str(OUTPUT_DIR)

training_log_file = OUTPUT_DIR / f"training_history_{timestamp}.jsonl"
reward_log_file = OUTPUT_DIR / f"reward_history_{timestamp}.json"

# Initialize files
try:
    # Include task_name and top_entropy_quantile in reward history metadata
    with open(reward_log_file, "w") as f:
        json.dump(
            {
                "task_name": TASK_NAME,
                "top_entropy_quantile": cfg.top_entropy_quantile,
                "entries": [],
            },
            f,
            indent=2,
        )
    with open(training_log_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "type": "config",
                    "model_name": MODEL_NAME,
                    "task_name": TASK_NAME,
                    "top_entropy_quantile": cfg.top_entropy_quantile,
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
