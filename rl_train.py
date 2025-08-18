
import os
import math
from dataclasses import dataclass
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    pipeline,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig


# =========================
# 1) Config
# =========================
base_model_name = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
output_dir = os.environ.get("OUTPUT_DIR", "./ppo_out")
os.makedirs(output_dir, exist_ok=True)

# TinyLlama uses chat templates; we'll still treat it as a plain causal LM with a very simple prompt format.
# If you use an instruction-tuned model, you'll usually get better responses.

ppo_config = PPOConfig(
    model_name=base_model_name,
    learning_rate=1e-5,
    batch_size=4,               # effective batch size per PPO step
    mini_batch_size=2,          # split for PPO optimization
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    target_kl=0.1,
    ppo_epochs=4,
    seed=42,
    accelerator_kwargs={"even_batches": False},
)

# LoRA keeps memory small
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# =========================
# 2) Toy prompts dataset
# =========================
# You can replace with a real domain-specific dataset. Keep it small for a demo.
prompts = [
    "Give a friendly, helpful answer: How can I stay focused while studying?",
    "Be concise but kind: Tips to improve time management?",
    "Explain briefly: What's a good way to learn a new programming language?",
    "Suggest 3 ideas: Healthy snack options for work?",
    "One paragraph: How to build a daily exercise habit?",
    "In 2-3 sentences: How to prepare for a job interview?",
]

dataset = Dataset.from_dict({"query": prompts})

# =========================
# 3) Tokenizer & Models
# =========================
print(f"Loading base model: {base_model_name}")

device_map = "auto" if torch.cuda.is_available() else None

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    # Ensure pad token exists; needed for batched generation
    tokenizer.pad_token = tokenizer.eos_token

auto_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_name,
    torch_dtype=auto_dtype if torch.cuda.is_available() else torch.float32,
    device_map=device_map,
    peft_config=peft_config,
)

# reference (frozen) model for KL penalty
ref_model = create_reference_model(model)

# simple sentiment reward model
print("Loading reward pipeline (sentiment-analysis)...")
reward_pipe = pipeline(
    "sentiment-analysis",
    model="lvwerra/distilbert-imdb",
    device=0 if torch.cuda.is_available() else -1,
)

# =========================
# 4) PPO Trainer
# =========================
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# =========================
# 5) Generation settings
# =========================
gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    pad_token_id=tokenizer.pad_token_id,
)

# =========================
# 6) Helper: compute rewards from text
# =========================
def reward_fn(responses: List[str]) -> List[float]:
    """Map each generated response to a scalar reward using sentiment.
    POSITIVE => +score, NEGATIVE => -score. Add small length bonus for diverse outputs.
    """
    results = reward_pipe(responses, truncation=True)
    rewards = []
    for res, text in zip(results, responses):
        score = float(res.get("score", 0.0))
        label = str(res.get("label", "POSITIVE")).upper()
        signed = score if "POS" in label else -score
        # light shaping: encourage some substance but avoid verbosity
        length_bonus = min(len(text.split()) / 100.0, 0.2)
        rewards.append(signed + length_bonus)
    return rewards

# =========================
# 7) Training loop (a few PPO steps)
# =========================
num_ppo_steps = int(os.environ.get("PPO_STEPS", 3))
print(f"Starting PPO for {num_ppo_steps} steps…")

model.train()
for step_idx in range(num_ppo_steps):
    # Sample a mini-batch of queries
    batch = ppo_trainer._prepare_dataset(
        ppo_trainer.dataset.shuffle(seed=ppo_config.seed).select(range(ppo_config.batch_size))
    )
    queries = batch["query"]

    # Tokenize
    query_toks = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(model.device)

    # Generate responses with the current policy
    with torch.no_grad():
        response_toks = model.generate(**query_toks, **gen_kwargs)

    # Decode only the generated continuation (past the prompt)
    responses = []
    for in_ids, out_ids in zip(query_toks["input_ids"], response_toks):
        gen_part = out_ids[len(in_ids):] if len(out_ids) > len(in_ids) else out_ids[-1:]
        responses.append(tokenizer.decode(gen_part, skip_special_tokens=True).strip())

    # Compute rewards
    rewards = reward_fn(responses)

    # Run a PPO optimization step
    stats = ppo_trainer.step(queries, responses, rewards)

    # Log a couple of lines to stdout
    mean_reward = sum(rewards) / max(len(rewards), 1)
    kl = float(stats.get("kl", torch.tensor(0.0)))
    print(f"Step {step_idx+1}/{num_ppo_steps} | mean_reward={mean_reward:.4f} | kl={kl:.4f}")
    for q, r, rw in zip(queries, responses, rewards):
        print("\nQ:", q)
        print("A:", r)
        print(f"reward={rw:.3f}")

# =========================
# 8) Save adapters & value head
# =========================
print("Saving PPO-tuned adapters to:", output_dir)
ppo_trainer.save_pretrained(output_dir)
print("Done.")
