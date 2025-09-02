import os

import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import parse, LatexExtractionConfig, verify
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from configs.base_config import MODEL_CHECKPOINT_DIR, MODEL_OUTPUT_DIR, MODEL_NAME, MODEL_DOWNLOAD_DIR
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from prepare_dataset import get_dataset

dataset = get_dataset()

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            nits=False,
                            boxed="all",
                            equations=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    torch_dtype=torch.bfloat16,  # 混合精度
    device_map="auto",  # 自动分配到 GPU
).to("cuda")

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", 'v_proj', 'o_proj', 'down_proj', 'up_proj', 'gate_proj'],  # 根据模型结构调整
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config).to("cuda")

training_args = GRPOConfig(
    output_dir=MODEL_CHECKPOINT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=1.0e-06,
    logging_steps=1,
    per_device_train_batch_size=8,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=100,
    report_to='none',
    bf16=True,
)

trainer = GRPOTrainer(
    model=peft_model,
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
)

def check_point_exists():
    if [os.path.join(MODEL_CHECKPOINT_DIR, d) for d in os.listdir(MODEL_CHECKPOINT_DIR) if d.startswith("checkpoint")]:
        return True
    return False


trainer.train(check_point_exists())
trainer.save_model(MODEL_OUTPUT_DIR)