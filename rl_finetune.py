import os

from configs.base_config import MODEL_CHECKPOINT_DIR, MODEL_OUTPUT_DIR
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# TODO reward定义
def reward():
    return []

training_args = GRPOConfig(
    output_dir=MODEL_CHECKPOINT_DIR,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
)

# TODO 支持从checkpoint继续训练
trainer = GRPOTrainer(
    model="Model",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
)

def check_point_exists():
    if [os.path.join(MODEL_CHECKPOINT_DIR, d) for d in os.listdir(MODEL_CHECKPOINT_DIR) if d.startswith("checkpoint")]:
        return True
    return False

trainer.train(check_point_exists())
trainer.save_model(MODEL_OUTPUT_DIR)