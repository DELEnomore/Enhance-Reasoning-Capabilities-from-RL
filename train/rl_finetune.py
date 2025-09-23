import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.base_config import MODEL_CHECKPOINT_DIR, MODEL_DOWNLOAD_DIR
from trl import GRPOConfig, GRPOTrainer

from train.dataset.numina_math_qwq_dataset import NuminaMathQwQDataset
from train.dataset.open_rs_dataset import OpenRsDataset
from train.rewards import accuracy_reward


CHECKPOINT_DIR = f'{MODEL_CHECKPOINT_DIR}/rl_finetune'

OUTPUT_DIR = CHECKPOINT_DIR + '/best_model'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DOWNLOAD_DIR)

dataset = OpenRsDataset(tokenizer).get_data('rl', split='train')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    torch_dtype=torch.bfloat16,  # 混合精度
    device_map="auto",  # 自动分配到 GPU
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", 'v_proj', 'o_proj', 'down_proj', 'up_proj', 'gate_proj'],  # 根据模型结构调整
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

training_args = GRPOConfig(
    output_dir=CHECKPOINT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=1.0e-06,
    logging_steps=1,
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=3584,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=20,
    report_to='none',
    bf16=False,
)

trainer = GRPOTrainer(
    model=peft_model,
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
)

def check_point_exists():
    if [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint")]:
        return True
    return False


trainer.train(check_point_exists())
trainer.save_model(OUTPUT_DIR)