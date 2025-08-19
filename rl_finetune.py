from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import PPOTrainer

from configs.base_config import MODEL_DOWNLOAD_DIR
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DOWNLOAD_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    device_map="auto",  # 自动分配到 GPU
)


lora_config = LoraConfig(
    r=16,           # LoRA秩
    lora_alpha=32,  # Alpha参数（缩放因子）
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


def reward(texts):
    pass

training_args = TrainingArguments(
    output_dir="./grpo_lora_results",
    per_device_train_batch_size=2,   # 根据GPU显存调整
    gradient_accumulation_steps=4,   # 梯度累积
    learning_rate=1.5e-5,            # LoRA的典型学习率
    max_grad_norm=0.3,               # 梯度裁剪
    logging_steps=10,
    report_to="none",                # 禁用默认报告
    remove_unused_columns=False,     # PPO需要保留所有列
)

ppo_trainer = PPOTrainer(
    model=model,
    config=training_args,
    tokenizer=tokenizer,
    dataset=dataset,
)