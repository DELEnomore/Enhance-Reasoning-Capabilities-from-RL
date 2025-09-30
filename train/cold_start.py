import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, \
    DataCollatorForLanguageModeling
from configs.base_config import RL_MODEL_CHECKPOINT_DIR, MODEL_DOWNLOAD_DIR, COLD_START_MODEL_CHECKPOINT_DIR
from train.dataset.numina_math_qwq_dataset import NuminaMathQwQDataset
from train.dataset.open_rs_dataset import OpenRsDataset


OUTPUT_DIR = COLD_START_MODEL_CHECKPOINT_DIR + '/best_model'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DOWNLOAD_DIR, repo_type='')

dataset = NuminaMathQwQDataset(tokenizer).get_data('sft', split='train')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    torch_dtype=torch.bfloat16,  # 混合精度
    device_map="auto",  # 自动分配到 GPU
).to('cuda')

lora_config = LoraConfig(
    r=8,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放因子
    lora_dropout=0.05,  # Dropout 概率
    bias="none",  # LoRA bias 设置
    task_type="CAUSAL_LM",  # 任务类型：自回归文本生成
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # 如果需要根据具体模型结构，调整 target_modules
)

peft_model = get_peft_model(model, lora_config).to("cuda")
peft_model.print_trainable_parameters()  # 查看可训练参数量

def check_point_exists():
    if [os.path.join(COLD_START_MODEL_CHECKPOINT_DIR, d) for d in os.listdir(COLD_START_MODEL_CHECKPOINT_DIR) if d.startswith("checkpoint")]:
        return True
    return False

training_args = TrainingArguments(
    output_dir=COLD_START_MODEL_CHECKPOINT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1000,
    eval_strategy='no',
    save_strategy="steps",
    save_steps=1000,
    report_to="none",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)


trainer.train(check_point_exists())
trainer.save_model(OUTPUT_DIR)