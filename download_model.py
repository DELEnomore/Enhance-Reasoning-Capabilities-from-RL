import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.base_config import MODEL_NAME, MODEL_CACHE_DIR

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # 自动分配到 GPU
    cache_dir=MODEL_CACHE_DIR
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, cache_dir=MODEL_CACHE_DIR)