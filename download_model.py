from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.base_config import MODEL_NAME, MODEL_DOWNLOAD_DIR

login('')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
tokenizer.save_pretrained(MODEL_DOWNLOAD_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # 自动分配到 GPU
)
model.save_pretrained(MODEL_DOWNLOAD_DIR)