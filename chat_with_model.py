import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from configs.base_config import MODEL_DOWNLOAD_DIR
import json

def format_chat_input(input, tokenizer):
    chatml_input = [{"role": "user", "content": input + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tokenizer.apply_chat_template(chatml_input, tokenize=True, add_generation_prompt=True)


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DOWNLOAD_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    device_map="auto",  # 自动分配到 GPU
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 移动模型到设备
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)

print("开始对话！输入 'exit' 结束对话。")
while True:
    user_input = input("用户: ")
    if user_input.lower() == 'exit':
        print("对话结束。")
        break

    formatted_input = format_chat_input(user_input, tokenizer)

    # 生成回复
    outputs = model.generate(
        formatted_input,
        max_length=2000,                   # 最大生成长度
        num_return_sequences=1,          # 生成1条回复
        # no_repeat_ngram_size=2,          # 避免重复n-gram
        # top_k=50,                        # 限制top-k采样
        # top_p=0.95,                      # 核采样
        # temperature=0.6,                 # 控制随机性
        pad_token_id=tokenizer.eos_token_id  # 防止警告
    )

    # 解码并输出结果
    response = tokenizer.decode(outputs[0])
    print("助手:", json.dumps(response))