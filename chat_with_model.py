from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from configs.base_config import MODEL_DOWNLOAD_DIR


def format_chatml(input, output):
    messages = []
    if input:
        messages.append({"role": "user", "content": input})
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def format_chat_input(input, tokenizer):
    chatml_input = format_chatml(input, None)
    formatted_input = tokenizer.apply_chat_template(chatml_input, tokenize=False, add_generation_prompt=True)
    return formatted_input


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DOWNLOAD_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DOWNLOAD_DIR,
    device_map="auto",  # 自动分配到 GPU
)

model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)

print("开始对话！输入 'exit' 结束对话。")
while True:
    user_input = input("你: ")
    if user_input.lower() == "exit":
        print("对话结束。")
        break
    formatted_input = format_chat_input(user_input, tokenizer)
    response = pipe(formatted_input, truncation=True, max_length=500)
    print(f"模型: {response[0]['generated_text']}")