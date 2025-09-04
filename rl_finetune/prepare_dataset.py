from datasets import load_dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR

# login()

def batch_format_data(data, tokenizer):
    problems = data['problem']
    answers = data['answer']
    formated_data = {
        'prompt':[
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                ], tokenize=False
            )for problem, answer in zip(problems, answers)
        ],
        'answer': answers
    }
    return formated_data

def get_dataset(tokenizer):
    dataset = load_dataset("open-r1/OpenR1-Math-220k", 'default', split="train", cache_dir=DATASET_CACHE_DIR)
    formated_data = dataset.map(batch_format_data, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    return formated_data
