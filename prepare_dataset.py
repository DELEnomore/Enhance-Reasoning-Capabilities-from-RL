from datasets import load_dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR

# login()

def batch_format_data(data, tokenizer):
    problems = data['question']
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

