import re

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR

# login('')

DATASET_NAME = "BeingIsA/NuminaMath-CoT"



def batch_format_rl_data(data, tokenizer):
    problems = data['problem']
    answers = data['answer']
    formated_data = {
        'prompt':[
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem}
                ], tokenize=False
            )for problem in problems
        ],
        'answer': [f'${str(answer)}$' for answer in answers]
    }
    return formated_data

def batch_format_sft_data(data, tokenizer):
    formated_data = tokenizer.apply_chat_template(data['messages'], tokenize=True)
    return {
        'input_ids': formated_data,
        'labels': formated_data
    }

def extract_answer_column(example):
    solution = example['solution']
    pattern = r'\\boxed\{((?:[^{}]*(?:\{[^{}]*\}[^{}]*)*))\}'
    matches = re.findall(pattern, solution)
    answer = ''
    if matches:
        answer = matches[-1].strip()
    else:
        print(f'match failed, solution: {solution}')
    example['answer'] = answer
    return example


def get_sft_data(tokenizer):
    dataset = load_dataset(DATASET_NAME, cache_dir=DATASET_CACHE_DIR)
    formatted_data = dataset.map(batch_format_sft_data, fn_kwargs={'tokenizer': tokenizer}, batched=True, load_from_cache_file=False)
    return formatted_data


def get_rl_data(tokenizer):
    return None

if __name__ == '__main__':
    get_sft_data(None)