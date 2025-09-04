from datasets import load_dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR

login()

def batch_format_data(problems, answers):
    formated_problems = [{"role": "user", "content": input + "\nPlease reason step by step, and put your final answer within \\boxed{}."} for input in problems]
    return {'prompt': formated_problems,
            'answer': answers}

def get_dataset():
    data = load_dataset("open-r1/OpenR1-Math-220k", 'default', split="train", cache_dir=DATASET_CACHE_DIR)

    problems = data['problem']
    answers = data['answer']

    formated_data = batch_format_data(problems, answers)
    return formated_data
