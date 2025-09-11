import re

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR
from train.dataset.dataset_base import DatasetBase


# login('')
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


class NuminaMathDataset(DatasetBase):
    DATASET_NAME = "BeingIsA/NuminaMath-CoT"
    QUESTION_NAME = 'problem'
    SOLUTION_NAME = 'solution'
    ANSWER_NAME = 'answer'