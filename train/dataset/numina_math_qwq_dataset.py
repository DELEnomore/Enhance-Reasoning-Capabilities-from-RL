import re

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR
from train.dataset.dataset_base import DatasetBase, batch_format_chatml


# login('')


class NuminaMathQwQDataset(DatasetBase):
    DATASET_NAME = "BeingIsA/NuminaMath-QwQ-CoT"
    QUESTION_NAME = 'prompt'
    SOLUTION_NAME = 'response'
    ANSWER_NAME = 'ground_truth'

if __name__ == '__main__':
    dataset = load_dataset('BeingIsA/NuminaMath-QwQ-CoT', split='train', cache_dir=DATASET_CACHE_DIR)
    dataset = dataset.filter(lambda x:x['correct']==True)
    print('Done')