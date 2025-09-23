

import re

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR
from train.dataset.dataset_base import DatasetBase, batch_format_chatml


# login('')


class OpenRsDataset(DatasetBase):
    DATASET_NAME = "knoveleng/open-rs"
    QUESTION_NAME = 'problem'
    SOLUTION_NAME = 'solution'
    ANSWER_NAME = 'answer'

if __name__ == '__main__':
    dataset = load_dataset('knoveleng/open-rs', split='train', cache_dir=DATASET_CACHE_DIR)
    dataset = dataset.filter(lambda x:x['correct']==True)
    print('Done')