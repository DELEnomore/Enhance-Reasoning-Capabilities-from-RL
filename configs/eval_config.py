from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

from configs.base_config import MODEL_NAME

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

datasets = math_datasets
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-1.5b-instruct-hf',
        path=MODEL_NAME,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
