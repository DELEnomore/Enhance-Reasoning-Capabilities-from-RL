from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

from configs.base_config import MODEL_NAME, MODEL_CACHE_DIR

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

def get_model_path():
    org_name, model_name = MODEL_NAME.split('/')
    result = MODEL_CACHE_DIR + '/' + f'models--{org_name}--{model_name}'

    return MODEL_CACHE_DIR + '/' + 'models--Qwen--Qwen2.5-1.5B-Instruct'


MODEL_PATH = get_model_path()


datasets = math_datasets
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-1.5b-instruct-hf',
        path=MODEL_PATH,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=0),
    )
]
