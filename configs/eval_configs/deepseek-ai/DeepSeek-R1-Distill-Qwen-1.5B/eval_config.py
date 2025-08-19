from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

datasets = math_datasets
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-1.5B',
        path='drive/MyDrive/colab_workspace/download_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        max_seq_len=5000,
        max_out_len=5000,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
