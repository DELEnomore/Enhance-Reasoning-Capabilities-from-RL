from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

datasets = math_datasets
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-1.5B',
        path='drive/MyDrive/colab_workspace/download_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        max_out_len=16384,
        max_seq_len=16384,
        batch_size=8,
        generation_kwargs=dict(
            num_return_sequences=4,
            top_p=0.95,
            temperature=0.6,
            repetition_penalty=1.1
        ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
