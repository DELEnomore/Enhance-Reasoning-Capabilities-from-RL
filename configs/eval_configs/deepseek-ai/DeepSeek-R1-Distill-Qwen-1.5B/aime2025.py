from mmengine.config import read_base

from opencompass.datasets import CustomDataset
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils import extract_non_reasoning_content
with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_reader_cfg, aime2024_infer_cfg, \
    aime2024_eval_cfg

datasets = [
    dict(
        abbr=f'aime2025',
        type=CustomDataset,
        path='opencompass/aime2025',
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg,
    )
]
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-1.5B',
        path='drive/MyDrive/colab_workspace/download_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/evaluate',
        max_seq_len=6000,
        max_out_len=6000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            # TODO 这个不支持>1
            num_return_sequences=1,
            repetition_penalty=1.1
          ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]