from mmengine.config import read_base
from opencompass.datasets import CustomDataset
from opencompass.models import HuggingFacewithChatTemplate
with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
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
] + math_datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-1.5B',
        path='../drive/MyDrive/colab_workspace/download_models/Qwen/Qwen2.5-1.5B-Instruct',
        max_seq_len=3000,
        max_out_len=3000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            num_return_sequences=4,
            repetition_penalty=1.1
          ),
        run_cfg=dict(num_gpus=1),
    ),
] + [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f'Qwen2.5-1.5B-checkpoint-{x}000',
        path='../drive/MyDrive/colab_workspace/download_models/Qwen/Qwen2.5-1.5B-Instruct',
        peft_path=f'../drive/MyDrive/colab_workspace/finetune_models/Qwen/Qwen2.5-1.5B-Instruct/rl_finetune/checkpoint-{x}000',
        max_seq_len=3000,
        max_out_len=3000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            num_return_sequences=4,
            repetition_penalty=1.1
        ),
        run_cfg=dict(num_gpus=1),
    ) for x in range(1, 8)

]
