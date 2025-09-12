from mmengine.config import read_base

from opencompass.models import HuggingFacewithChatTemplate
with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets

datasets = []
datasets += math_datasets

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
            # TODO 这个不支持>1
            num_return_sequences=2,
            repetition_penalty=1.1
          ),
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-1.5B-BestModel',
        path='../drive/MyDrive/colab_workspace/download_models/Qwen/Qwen2.5-1.5B-Instruct',
        peft_path='../drive/MyDrive/colab_workspace/finetune_models/Qwen/Qwen2.5-1.5B-Instruct/cold_start/best_model',
        max_seq_len=3000,
        max_out_len=3000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            # TODO 这个不支持>1
            num_return_sequences=2,
            repetition_penalty=1.1
        ),
        run_cfg=dict(num_gpus=1),
    ),
] + [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f'Qwen2.5-1.5B-checkpoint-{x}000',
        path='../drive/MyDrive/colab_workspace/download_models/Qwen/Qwen2.5-1.5B-Instruct',
        peft_path=f'../drive/MyDrive/colab_workspace/finetune_models/Qwen/Qwen2.5-1.5B-Instruct/cold_start/checkpoint-{x}000',
        max_seq_len=3000,
        max_out_len=3000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            # TODO 这个不支持>1
            num_return_sequences=2,
            repetition_penalty=1.1
        ),
        run_cfg=dict(num_gpus=1),
    ) for x in range(1, 5)

]
