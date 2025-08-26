from mmengine.config import read_base
from opencompass.datasets import CustomDataset
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils import extract_non_reasoning_content
with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets, math_reader_cfg, math_infer_cfg, \
        math_eval_cfg

datasets = dict(
        type=CustomDataset,
        abbr='math-500',
        path='opencompass/math',
        file_name='test_prm800k_500.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-1.5B',
        path='drive/MyDrive/colab_workspace/download_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        max_seq_len=6000,
        max_out_len=6000,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            # TODO 这个不支持>1吗？ acc结果为0
            num_return_sequences=2,
            repetition_penalty=1.1
          ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]