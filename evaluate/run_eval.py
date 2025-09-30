from datetime import datetime
import os

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.base_config import MODEL_DOWNLOAD_DIR, RL_MODEL_CHECKPOINT_DIR, MODEL_NAME, EVAL_OUTPUT_DIR

time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
temp_model_path = '/.temp_model'
os.mkdir(temp_model_path)

def get_checkpoint_dirs(path="."):
    """
    获取指定路径下所有以'checkpoint'开头的目录名称
    """
    checkpoint_dirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            checkpoint_dirs.append(item)

    return checkpoint_dirs

def merge_lora_model(model_path, lora_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(temp_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(temp_model_path)

def main(model_path, lora_path, model_name=MODEL_NAME):
    if lora_path:
        merge_lora_model(model_path, lora_path)
        model_path = temp_model_path

    evaluation_tracker = EvaluationTracker(
        output_dir=f'{EVAL_OUTPUT_DIR}/{model_name}-{time}',
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,  # Set to path if using custom tasks
        # Remove the parameter below once your configuration is tested
        max_samples=10
    )

    generation_params = GenerationParameters(
        max_new_tokens=3,
        top_p=0.95,
        temperature=0.6,
        repetition_penalty=1.1,
    )

    model_config = VLLMModelConfig(
        model_name=model_path,
        generation_parameters=generation_params
    )

    pipeline = Pipeline(
        tasks="lighteval|aime24|0,lighteval|aime25|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()
    os.remove(temp_model_path)

if __name__ == "__main__":
    main(MODEL_DOWNLOAD_DIR, None, MODEL_NAME)
    # checkpoints = get_checkpoint_dirs(RL_MODEL_CHECKPOINT_DIR)
    # for checkpoint in checkpoints:
    #     main(MODEL_DOWNLOAD_DIR, os.path.join(RL_MODEL_CHECKPOINT_DIR, checkpoint), f'MODEL_NAME-{checkpoint}')