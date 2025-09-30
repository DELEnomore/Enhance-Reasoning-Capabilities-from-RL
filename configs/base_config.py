from pathlib import Path

WORK_PATH = Path(__file__).parent.parent.resolve()
GOOGLE_DRIVE_WORKSPACE_DIR = f'{WORK_PATH}/drive/MyDrive/colab_workspace'

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f'current model: {MODEL_NAME}')
MODEL_DOWNLOAD_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/download_models/{MODEL_NAME}'
RL_MODEL_CHECKPOINT_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/finetune_models/{MODEL_NAME}/rl_finetune'
COLD_START_MODEL_CHECKPOINT_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/finetune_models/{MODEL_NAME}/cold_start'
MODEL_OUTPUT_DIR = RL_MODEL_CHECKPOINT_DIR + '/best_model'
EVAL_OUTPUT_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/eval_output'

DATASET_CACHE_DIR = DATASET_CACHE_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/datasets'