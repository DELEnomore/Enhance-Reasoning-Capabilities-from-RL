
from huggingface_hub import login

GOOGLE_DRIVE_WORKSPACE_DIR = 'drive/MyDrive/colab_workspace/LLM'
_CACHE_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/cache'
MODEL_CACHE_DIR = _CACHE_DIR + '/model'
DATASET_CACHE_DIR = _CACHE_DIR + '/dataset'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'

MODEL_CHECKPOINT_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/model_output/' + MODEL_NAME
MODEL_OUTPUT_DIR = MODEL_CHECKPOINT_DIR + '/best_model'

login()
# 连接Google Drive。不使用可忽略这一行

