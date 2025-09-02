GOOGLE_DRIVE_WORKSPACE_DIR = 'drive/MyDrive/colab_workspace'

# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f'current model: {MODEL_NAME}')
MODEL_DOWNLOAD_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/download_models/{MODEL_NAME}'
MODEL_CHECKPOINT_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/finetune_models/{MODEL_NAME}'
MODEL_OUTPUT_DIR = MODEL_CHECKPOINT_DIR + '/best_model'

DATASET_CACHE_DIR = DATASET_CACHE_DIR = f'{GOOGLE_DRIVE_WORKSPACE_DIR}/datasets'