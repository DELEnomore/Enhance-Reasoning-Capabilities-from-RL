from huggingface_hub import login



GOOGLE_DRIVE_WORKSPACE_DIR = 'drive/MyDrive/colab_workspace'

MODEL_CACHE_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/models'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'

MODEL_CHECKPOINT_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/finetune_models/' + MODEL_NAME
MODEL_OUTPUT_DIR = MODEL_CHECKPOINT_DIR + '/best_model'

