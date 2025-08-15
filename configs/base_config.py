from huggingface_hub import login

def get_model_path():
    org_name, model_name = MODEL_NAME.split('/')
    return MODEL_CACHE_DIR + '/' + f'models--{org_name}--{model_name}'


GOOGLE_DRIVE_WORKSPACE_DIR = 'drive/MyDrive/colab_workspace'

MODEL_CACHE_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/models'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'

MODEL_CHECKPOINT_DIR = GOOGLE_DRIVE_WORKSPACE_DIR + '/finetune_models/' + MODEL_NAME
MODEL_OUTPUT_DIR = MODEL_CHECKPOINT_DIR + '/best_model'
MODEL_PATH = get_model_path()


login('')
# 连接Google Drive。不使用可忽略这一行

