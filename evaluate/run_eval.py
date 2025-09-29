import sys
import os
from opencompass.cli.main import main
from configs.base_config import MODEL_NAME, EVAL_OUTPUT_DIR

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

sys.argv = [
    "main",
    f"../configs/eval_configs/{MODEL_NAME}/rl_finetune.py",
    "--work-dir",
    EVAL_OUTPUT_DIR,
]
main()
