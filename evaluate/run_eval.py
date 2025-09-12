import sys
from opencompass.cli.main import main
from configs.base_config import MODEL_NAME, EVAL_OUTPUT_DIR

sys.argv = [
    "main",
    f"../configs/eval_configs/{MODEL_NAME}/aime2025.py",
    "--work-dir",
    EVAL_OUTPUT_DIR
]
main()
