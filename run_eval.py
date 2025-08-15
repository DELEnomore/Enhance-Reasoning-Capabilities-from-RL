import sys

from opencompass.cli.main import main

from configs.base_config import GOOGLE_DRIVE_WORKSPACE_DIR, MODEL_NAME

sys.argv = [
    "main",
    f"configs/{MODEL_NAME}/eval_config.py",
    "--work-dir",
    GOOGLE_DRIVE_WORKSPACE_DIR
]
main()
