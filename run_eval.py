import sys

from opencompass.cli.main import main

from configs.base_config import GOOGLE_DRIVE_WORKSPACE_DIR

sys.argv = [
    "main",
    "configs/eval_config.py",
    "--work-dir",
    GOOGLE_DRIVE_WORKSPACE_DIR
]
main()
