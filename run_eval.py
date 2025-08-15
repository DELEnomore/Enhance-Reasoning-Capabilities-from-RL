import sys

from opencompass.cli.main import main

sys.argv = [
    "main",
    "configs/eval_config.py"
]
main()
