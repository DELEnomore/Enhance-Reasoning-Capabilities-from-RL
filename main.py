from opencompass.cli.main import main
import sys
print(sys.version)
# 设置模拟参数
sys.argv = [
    "main",
    "--datasets", "math_500_gen",
    "--hf-type", "chat",
    "--hf-path", "Qwen/Qwen3-4B-Instruct-2507",
]
main()
