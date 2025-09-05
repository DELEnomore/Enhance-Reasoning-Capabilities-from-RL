from datasets import load_dataset, Dataset
from huggingface_hub import login

from configs.base_config import DATASET_CACHE_DIR

# login()

def batch_format_data(data, tokenizer):
    problems = data['problem']
    answers = data['answer']
    formated_data = {
        'prompt':[
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                ], tokenize=False
            )for problem in problems
        ],
        'answer': [f'${str(answer)}$' for answer in answers]
    }
    return formated_data

def get_dataset(tokenizer):
    dataset = load_dataset("open-r1/OpenR1-Math-220k", 'default', split="train", cache_dir=DATASET_CACHE_DIR)
    dataset = dataset.select_columns(['problem', 'answer'])
    # 删掉选择题
    filtered_dataset = dataset.filter(lambda x: not x['answer'].isalpha())
    print(f'dataset size: {len(filtered_dataset)}')
    formated_data = filtered_dataset.map(batch_format_data, fn_kwargs={'tokenizer': tokenizer}, batched=True, load_from_cache_file=False)
    return formated_data

if __name__ == '__main__':
    get_dataset(None)