from datasets import load_dataset

from configs.base_config import DATASET_CACHE_DIR


def format_chatml(input, output):
    messages = []
    if input:
        messages.append({"role": "user", "content": input})
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def batch_format_chatml(batch_input, batch_output):
    batch_message = []
    for input, output in zip(batch_input, batch_output):
        batch_message.append(format_chatml(input, output))

    return batch_message

class DatasetInterface:
    PROMPT_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.\n'

    DATASET_NAME=None

    QUESTION_NAME=None
    SOLUTION_NAME=None
    ANSWER_NAME=None

    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def batch_format_sft_data(self, data):
        prompt = [x + self.PROMPT_SUFFIX for x in data[self.QUESTION_NAME]]
        output = data[self.SOLUTION_NAME]

        formated_data = batch_format_chatml(prompt, output)
        formated_data = self.tokenizer.apply_chat_template(formated_data, tokenize=True)
        return {
            'input_ids': formated_data,
            'labels': formated_data
        }

    def batch_format_rl_data(self, data):
        problems = data[self.QUESTION_NAME]
        answers = data[self.ANSWER_NAME]
        formated_data = {
            'prompt': [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": problem + self.PROMPT_SUFFIX}
                    ], tokenize=False
                ) for problem in problems
            ],
            'answer': [f'${str(answer)}$' for answer in answers]
        }
        return formated_data

    def get_data(self, mode):
        """
        :param mode:
            'sft' or 'rl'
        :return:
        """
        map_func = None
        if mode == 'sft':
            map_func=self.batch_format_sft_data
        if mode == 'rl':
            map_func=self.batch_format_rl_data
        dataset = load_dataset(self.DATASET_NAME, cache_dir=DATASET_CACHE_DIR)
        formatted_data = dataset.map(map_func, batched=True, load_from_cache_file=False)
        return formatted_data

