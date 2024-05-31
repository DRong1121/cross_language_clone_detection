import os
import json
import jsonlines
from transformers import RobertaTokenizer

from core.tokenizer.java_tokenizer import JavaTokenizer
from core.tokenizer.python_tokenizer import PythonTokenizer

# define the path of pre-training dataset
ccs_root = '/Users/rongdang/Desktop/semantic-code-clone/dataset/cross-language/CodeNet_Microsoft'
correct_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'correct_functions.json')

# define the path of fine-tuning dataset
c4_root = '/Users/rongdang/Desktop/semantic-code-clone/dataset/cross-language/C4'
correct_programs = os.path.join(c4_root, 'preprocessed_dataset', 'correct_programs.json')

# define the path of CSN_Generated dataset
csn_generated = '/Users/rongdang/Desktop/semantic-code-clone/dataset/pre-training/CSN-Generated'


def count_code_lines(code_str):
    line_num = 0
    code_lines = code_str.split('\n')
    for line in code_lines:
        line = line.strip('\n').strip('\t').strip()
        if line != '\n' and line != '\t' and line != '':
            line_num += 1
    return line_num


def count_code_tokens(tokenizer, code_str):
    code = ' '.join(code_str.replace('\n', ' ').strip().split())
    code_tokens = tokenizer.tokenize(code)
    return len(code_tokens)


def analyze_pre_training_dataset():

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    java_line_nums = list()
    java_token_nums = list()
    python_line_nums = list()
    python_token_nums = list()

    with open(correct_functions, mode='r', encoding='utf-8') as correct_functions_file:
        json_data = json.load(correct_functions_file)
        java_pool = json_data['java']
        python_pool = json_data['python']

        for task, solutions in java_pool.items():
            for _, solution in solutions.items():
                java_line_num = count_code_lines(solution)
                java_token_num = count_code_tokens(tokenizer, solution)
                java_line_nums.append(java_line_num)
                java_token_nums.append(java_token_num)

        for task, solutions in python_pool.items():
            for _, solution in solutions.items():
                python_line_num = count_code_lines(solution)
                python_token_num = count_code_tokens(tokenizer, solution)
                python_line_nums.append(python_line_num)
                python_token_nums.append(python_token_num)

    correct_functions_file.close()

    print('-----Pre-training dataset statistics-----')
    print('avg number of lines in Java function: ' +
          str(round(sum(java_line_nums) / len(java_line_nums), 2)))
    print('avg number of code tokens in Java function: ' +
          str(round(sum(java_token_nums) / len(java_token_nums), 2)))
    print('avg number of lines in Python function: ' +
          str(round(sum(python_line_nums) / len(python_line_nums), 2)))
    print('avg number of code tokens in Python function: ' +
          str(round(sum(python_token_nums) / len(python_token_nums), 2)))


def analyze_fine_tuning_dataset():

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    java_line_nums = list()
    java_token_nums = list()
    python_line_nums = list()
    python_token_nums = list()

    with open(correct_programs, mode='r', encoding='utf-8') as correct_programs_file:
        json_data = json.load(correct_programs_file)
        java_pool = json_data['java']
        python_pool = json_data['python']

        for task, solutions in java_pool.items():
            for _, solution in solutions.items():
                java_line_num = count_code_lines(solution)
                java_token_num = count_code_tokens(tokenizer, solution)
                java_line_nums.append(java_line_num)
                java_token_nums.append(java_token_num)

        for task, solutions in python_pool.items():
            for _, solution in solutions.items():
                python_line_num = count_code_lines(solution)
                python_token_num = count_code_tokens(tokenizer, solution)
                python_line_nums.append(python_line_num)
                python_token_nums.append(python_token_num)

    correct_programs_file.close()

    print('-----Fine-tuning dataset statistics-----')
    print('avg number of lines in Java program: ' +
          str(round(sum(java_line_nums) / len(java_line_nums), 2)))
    print('avg number of code tokens in Java program: ' +
          str(round(sum(java_token_nums) / len(java_token_nums), 2)))
    print('avg number of lines in Python program: ' +
          str(round(sum(python_line_nums) / len(python_line_nums), 2)))
    print('avg number of code tokens in Python program: ' +
          str(round(sum(python_token_nums) / len(python_token_nums), 2)))


def analyze_csn_generate_dataset():
    # generated python functions:
    # unparsable functions in the training/valid/test set: 2398/73/131
    sets = ['train', 'valid']
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    java_line_nums = list()
    java_token_nums = list()
    python_line_nums = list()
    python_token_nums = list()

    for set_name in sets:
        file_name = 'pair_' + set_name + '.jsonl'
        file_path = os.path.join(csn_generated, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                java_line_num = count_code_lines(data['src_code'])
                java_token_num = count_code_tokens(tokenizer, data['src_code'])
                java_line_nums.append(java_line_num)
                java_token_nums.append(java_token_num)

                python_line_num = count_code_lines(data['tgt_code'])
                python_token_num = count_code_tokens(tokenizer, data['tgt_code'])
                python_line_nums.append(python_line_num)
                python_token_nums.append(python_token_num)
        f.close()

    print('-----CSN-Generated dataset statistics-----')
    print('avg number of lines in Java function: ' +
          str(round(sum(java_line_nums) / len(java_line_nums), 2)))
    print('avg number of code tokens in Java function: ' +
          str(round(sum(java_token_nums) / len(java_token_nums), 2)))
    print('avg number of lines in Python function: ' +
          str(round(sum(python_line_nums) / len(python_line_nums), 2)))
    print('avg number of code tokens in Python function: ' +
          str(round(sum(python_token_nums) / len(python_token_nums), 2)))


if __name__ == "__main__":

    # analyze_pre_training_dataset()
    # analyze_fine_tuning_dataset()

    # analyze_csn_generate_dataset()
    pass
