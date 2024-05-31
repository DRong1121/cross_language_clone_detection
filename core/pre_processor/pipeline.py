import os
import sys
import json
from pprint import pprint

from core.tokenizer.java_tokenizer import JavaTokenizer
from core.tokenizer.python_tokenizer import PythonTokenizer


def pipeline_tokenization(input_file, correct_file, error_file):
    correct_functions = {
        'java': {},
        'python': {}
    }
    error_functions = {
        'java': {},
        'python': {}
    }

    java_tokenizer = JavaTokenizer()
    python_tokenizer = PythonTokenizer()

    with open(input_file, 'r') as input_file:
        json_data = json.load(input_file)
        java_pool = json_data['java']
        python_pool = json_data['python']

        print('Start analyzing Java functions/programs...')
        for task_id, solutions in java_pool.items():
            for function_id, function in solutions.items():
                try:
                    java_tokens = java_tokenizer.tokenize_code(code=function)
                    if task_id not in correct_functions['java'].keys():
                        correct_functions['java'][task_id] = dict()
                        correct_functions['java'][task_id][function_id] = function
                    else:
                        correct_functions['java'][task_id][function_id] = function
                except Exception as e:
                    if task_id not in error_functions['java'].keys():
                        error_functions['java'][task_id] = dict()
                        error_functions['java'][task_id][function_id] = str(e)
                    else:
                        error_functions['java'][task_id][function_id] = str(e)

        print('Start analyzing Python functions/programs...')
        for task_id, solutions in python_pool.items():
            for function_id, function in solutions.items():
                try:
                    python_tokens = python_tokenizer.tokenize_code(code=function)
                    if task_id not in correct_functions['python'].keys():
                        correct_functions['python'][task_id] = dict()
                        correct_functions['python'][task_id][function_id] = function
                    else:
                        correct_functions['python'][task_id][function_id] = function
                except Exception as e:
                    if task_id not in error_functions['python'].keys():
                        error_functions['python'][task_id] = dict()
                        error_functions['python'][task_id][function_id] = str(e)
                    else:
                        error_functions['python'][task_id][function_id] = str(e)
    input_file.close()

    print('Start writing correct_functions.json/correct_programs.json...')
    with open(correct_file, mode='w', encoding='utf-8') as json_file_to_write:
        json_file_to_write.write(json.dumps(correct_functions, indent=4))
    json_file_to_write.close()

    print('Start writing error_functions.json/error_programs.json...')
    with open(error_file, mode='w', encoding='utf-8') as json_file_to_write:
        json_file_to_write.write(json.dumps(error_functions, indent=4))
    json_file_to_write.close()


if __name__ == "__main__":

    ccs_root = '../../../dataset/cross-language/CodeNet_Microsoft'
    c4_root = '../../../dataset/cross-language/C4'

    python_with_func = os.path.join(ccs_root, 'dataset', 'python_with_func.jsonl')
    pair_clones = os.path.join(c4_root, 'dataset', 'pair_clones.jsonl')

    all_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'all_functions.json')
    correct_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'correct_functions.json')
    error_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'error_functions.json')

    all_programs = os.path.join(c4_root, 'preprocessed_dataset', 'all_programs.json')
    correct_programs = os.path.join(c4_root, 'preprocessed_dataset', 'correct_programs.json')
    error_programs = os.path.join(c4_root, 'preprocessed_dataset', 'error_programs.json')

    # with open(pair_clones, 'r') as f:
    #     lines = f.readlines()
    #     sample_data = json.loads(lines[2065])
    #     java_code = sample_data['src_code']
    #     pprint(java_code)
    #
    #     # python_tokenizer = PythonTokenizer()
    #     # python_tokens = python_tokenizer.tokenize_code(code=py_code)
    #     # print(python_tokens)
    # f.close()

    # with open(all_programs, 'r') as f:
    #     json_data = json.load(f)
    #     java_pool = json_data['java']
    #     java_code = java_pool['351']['22928']
    #     pprint(java_code)
    #
    #     # java_tokenizer = JavaTokenizer()
    #     # java_tokens = java_tokenizer.tokenize_code(code=java_code)
    #     # print(java_tokens)
    #     # python_tokenizer = PythonTokenizer()
    #     # python_tokens = python_tokenizer.tokenize_code(code=py_code)
    #     # print(python_tokens)
    # f.close()

    java_code = r"""class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }"""
    java_tokenizer = JavaTokenizer()
    java_tokens = java_tokenizer.tokenize_code(code=java_code)
    print(java_tokens)

    py_code = r"""print("Hello, World!")"""
    python_tokenizer = PythonTokenizer()
    python_tokens = python_tokenizer.tokenize_code(code=py_code)
    print(python_tokens)

    # pipeline_tokenization(all_functions, correct_functions, error_functions)
    # pipeline_tokenization(all_programs, correct_programs, error_programs)
