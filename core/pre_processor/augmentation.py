import os
import json
import jsonlines

from core.tokenizer.java_tokenizer import JavaTokenizer
from core.tokenizer.python_tokenizer import PythonTokenizer


def construct_train_functions(input_dir, output_dir):
    # Total numbers of train pairs: 162174
    result_data = list()

    python_tokenizer = PythonTokenizer()
    java_functions = os.path.join(os.path.abspath(input_dir), 'java', 'train')
    python_functions = os.path.join(os.path.abspath(input_dir), 'python_generated_from_java', 'train')

    python_files = [os.path.join(python_functions, file) for file in os.listdir(python_functions)]
    for python_file in python_files:
        if python_file.endswith('.py'):
            idx = os.path.split(python_file)[-1].split('.')[0]
            java_file_name = str(idx) + '.java'
            java_file = os.path.join(java_functions, java_file_name)
            if os.path.exists(java_file):
                java_code = open(java_file).read().strip()
                python_code = open(python_file).read().strip()
                try:
                    python_tokens = python_tokenizer.tokenize_code(code=python_code)
                    result_item = dict()
                    result_item['task_id'] = str(int(idx) + 10000)
                    result_item['src_code'] = java_code
                    result_item['tgt_code'] = python_code
                    result_item['category'] = 'java-python'
                    result_data.append(result_item)
                except Exception as e:
                    continue

    print('Start writing CSN-Generated pair_train.jsonl...')
    print('Total numbers of train pairs: ' + str(len(result_data)))
    output_file = os.path.join(output_dir, 'pair_train.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(result_data)
    writer.close()


def construct_valid_functions(input_dir, output_dir):
    # Total numbers of valid pairs: 15913
    result_data = list()

    python_tokenizer = PythonTokenizer()

    java_valid_dir = os.path.join(os.path.abspath(input_dir), 'java', 'valid')
    python_valid_dir = os.path.join(os.path.abspath(input_dir), 'python_generated_from_java', 'valid')

    java_test_dir = os.path.join(os.path.abspath(input_dir), 'java', 'test')
    python_test_dir = os.path.join(os.path.abspath(input_dir), 'python_generated_from_java', 'test')

    python_valid_files = [os.path.join(python_valid_dir, file) for file in os.listdir(python_valid_dir)]
    for python_file in python_valid_files:
        if python_file.endswith('.py'):
            idx = os.path.split(python_file)[-1].split('.')[0]
            java_file_name = str(idx) + '.java'
            java_file = os.path.join(java_valid_dir, java_file_name)
            if os.path.exists(java_file):
                java_code = open(java_file).read().strip()
                python_code = open(python_file).read().strip()
                try:
                    python_tokens = python_tokenizer.tokenize_code(code=python_code)
                    result_item = dict()
                    result_item['task_id'] = str(int(idx) + 10000)
                    result_item['src_code'] = java_code
                    result_item['tgt_code'] = python_code
                    result_item['category'] = 'java-python'
                    result_data.append(result_item)
                except Exception as e:
                    continue

    python_test_files = [os.path.join(python_test_dir, file) for file in os.listdir(python_test_dir)]
    for python_file in python_test_files:
        if python_file.endswith('.py'):
            idx = os.path.split(python_file)[-1].split('.')[0]
            java_file_name = str(idx) + '.java'
            java_file = os.path.join(java_test_dir, java_file_name)
            if os.path.exists(java_file):
                java_code = open(java_file).read().strip()
                python_code = open(python_file).read().strip()
                try:
                    python_tokens = python_tokenizer.tokenize_code(code=python_code)
                    result_item = dict()
                    result_item['task_id'] = str(int(idx) + 20000)
                    result_item['src_code'] = java_code
                    result_item['tgt_code'] = python_code
                    result_item['category'] = 'java-python'
                    result_data.append(result_item)
                except Exception as e:
                    continue

    print('Start writing CSN-Generated pair_valid.jsonl...')
    print('Total numbers of valid pairs: ' + str(len(result_data)))
    output_file = os.path.join(output_dir, 'pair_valid.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(result_data)
    writer.close()


def merge_jsonl_files(input_dir_1, input_dir_2, output_dir):
    # Total number of merged train pairs: 291630
    # Total number of merged valid pairs: 30357

    merge_train = list()
    merge_valid = list()

    sets = ['train', 'valid']
    for set_name in sets:
        print('Start merging ' + set_name + ' jsonl files...')
        original_file_path = os.path.join(input_dir_1, 'pair_' + set_name + '.jsonl')
        with open(original_file_path, 'r') as input_file_1:
            original_file_content = input_file_1.readlines()
            for original_line in original_file_content:
                original_data = json.loads(original_line)
                if set_name == 'train':
                    merge_train.append(original_data)
                elif set_name == 'valid':
                    merge_valid.append(original_data)
        input_file_1.close()

        generated_file_path = os.path.join(input_dir_2, 'pair_' + set_name + '.jsonl')
        with open(generated_file_path, 'r') as input_file_2:
            generated_file_content = input_file_2.readlines()
            for generated_line in generated_file_content:
                generated_data = json.loads(generated_line)
                if set_name == 'train':
                    merge_train.append(generated_data)
                elif set_name == 'valid':
                    merge_valid.append(generated_data)
        input_file_2.close()

    print('Start writing CSN-Augmented pair_train.jsonl...')
    print('Total numbers of train pairs: ' + str(len(merge_train)))
    output_file = os.path.join(output_dir, 'pair_train.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(merge_train)
    writer.close()

    print('Start writing CSN-Augmented pair_valid.jsonl...')
    print('Total numbers of valid pairs: ' + str(len(merge_valid)))
    output_file = os.path.join(output_dir, 'pair_valid.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(merge_valid)
    writer.close()


if __name__ == "__main__":

    csn_functions = '../../../dataset/cross-language/CodeSearchNet_Microsoft/functions'
    csn_generated = '../../../dataset/pre-training/CSN-Generated'
    csn_augmented = '../../../dataset/pre-training/CSN-Augmented'
    original = '../../../dataset/pre-training/Java-Python'

    # construct_train_functions(input_dir=csn_functions, output_dir=csn_generated)
    # construct_valid_functions(input_dir=csn_functions, output_dir=csn_generated)
    # merge_jsonl_files(input_dir_1=original, input_dir_2=csn_generated, output_dir=csn_augmented)
