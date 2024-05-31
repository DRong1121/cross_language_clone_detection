# pre-training dataset: zero shot code-to-code search dataset released by Microsoft, based on the CodeNet dataset
# 1. pre-process: remove comments, special tokens, documents that are not in English
# 2. construction: construct ~100,000 similar function pairs based on 'label'
# 3. splitting (train/valid)

import re
import os
import json
import jsonlines
from pprint import pprint

# define dataset root_dir
ccs_root = '../../../dataset/cross-language/CodeNet_Microsoft'
pretraining_root = '../../../dataset/pre-training'
# java comment patterns
java_single_line_pattern = r'\/{2}(?P<java_comment_line>.*)\n'
# python comment patterns
py_single_line_pattern = r'#(?P<py_comment_line>.*)\n'


def remove_java_comments_by_line(func_data):
    """
        :param func_data
        :return: func_data remove single line comments
    """
    func_lines = func_data.split('\n')
    filtered_lines = list()
    for func_line in func_lines:
        func_line += '\n'
        single_line_match_result = re.search(java_single_line_pattern, func_line)
        if single_line_match_result:
            single_line_comment = '//' + single_line_match_result.group('java_comment_line')
            replaced_line = func_line.replace(single_line_comment, '')
            if not replaced_line.strip('\n').strip('\t').strip() == '':
                filtered_lines.append(replaced_line)
        else:
            filtered_lines.append(func_line)
    return ''.join(filtered_lines)


def remove_java_comments_by_func(func_data):
    """
        :param func_data
        :return: func_data remove multiple lines comments
    """
    func_lines = func_data.split('\n')
    comment_list = list()
    start_idx = 0
    while start_idx < len(func_lines):
        start_line = func_lines[start_idx]
        striped_start_line = start_line.strip('\n').strip('\t').strip()
        if striped_start_line.startswith('/*'):
            comment_item = dict()
            comment_item['start_idx'] = start_idx
            if striped_start_line.endswith('*/'):
                comment_item['end_idx'] = start_idx
                comment_list.append(comment_item)
                start_idx += 1
            else:
                start_idx += 1
                for end_idx in range(start_idx, len(func_lines)):
                    end_line = func_lines[end_idx]
                    striped_end_line = end_line.strip('\n').strip('\t').strip()
                    if striped_end_line.endswith('*/'):
                        comment_item['end_idx'] = end_idx
                        comment_list.append(comment_item)
                        start_idx = end_idx + 1
                        break
        else:
            start_idx += 1

    filtered_func_data = list()
    start = 0
    for i in range(0, len(comment_list)):
        end = comment_list[i]['start_idx']
        filtered_func_data.extend(func_lines[start: end])
        start = comment_list[i]['end_idx'] + 1
    filtered_func_data.extend(func_lines[start:])

    result_func_data = list()
    for filtered_func_line in filtered_func_data:
        if not filtered_func_line.strip('\n').strip('\t').strip() == '':
            result_func_data.append(filtered_func_line)
    return '\n'.join(result_func_data)


def preprocess_java_with_func(input_dir, output_file_path):
    java_preprocess = list()

    with open(input_dir, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            preprocessed_data = dict()
            sample_data = json.loads(line)
            func_data = remove_java_comments_by_line(sample_data['func'])
            func_data = remove_java_comments_by_func(func_data)
            preprocessed_data['func'] = func_data
            preprocessed_data['index'] = sample_data['index']
            preprocessed_data['label'] = sample_data['label']
            java_preprocess.append(preprocessed_data)
    input_file.close()

    print('Start writing preprocessed java_with_func.jsonl...')
    with jsonlines.open(output_file_path, 'w') as output_file:
        output_file.write_all(java_preprocess)
    output_file.close()


def remove_py_comments_by_line(func_data):
    """
        :param func_data
        :return: func_data remove single line comments
    """
    func_lines = func_data.split('\n')
    filtered_lines = list()
    for func_line in func_lines:
        func_line += '\n'
        if '"#"' not in func_line and '\'#\'' not in func_line:
            single_line_match_result = re.search(py_single_line_pattern, func_line)
            if single_line_match_result:
                py_comment_line = single_line_match_result.group('py_comment_line')
                # print(py_comment_line)
                single_line_comment = '#' + py_comment_line
                replaced_line = func_line.replace(single_line_comment, '')
                if not replaced_line.strip('\n').strip() == '':
                    filtered_lines.append(replaced_line)
            else:
                filtered_lines.append(func_line)
        else:
            filtered_lines.append(func_line)
    return ''.join(filtered_lines)


def remove_py_comments_by_func(func_data):
    """
        :param func_data
        :return: func_data remove multiple lines comments
    """
    func_lines = func_data.split('\n')
    comment_list = list()
    start_idx = 0
    while start_idx < len(func_lines):
        start_line = func_lines[start_idx]
        striped_start_line = start_line.strip('\n').strip()
        if striped_start_line.startswith('"""'):
            comment_item = dict()
            comment_item['start_idx'] = start_idx
            if striped_start_line.endswith('"""') and striped_start_line != '"""':
                comment_item['end_idx'] = start_idx
                comment_list.append(comment_item)
                start_idx += 1
            else:
                start_idx += 1
                for end_idx in range(start_idx, len(func_lines)):
                    end_line = func_lines[end_idx]
                    striped_end_line = end_line.strip('\n').strip()
                    if striped_end_line.endswith('"""'):
                        comment_item['end_idx'] = end_idx
                        comment_list.append(comment_item)
                        start_idx = end_idx + 1
                        break
        elif striped_start_line.startswith('\'\'\''):
            comment_item = dict()
            comment_item['start_idx'] = start_idx
            if striped_start_line.endswith('\'\'\'') and striped_start_line != '\'\'\'':
                comment_item['end_idx'] = start_idx
                comment_list.append(comment_item)
                start_idx += 1
            else:
                start_idx += 1
                for end_idx in range(start_idx, len(func_lines)):
                    end_line = func_lines[end_idx]
                    striped_end_line = end_line.strip('\n').strip()
                    if striped_end_line.endswith('\'\'\''):
                        comment_item['end_idx'] = end_idx
                        comment_list.append(comment_item)
                        start_idx = end_idx + 1
                        break
        else:
            start_idx += 1

    filtered_func_data = list()
    start = 0
    for i in range(0, len(comment_list)):
        end = comment_list[i]['start_idx']
        filtered_func_data.extend(func_lines[start: end])
        start = comment_list[i]['end_idx'] + 1
    filtered_func_data.extend(func_lines[start:])

    result_func_data = list()
    for filtered_func_line in filtered_func_data:
        if not filtered_func_line.strip('\n').strip() == '':
            result_func_data.append(filtered_func_line)
    return '\n'.join(result_func_data)


def preprocess_py_with_func(input_dir, output_file_path):
    py_preprocess = list()

    with open(input_dir, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            preprocessed_data = dict()
            sample_data = json.loads(line)
            func_data = remove_py_comments_by_line(sample_data['func'])
            func_data = remove_py_comments_by_func(func_data)
            preprocessed_data['func'] = func_data
            preprocessed_data['index'] = sample_data['index']
            preprocessed_data['label'] = sample_data['label']
            py_preprocess.append(preprocessed_data)
    input_file.close()

    print('Start writing preprocessed python_with_func.jsonl...')
    with jsonlines.open(output_file_path, 'w') as output_file:
        output_file.write_all(py_preprocess)
    output_file.close()


def construct_pair_functions(input_file_path, output_file_path):
    result_data = list()
    with open(input_file_path, mode='r', encoding='utf-8') as input_file:
        json_data = json.load(input_file)
        src_language = 'java'
        tgt_language = 'python'
        assert json_data[src_language].keys() == json_data[tgt_language].keys()
        tasks = json_data[src_language].keys()
        for task_id in tasks:
            src_pool = json_data[src_language][task_id]
            tgt_pool = json_data[tgt_language][task_id]
            for src_submit_id, src_func in src_pool.items():
                for tgt_submit_id, tgt_func in tgt_pool.items():
                    result_item = dict()
                    result_item['task_id'] = task_id
                    result_item['src_id'] = src_submit_id
                    result_item['src_code'] = src_func
                    result_item['tgt_id'] = tgt_submit_id
                    result_item['tgt_code'] = tgt_func
                    result_item['category'] = src_language + '-' + tgt_language
                    result_data.append(result_item)
    input_file.close()

    print('Start writing Java-Python pair_functions.jsonl...')
    print('Total numbers of training pairs: ' + str(len(result_data)))
    with jsonlines.open(output_file_path, 'w') as writer:
        writer.write_all(result_data)
    writer.close()


def split_pair_functions(input_file_path, train_file_path, eval_file_path):
    # pair_nums: 143900 in total
    # line_idx: 129456 for splitting

    train_examples = list()
    eval_examples = list()

    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
        train_lines = lines[0: 129456]
        for train_line in train_lines:
            train_sample = json.loads(train_line)
            train_item = dict()
            train_item['task_id'] = train_sample['task_id']
            train_item['src_id'] = train_sample['src_id']
            train_item['src_code'] = train_sample['src_code']
            train_item['tgt_id'] = train_sample['tgt_id']
            train_item['tgt_code'] = train_sample['tgt_code']
            train_item['category'] = train_sample['category']
            train_examples.append(train_item)

        eval_lines = lines[129456:]
        for eval_line in eval_lines:
            eval_sample = json.loads(eval_line)
            eval_item = dict()
            eval_item['task_id'] = eval_sample['task_id']
            eval_item['src_id'] = eval_sample['src_id']
            eval_item['src_code'] = eval_sample['src_code']
            eval_item['tgt_id'] = eval_sample['tgt_id']
            eval_item['tgt_code'] = eval_sample['tgt_code']
            eval_item['category'] = eval_sample['category']
            eval_examples.append(eval_item)
    input_file.close()

    print('Start writing Java-Python pair_train.jsonl...')
    print('Total numbers of training pairs: ' + str(len(train_examples)))
    with jsonlines.open(train_file_path, 'w') as writer:
        writer.write_all(train_examples)
    writer.close()

    print('Start writing Java-Python pair_valid.jsonl...')
    print('Total numbers of validation pairs: ' + str(len(eval_examples)))
    with jsonlines.open(eval_file_path, 'w') as writer:
        writer.write_all(eval_examples)
    writer.close()


if __name__ == "__main__":

    language = 'python'
    language_file = language + '_with_func.jsonl'
    language_original_file = os.path.join(ccs_root, 'dataset', language_file)
    language_preprocessed_file = os.path.join(ccs_root, 'preprocessed_dataset', language_file)

    # with open(language_original_file, 'r') as f:
    #     lines = f.readlines()
    #     sample_data = json.loads(lines[14495])
    #     # print(sample_data['func'])
    #     func_data = remove_py_comments_by_line(sample_data['func'])
    #     # print(func_data)
    #     func_data = remove_py_comments_by_func(func_data)
    #     print(func_data)
    # f.close()

    # preprocess_java_with_func(language_original_file, language_preprocessed_file)
    # preprocess_py_with_func(language_original_file, language_preprocessed_file)

    correct_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'correct_functions.json')
    pair_functions = os.path.join(ccs_root, 'preprocessed_dataset', 'pair_functions.jsonl')
    train_functions = os.path.join(pretraining_root, 'Java-Python', 'pair_train.jsonl')
    valid_functions = os.path.join(pretraining_root, 'Java-Python', 'pair_valid.jsonl')

    # construct_pair_functions(correct_functions, pair_functions)
    # split_pair_functions(pair_functions, train_functions, valid_functions)
