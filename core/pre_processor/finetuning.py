# fine-tuning dataset: AtCoder and Google CodeJam datasets released by C4
# 1. pre-process: remove comments, special tokens, documents that are not in English
# 2. construction: construct 40,863 clone/non-clone program pairs based on 'task_id'
# 3. splitting (train/valid/test)

import os
import json
import jsonlines
import numpy as np
from pprint import pprint

from pretraining import (remove_java_comments_by_line, remove_java_comments_by_func,
                         remove_py_comments_by_line, remove_py_comments_by_func)

c4_root = '../../../dataset/cross-language/C4'
finetuning_root = '../../../dataset/fine-tuning'

dirty_str = 'Note: ./Main.java uses unchecked or unsafe operations.\nNote: Recompile with -Xlint:unchecked for details.'


def preprocess_pair_programs(input_file_path, output_file_path):

    result_data = list()

    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            sample_data = json.loads(line)
            java_code = remove_java_comments_by_line(sample_data['src_code'])
            java_code = remove_java_comments_by_func(java_code)
            java_code = java_code.replace('\\\'', '\'')
            if dirty_str in java_code:
                java_code = java_code.replace(dirty_str, '').strip()
            if java_code.endswith('\'') or java_code.endswith('"'):
                java_code = java_code[:-1].strip()
            sample_data['src_code'] = java_code

            py_code = remove_py_comments_by_line(sample_data['tgt_code'])
            py_code = remove_py_comments_by_func(py_code)
            if py_code.endswith('\'') or py_code.endswith('"'):
                py_code = py_code[:-1].strip()
            sample_data['tgt_code'] = py_code

            result_data.append(sample_data)
    input_file.close()

    print('Start writing Java-Python preprocessed pair_clones.jsonl...')
    print('Total numbers of program pairs: ' + str(len(result_data)))
    with jsonlines.open(output_file_path, 'w') as output_file:
        output_file.write_all(result_data)
    output_file.close()


def construct_clone_pair_programs(input_file_path, output_file_path):
    result_data = list()
    with open(input_file_path, mode='r', encoding='utf-8') as input_file:
        json_data = json.load(input_file)
        src_language = 'java'
        tgt_language = 'python'
        # assert json_data[src_language].keys() == json_data[tgt_language].keys()
        tasks = json_data[tgt_language].keys()
        for task_id in tasks:
            try:
                src_pool = json_data[src_language][task_id]
            except KeyError:
                continue
            tgt_pool = json_data[tgt_language][task_id]
            for tgt_submit_id, tgt_func in tgt_pool.items():
                for src_submit_id, src_func in src_pool.items():
                    result_item = dict()
                    result_item['task_id'] = task_id
                    result_item['src_id'] = src_submit_id
                    result_item['src_code'] = src_func
                    result_item['tgt_id'] = tgt_submit_id
                    result_item['tgt_code'] = tgt_func
                    result_item['category'] = src_language + '-' + tgt_language
                    result_data.append(result_item)
    input_file.close()

    print('Start writing Java-Python pair_programs.jsonl...')
    print('Total numbers of program pairs: ' + str(len(result_data)))
    with jsonlines.open(output_file_path, 'w') as writer:
        writer.write_all(result_data)
    writer.close()


def construct_non_clone_pair_programs(input_file_path, output_file_path,
                                      select_per_num, select_total_num):
    # java tasks: 989 out of 993 (804/97/88), python tasks: 989 out of 1007 (804/97/88)
    # java solutions: 5654 out of 5660 (4572/560/522), python solutions: 6211 out of 6239 (4994/640/577)

    non_clone_pairs = list()
    result_pairs = list()

    with open(input_file_path, mode='r', encoding='utf-8') as json_file_to_read:
        json_data = json.load(json_file_to_read)
        assert json_data['java'].keys() == json_data['python'].keys()
        tasks = json_data['java'].keys()
        java_programs = json_data['java']
        python_programs = json_data['python']

        for candidate_task in tasks:
            candidate_programs = java_programs[candidate_task]
            for src_id, src_code in candidate_programs.items():
                item_list = list()
                for potential_task in tasks:
                    if potential_task != candidate_task:
                        potential_programs = python_programs[potential_task]
                        for tgt_id, tgt_code in potential_programs.items():
                            item = dict()
                            item['tgt_id'] = tgt_id
                            item['tgt_code'] = tgt_code
                            item_list.append(item)

                index_list = np.arange(len(item_list))
                np.random.shuffle(index_list)
                select_list = index_list[0: select_per_num]
                for select_index in select_list:
                    select_program = item_list[select_index]
                    non_clone_item = dict()
                    non_clone_item['task_id'] = [candidate_task, potential_task]
                    non_clone_item['src_id'] = src_id
                    non_clone_item['src_code'] = src_code
                    non_clone_item['tgt_id'] = select_program['tgt_id']
                    non_clone_item['tgt_code'] = select_program['tgt_code']
                    non_clone_item['category'] = 'java-python'
                    non_clone_pairs.append(non_clone_item)

        final_index_list = np.arange(len(non_clone_pairs))
        np.random.shuffle(final_index_list)
        for final_select_index in final_index_list[0: select_total_num]:
            final_select_item = non_clone_pairs[final_select_index]
            result_pairs.append(final_select_item)
    json_file_to_read.close()

    print('Start writing Java-Python preprocessed pair_non_clones.jsonl...')
    print('Total numbers of program pairs: ' + str(len(result_pairs)))
    with jsonlines.open(output_file_path, 'w') as output_file:
        output_file.write_all(result_pairs)
    output_file.close()


def split_pair_programs(input_file_pos,
                        train_file_path, eval_file_path, test_file_path,
                        train_file_neg=None, valid_file_neg=None, test_file_neg=None):
    # train: eval: test splitting based on the pair_clone_programs.jsonl file
    # train set clone pairs: [0: 32657)
    # valid set clone pairs: [32657: 36763)
    # test set clone pairs: [36763, 40863)

    # total number of train examples: 65314, 32657 each
    # total number of valid examples: 8212, 4106 each
    # total number of test examples: 8200, 4100 each

    train_examples = list()
    valid_examples = list()
    test_examples = list()

    with open(input_file_pos, 'r') as pair_clone_file:
        pos_lines = pair_clone_file.readlines()
        train_lines = pos_lines[0:32657]
        valid_lines = pos_lines[32657:36763]
        test_lines = pos_lines[36763:]
        for train_line in train_lines:
            train_sample = json.loads(train_line)
            item = dict()
            item['src_id'] = train_sample['src_id']
            item['src_code'] = train_sample['src_code']
            item['tgt_id'] = train_sample['tgt_id']
            item['tgt_code'] = train_sample['tgt_code']
            item['category'] = train_sample['category']
            # item['label'] = 1
            item['task_id'] = train_sample['task_id']
            train_examples.append(item)
        for valid_line in valid_lines:
            valid_sample = json.loads(valid_line)
            item = dict()
            item['src_id'] = valid_sample['src_id']
            item['src_code'] = valid_sample['src_code']
            item['tgt_id'] = valid_sample['tgt_id']
            item['tgt_code'] = valid_sample['tgt_code']
            item['category'] = valid_sample['category']
            # item['label'] = 1
            item['task_id'] = valid_sample['task_id']
            valid_examples.append(item)
        for test_line in test_lines:
            test_sample = json.loads(test_line)
            item = dict()
            item['src_id'] = test_sample['src_id']
            item['src_code'] = test_sample['src_code']
            item['tgt_id'] = test_sample['tgt_id']
            item['tgt_code'] = test_sample['tgt_code']
            item['category'] = test_sample['category']
            # item['label'] = 1
            item['task_id'] = test_sample['task_id']
            test_examples.append(item)
    pair_clone_file.close()

    # with open(train_file_neg, 'r') as pair_non_clone_train:
    #     neg_lines = pair_non_clone_train.readlines()
    #     for neg_line in neg_lines:
    #         train_sample = json.loads(neg_line)
    #         item = dict()
    #         item['src_id'] = train_sample['src_id']
    #         item['src_code'] = train_sample['src_code']
    #         item['tgt_id'] = train_sample['tgt_id']
    #         item['tgt_code'] = train_sample['tgt_code']
    #         item['category'] = train_sample['category']
    #         item['label'] = 0
    #         train_examples.append(item)
    # pair_non_clone_train.close()

    # with open(valid_file_neg, 'r') as pair_non_clone_valid:
    #     neg_lines = pair_non_clone_valid.readlines()
    #     for neg_line in neg_lines:
    #         valid_sample = json.loads(neg_line)
    #         item = dict()
    #         item['src_id'] = valid_sample['src_id']
    #         item['src_code'] = valid_sample['src_code']
    #         item['tgt_id'] = valid_sample['tgt_id']
    #         item['tgt_code'] = valid_sample['tgt_code']
    #         item['category'] = valid_sample['category']
    #         item['label'] = 0
    #         valid_examples.append(item)
    # pair_non_clone_valid.close()

    # with open(test_file_neg, 'r') as pair_non_clone_test:
    #     neg_lines = pair_non_clone_test.readlines()
    #     for neg_line in neg_lines:
    #         test_sample = json.loads(neg_line)
    #         item = dict()
    #         item['src_id'] = test_sample['src_id']
    #         item['src_code'] = test_sample['src_code']
    #         item['tgt_id'] = test_sample['tgt_id']
    #         item['tgt_code'] = test_sample['tgt_code']
    #         item['category'] = test_sample['category']
    #         item['label'] = 0
    #         test_examples.append(item)
    # pair_non_clone_test.close()

    print('Start writing Java-Python preprocessed pair_train.jsonl...')
    print('Total numbers of program pairs: ' + str(len(train_examples)))
    with jsonlines.open(train_file_path, 'w') as output_file:
        output_file.write_all(train_examples)
    output_file.close()

    print('Start writing Java-Python preprocessed pair_valid.jsonl...')
    print('Total numbers of program pairs: ' + str(len(valid_examples)))
    with jsonlines.open(eval_file_path, 'w') as output_file:
        output_file.write_all(valid_examples)
    output_file.close()

    print('Start writing Java-Python preprocessed pair_test.jsonl...')
    print('Total numbers of program pairs: ' + str(len(test_examples)))
    with jsonlines.open(test_file_path, 'w') as output_file:
        output_file.write_all(test_examples)
    output_file.close()


if __name__ == "__main__":

    original_clones = os.path.join(c4_root, 'dataset', 'pair_clones.jsonl')
    preprocessed_clones = os.path.join(c4_root, 'preprocessed_dataset', 'pair_clones.jsonl')

    correct_programs = os.path.join(c4_root, 'preprocessed_dataset', 'correct_programs.json')
    pair_clone_programs = os.path.join(c4_root, 'preprocessed_dataset', 'pair_clone_programs.jsonl')
    pair_clones_train = os.path.join(finetuning_root, 'C4', 'pair_train.jsonl')
    pair_clones_valid = os.path.join(finetuning_root, 'C4', 'pair_valid.jsonl')
    pair_clones_test = os.path.join(finetuning_root, 'C4', 'pair_test.jsonl')

    train_pool = os.path.join(c4_root, 'preprocessed_dataset', 'train_programs.json')
    valid_pool = os.path.join(c4_root, 'preprocessed_dataset', 'valid_programs.json')
    test_pool = os.path.join(c4_root, 'preprocessed_dataset', 'test_programs.json')
    pair_non_clones_train = os.path.join(c4_root, 'preprocessed_dataset',
                                         'pair_non_clone_programs_train.jsonl')
    pair_non_clones_valid = os.path.join(c4_root, 'preprocessed_dataset',
                                         'pair_non_clone_programs_valid.jsonl')
    pair_non_clones_test = os.path.join(c4_root, 'preprocessed_dataset',
                                        'pair_non_clone_programs_test.jsonl')

    train_programs = os.path.join(finetuning_root, 'Java-Python', 'pair_train.jsonl')
    valid_programs = os.path.join(finetuning_root, 'Java-Python', 'pair_valid.jsonl')
    test_programs = os.path.join(finetuning_root, 'Java-Python', 'pair_test.jsonl')

    # preprocess_pair_programs(original_clones, preprocessed_clones)

    # construct clone program pairs
    # construct_clone_pair_programs(correct_programs, pair_clone_programs)

    # construct non-clone program pairs for the training set
    # construct_non_clone_pair_programs(train_pool, pair_non_clones_train,
    #                                   8, 32657)
    # construct non-clone program pairs for the validation set
    # construct_non_clone_pair_programs(valid_pool, pair_non_clones_valid,
    #                                   8, 4106)
    # construct non-clone program pairs for the test set
    # construct_non_clone_pair_programs(test_pool, pair_non_clones_test,
    #                                   8, 4100)

    # split the fine-tuning dataset
    # split_pair_programs(pair_clone_programs,
    #                     pair_clones_train, pair_clones_valid, pair_clones_test)
