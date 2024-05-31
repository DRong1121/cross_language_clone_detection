import os
import json
from pprint import pprint

csn_root = '../../../dataset/cross-language/CodeSearchNet_Microsoft/dataset'
csn_split_root = '../../../dataset/cross-language/CodeSearchNet_Microsoft/functions'

correct_functions = '../../../dataset/cross-language/CodeNet_Microsoft/preprocessed_dataset/correct_functions.json'
cn_split_root = '../../../dataset/cross-language/CodeNet_Microsoft/functions'


def split_csn_dataset(dataset_root, split_root):

    languages = ['java', 'python']
    sets = ['train.jsonl', 'valid.jsonl', 'test.jsonl']

    for language in languages:
        print('Splitting ' + language + ' functions...')
        for set_name in sets:
            set_file_path = os.path.join(dataset_root, language, set_name)
            print('Splitting set file: ' + set_file_path)
            with open(set_file_path, 'r') as input_file:
                sample_file = input_file.readlines()
                for idx in range(0, len(sample_file)):
                    sample_data = json.loads(sample_file[idx])
                    code_data = sample_data['code']

                    if language == 'java':
                        function_name = str(idx) + '.java'
                    else:
                        function_name = str(idx) + '.py'

                    function_file_path = os.path.join(split_root, language, set_name.split('.')[0], function_name)
                    with open(function_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(code_data)
                    output_file.close()
            input_file.close()


def split_cn_dataset(data_file, split_root):

    languages = ['java', 'python']

    with open(data_file, mode='r') as input_file:
        json_data = json.load(input_file)
        for language in languages:
            print('Splitting ' + language + ' functions...')
            function_dict = json_data[language]
            for task_id in function_dict.keys():
                task_dict = function_dict[task_id]
                for submit_id, code in task_dict.items():
                    if language == 'java':
                        function_name = str(submit_id) + '.java'
                    else:
                        function_name = str(submit_id) + '.py'

                    function_file_path = os.path.join(split_root, language, function_name)
                    with open(function_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(code)
                    output_file.close()
    input_file.close()


if __name__ == "__main__":
    pass
    # split_csn_dataset(dataset_root=csn_root, split_root=csn_split_root)
    # split_cn_dataset(data_file=correct_functions, split_root=cn_split_root)
