#!/bin/bash
cd ..

train_filepath='../../dataset/fine-tuning/Java-Python/pair_train.jsonl'
valid_filepath='../../dataset/fine-tuning/Java-Python/pair_valid.jsonl'
test_filepath='../../dataset/fine-tuning/Java-Python/pair_test.jsonl'

saved_dir='../../checkpoint/CLONE_MODEL_'
date_now=$(date +"%Y%m%d_%H%M%S")
saved_path=${saved_dir}${date_now}

config_name='microsoft/graphcodebert-base'
model_name=''
tokenizer_type='graphcodebert'
load_model_path='../../result/Experiment/pre-train/CSN-Augmented/GraphCodeBERT/CODE_ENCODER--epoch_4.bin'

gpu=2

python fine_tuning_procedure.py --train_filepath=$train_filepath --valid_filepath=$valid_filepath --saved_dir=$saved_path --config_name_or_path=$config_name --model_name_or_path=$model_name --tokenizer_type=$tokenizer_type --load_model_path=$load_model_path --do_train --do_eval --use_cuda=True --gpu=$gpu
