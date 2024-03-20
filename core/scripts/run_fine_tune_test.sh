#!/bin/bash
cd ..

test_filepath='../../dataset/fine-tuning/Java-Python/pair_test.jsonl'

config_name='microsoft/graphcodebert-base'
model_name=''
tokenizer_type='graphcodebert'
load_model_path='../../result/Experiment/fine-tune/CSN-Augmented/GraphCodeBERT/CLONE_MODEL--epoch_1.bin'

threshold=0.64

gpu=2

python fine_tuning_procedure.py --test_filepath=$test_filepath --config_name_or_path=$config_name --model_name_or_path=$model_name --tokenizer_type=$tokenizer_type --load_model_path=$load_model_path --do_test --threshold=$threshold --use_cuda=True --gpu=$gpu
