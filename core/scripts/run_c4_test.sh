#!/bin/bash
cd ..

test_filepath='../../dataset/fine-tuning/C4/pair_test.jsonl'

config_name='microsoft/graphcodebert-base'
model_name=''
tokenizer_type='graphcodebert'
load_model_path='../../result/Baseline/C4/GraphCodeBERT/C4_MODEL--epoch_6.bin'

threshold=0.61

gpu=0

python c4_distributed.py --test_filepath=$test_filepath --config_name_or_path=$config_name --model_name_or_path=$model_name --tokenizer_type=$tokenizer_type --load_model_path=$load_model_path --do_test --threshold=$threshold --use_cuda=True --gpu=$gpu
