#!/bin/bash
cd ..

train_filepath='../../dataset/pre-training/CSN-Augmented/pair_train.jsonl'
valid_filepath='../../dataset/pre-training/CSN-Augmented/pair_valid.jsonl'

saved_dir='../../checkpoint/CODE_ENCODER_'
date_now=$(date +"%Y%m%d_%H%M%S")
saved_path=${saved_dir}${date_now}

model_name='microsoft/graphcodebert-base'
tokenizer_type='graphcodebert'
load_model_path='../../checkpoint/ContraBERT_C/pytorch_model.bin'

num_epochs=20
train_batch_size=16
valid_batch_size=16
learning_rate=2e-5

temperature=10

gpu=2

python pre_training_distributed.py --train_filepath=$train_filepath --valid_filepath=$valid_filepath --saved_dir=$saved_path --train_batch_size=$train_batch_size --valid_batch_size=$valid_batch_size --learning_rate=$learning_rate --config_name_or_path=$model_name --model_name_or_path=$model_name --tokenizer_type=$tokenizer_type --do_train --do_eval --use_cuda=True --gpu=$gpu
