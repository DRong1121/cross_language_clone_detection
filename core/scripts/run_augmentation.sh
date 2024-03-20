#!/bin/bash
cd ..

data_path='../../dataset/cross-language/CodeSearchNet_Microsoft/functions/java/train'
generated_path='../../dataset/cross-language/CodeSearchNet_Microsoft/functions/python_generated_from_java/train'
start_index=57562

model_path='../../checkpoint/Transcoder/translator_transcoder_size_from_DOBF.pth'
use_gpu=False

src_lang='java'
tgt_lang='python'

python -m transcoder.pipeline --data_path=$data_path --generated_path=$generated_path --start_index=$start_index --model_path=$model_path --gpu=$use_gpu --src_lang=$src_lang --tgt_lang=$tgt_lang
