#!/usr/bin/env bash
model_info=$1
n_gpu=$2
epoch=$3

echo "DEBUG: \$1 = $1"
echo "DEBUG: \$2 = $2"
echo "DEBUG: \$3 = $3"


git clone https://huggingface.co/t5-$model_info

export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Dataset: set (--nq 0 -- trivia 1) or (--nq 1 -- trivia 0)
python main.py --decode_embedding 2 --n_gpu $n_gpu --mode train --query_type gtq_doc_aug_qg --adaptor_layer_num 4 \
--model_info $model_info --train_batch_size 1 --eval_batch_size 1 --test1000 0 --dropout_rate 0.1 --Rdrop 0.15 \
--adaptor_decode 1 --adaptor_efficient 1 --aug_query 1 --aug_query_type corrupted_query --input_dropout 1 --id_class bert_k30_c30_1 \
--kary 30 --output_vocab_size 30 --doc_length 64 --denoising 0 --max_output_length 10 \
--trivia 0 --nq 1 --model_name_or_path ./t5- --num_train_epochs $epoch 
#--resume_from_checkpoint '' 
