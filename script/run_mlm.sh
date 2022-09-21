#!/bin/csh
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu@@mjiang2
#$ -pe smp 1
#$ -l gpu=4


# /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/torch/bin/python -u mlm.py \
#     --output_dir output_new_techqa \
#     --model_type bert \
#     --model_name_or_path albert-base-v2 \
#     --do_train \
#     --do_eval \
#     --train_data_file /afs/crc.nd.edu/group/dmsquare/vol2/wyu1/MLM/techqa/training_corpus.raw \
#     --eval_data_file /afs/crc.nd.edu/group/dmsquare/vol2/wyu1/MLM/techqa/dev_corpus.raw \
#     --overwrite_output_dir \
#     --evaluate_during_training \
#     --num_train_epochs 20 \
#     --per_gpu_eval_batch_size=8   \
#     --per_gpu_train_batch_size=8   \
#     --save_steps 5000 \
#     --fp16 \
#     --mlm

nohup  python mlm/mlm.py \
    --output_dir output_new_techqa \
    --model_type bert \
    --model_name_or_path output_new_techqa/checkpoint-50000 \
    --do_train \
    --train_data_file ~/data/training_corpus.raw  \
    --eval_data_file ~/data/dev_corpus.raw \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --per_device_eval_batch_size=2  \
    --per_device_train_batch_size=4  \
    --save_steps 10000  \
    --fp16 \
    --mlm \
    --tokenizer_name bert-large-uncased \
    --eval_accumulation_steps=1 \
    --gradient_accumulation_steps=2 >bert_large_uncased_pretrain_3.log &