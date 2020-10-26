
set dataset='askubuntu'

python -u ../src/run_main.py \
    --model_type albert \
    --model_name_or_path albert-base-v2 \
    --dataset ${dataset} \
    --fp16 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs 300 \
    --learning_rate 5.5e-6 \
    --train_file data/${dataset}/training_Q_A.raw \
    --predict_file data/${dataset}/dev_Q_A.raw \
    --overwrite_output_dir \
    --output_dir outputs_${dataset} \
