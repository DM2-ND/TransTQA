
set dataset='techqa'

python -u ../src/run_main.py \
    --model_type albert \
    --model_name_or_path ${put pretained model path here} \
    --dataset ${dataset} \
    --fp16 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs 300 \
    --learning_rate 5.5e-6 \
    --train_file data/${dataset}/training_Q_A.json \
    --predict_file data/${dataset}/dev_Q_A.json \
    --overwrite_output_dir \
    --output_dir outputs_${dataset} \
