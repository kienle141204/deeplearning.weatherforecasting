!source venv/scirpts/activate 

!python run.py \
    --model SwinLSTM \
    --data_path './data/2024.csv' \
    --input_channels 7 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --is_training 1 \
    --train_epochs 100 \
    --seq_len 24 \
    --pred_len 24 \
    --lr_patience 2 \
    --input_img_size 16 \
    --early_stop_patience 5 \
    --patch_size 2