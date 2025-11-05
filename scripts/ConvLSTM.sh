!source venv/scirpts/activate 

!python run.py \
    --model ConvLSTM \
    --data_path './data/2024.csv' \
    --batch_size 32 \
    --input_channels 7 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --is_training 1 \
    --train_epochs 100 \
    --seq_len 24 \
    --pred_len 24 \
    --kernel_size 3 \
    --hidden_channels 32 64 \
    --num_layers 2 \
    --lr_patience 2 \
    --early_stop_patience 5