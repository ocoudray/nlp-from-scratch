python scripts/train_mlm.py \
    --model_name custom_bert3 \
    --accumulate_grad_batches 64 \
    --log_every_n_steps 1 \
    --batch_size 8 \
    --d_model 768 \
    --max_len 512 \
    --num_heads 12 \
    --num_layers 12 \
    --offset 22250