CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_mlm.py \
    --model_name_or_path /remote-home/xymou/bert/roberta-base/ \
    --train_file ./data/raw/tweet_train.txt \
    --eval_file ./data/raw/tweet_dev.txt \
    --train_disk ./data/disk/mlm_train/ \
    --eval_disk ./data/disk/mlm_dev/ \
    --output_dir ckpt/mlm_backbone \
    --eval_tokenizer ckpt/mlm_backbone \
    --num_train_epochs 10 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_seq_length 128 \
    --warmup_steps 960 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end \
    --eval_steps 200 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_dir ./logs/train_mlm \
    --logging_steps 10 \
    --do_train \
    --do_eval \
    --do_mlm \
    --fp16 \
    "$@"