CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=2
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_joint_cl.py \
    --model_name_or_path ./ckpt/mlm_backbone/ \
    --train_file ./data/raw/all_train.csv \
    --eval_file ./data/raw/all_dev.csv \
    --train_disk ./data/disk/cl_train \
    --eval_disk ./data/disk/cl_dev \
    --output_dir ./result/uppam \
    --eval_tokenizer ./result/uppam \
    --num_train_epochs 10 \
    --pad_to_max_length \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --warmup_steps 270 \
    --cl_loss triplet \
    --act both \
    --leg_act general \
    --evaluation_strategy steps \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end \
    --eval_steps 50 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_dir ./logs/train_cl \
    --logging_steps 10 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"