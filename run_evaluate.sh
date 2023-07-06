CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model_name_or_path ./ckpt/uppam/ \
    --eval_tokenizer ./ckpt/uppam/ \
    --task_path ./PoliEval/data/ \
    --eval_lr 1e-5 \
    --eval_seed 42 \
    --eval_max_len 256 \
    --output_dir result/tmp \
    --do_eval \
    --eval_tasks "LEG_BIAS_cong,PUB_STANCE_poldeb" \
    --fp16 \
    "$@"