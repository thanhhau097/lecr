import multiprocessing
import subprocess


def run_generate_data(i):
    bs = 128
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={i} python train.py --output_dir ./outputs_fold_{i}/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --logging_strategy steps --logging_steps 200 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --tokenizer_name sentence-transformers/all-roberta-large-v1  --model_name sentence-transformers/all-roberta-large-v1 --fold {i} --dataloader_num_workers 32 --learning_rate 1e-5  --num_train_epochs 50 --per_device_train_batch_size {bs} --per_device_eval_batch_size {bs} --remove_unused_columns False --load_best_model_at_end --objective siamese --max_len 128 --top_k_neighbors 50 --is_sentence_transformers --gradient_accumulation_steps 4",
        shell=True,
    )


if __name__ == "__main__":
    num_processes = 8
    with multiprocessing.Pool(num_processes) as p:
        p.map(
            run_generate_data,
            list(range(num_processes)),
        )
