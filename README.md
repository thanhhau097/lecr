# Training

### Sentence transformers
```
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ./outputs/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --tokenizer_name sentence-transformers/all-MiniLM-L6-v2  --model_name sentence-transformers/all-MiniLM-L6-v2 --fold 0 --dataloader_num_workers 12 --learning_rate 2e-5  --num_train_epochs 10 --per_device_train_batch_size 160 --per_device_eval_batch_size 160 --remove_unused_columns False --overwrite_output_dir --load_best_model_at_end --objective both --max_len 256 --is_sentence_transformers

CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ./outputs/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --tokenizer_name sentence-transformers/all-MiniLM-L6-v2  --model_name sentence-transformers/all-MiniLM-L6-v2 --fold 0 --dataloader_num_workers 12 --learning_rate 2e-5  --num_train_epochs 10 --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --remove_unused_columns False --overwrite_output_dir --load_best_model_at_end --objective both --max_len 128 --is_sentence_transformers
```

### xlm-roberta-base
```
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ./outputs/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --tokenizer_name xlm-roberta-base  --model_name xlm-roberta-base --fold 0 --dataloader_num_workers 12 --learning_rate 2e-5  --num_train_epochs 10 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --remove_unused_columns False --load_best_model_at_end --objective both --max_len 128
```


# TODO:
- [x] Add all positive pairs to the training set
- [x] Add pair content-content
- [x] SentenceBert Softmax loss: https://arxiv.org/pdf/1908.10084.pdf
- [x] Training with content texts
- [ ] Add parent and child topic to topic text
- [ ] Add f2 score directly to evaluation epoch
- [ ] Pretrained using translation: https://www.sbert.net/examples/training/multilingual/README.html
- [ ] Hard negatives: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873
- [ ] Try another loss function: https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss

# Generate negative data based on this:
1. https://www.kaggle.com/code/ragnar123/lecr-unsupervised-train-set-public
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline

# Resources
1. https://www.kaggle.com/code/kaizen97/sbert-fuzzy-annoy/comments
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline
3. Contrastive loss: https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
4. https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/375313
5. https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/tokenizer.py#L295