# Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ./outputs/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --model_name xlm-roberta-base --fold 0 --dataloader_num_workers 4 --learning_rate 2e-5 --num_train_epochs 5 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --remove_unused_columns False --overwrite_output_dir --load_best_model_at_end --objective both
```

# TODO:
- [x] Add all positive pairs to the training set
- [ ] Add pair content-content
- [ ] SentenceBert Softmax loss: https://arxiv.org/pdf/1908.10084.pdf

# Generate negative data based on this:
1. https://www.kaggle.com/code/ragnar123/lecr-unsupervised-train-set-public
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline

# Resources
1. https://www.kaggle.com/code/kaizen97/sbert-fuzzy-annoy/comments
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline
3. Contrastive loss: https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
4. https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/375313
5. https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/tokenizer.py#L295