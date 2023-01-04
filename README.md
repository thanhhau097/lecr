# Training
```
python -m torch.distributed.launch --nproc_per_node 2 train.py --output_dir ./outputs/ --num_train_epochs 10 --do_train --do_eval --metric_for_best_model eval_loss --overwrite_output_dir --per_device_train_batch_size 4 --model_name xlm-roberta-base --fold 0 --dataloader_num_workers 64 --evaluation_strategy epoch --save_strategy epoch
```


# Generate negative data based on this:
1. https://www.kaggle.com/code/ragnar123/lecr-unsupervised-train-set-public
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline

# Resources
1. https://www.kaggle.com/code/kaizen97/sbert-fuzzy-annoy/comments
2. https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline
3. Contrastive loss: https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
4. https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/375313
5. https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/tokenizer.py#L295