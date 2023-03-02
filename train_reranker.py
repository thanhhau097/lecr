import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import init_tokenizer
from model import Scorer
from model_args import ModelArguments
from rerank_dataset import LECRerankDataset, collate_fn
from rerank_engine import CustomTrainer, compute_metrics
from utils import get_processed_text_dict

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    torch.cuda.set_device(training_args.local_rank)
    torch.cuda.empty_cache()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    fold = data_args.fold

    print("Reading correlation data CSV", data_args.data_path)
    data_df = pd.read_csv(data_args.data_path)
    print("Reading topic data CSV", data_args.topic_path)
    topic_df = pd.read_csv(data_args.topic_path)
    print("Reading content data CSV", data_args.content_path)
    content_df = pd.read_csv(data_args.content_path)
    tokenizer = init_tokenizer(model_args.tokenizer_name)
    topic_dict, content_dict = get_processed_text_dict(
        topic_df, content_df, tokenizer.sep_token, True
    )
    print("Reading content embs dict ...")
    content_embs_dict = torch.load(data_args.content_embs_dict_path)
    print("Reading correlations df ... ")
    correlations_df = pd.read_csv(data_args.correlation_path)
    topics2relevantcontents = correlations_df.set_index("topic_id")["content_ids"].to_dict()

    train_topic_ids = set(data_df[data_df["fold"] != fold].topic_id.values)
    val_topic_ids = set(data_df[data_df["fold"] == fold].topic_id.values)
    train_topic_dict = {k: topic_dict.get(k) for k in train_topic_ids}
    val_topic_dict = {k: topic_dict.get(k) for k in val_topic_ids}
    train_knn_df = pd.read_csv(f"data/train_knn_fold{fold}.csv")
    val_knn_df = pd.read_csv(f"data/val_knn_fold{fold}.csv")

    train_dataset = LECRerankDataset(
        tokenizer_name=model_args.tokenizer_name,
        max_len=data_args.max_len,
        topics_dict=train_topic_dict,
        content_embs_dict=content_embs_dict,
        knn_df=train_knn_df,
        topic2relevantcontents=topics2relevantcontents,
    )
    val_dataset = LECRerankDataset(
        tokenizer_name=model_args.tokenizer_name,
        max_len=data_args.max_len,
        topics_dict=val_topic_dict,
        content_embs_dict=content_embs_dict,
        knn_df=val_knn_df,
        topic2relevantcontents=topics2relevantcontents,
    )

    # Initialize trainer
    print("Initializing model...")
    d_model = train_dataset[0][1].shape[-1]
    model = Scorer(
        d_model,
        tokenizer_name=model_args.tokenizer_name,
        model_name=model_args.model_name,
        objective=model_args.objective,
        is_sentence_transformers=model_args.is_sentence_transformers,
    )
    model = model.cuda()
    if model_args.query_resume is not None:
        model.query_encoder.load_state_dict(torch.load(model_args.query_resume, "cpu"))
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        model.load_state_dict(checkpoint)

    # device = f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)

    print("Start training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        output = trainer.predict(val_dataset, metric_key_prefix="eval")
        thresholds = np.arange(0.05, 0.2, 0.01)
        scores = []
        for threshold in thresholds:
            score = compute_metrics(output, threshold=threshold)["f2"]
            scores.append(score)
        best_threshold = thresholds[np.argmax(scores)]
        best_score = np.max(scores)
        logger.info(f"Best threshold: {best_threshold}, best score: {best_score}")


if __name__ == "__main__":
    main()
