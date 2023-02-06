import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import DatasetUpdateCallback, LECRDataset, collate_fn, init_tokenizer
from engine import CustomTrainer, compute_metrics
from model import Model
from model_args import ModelArguments
from utils import get_processed_text_dict


torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
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
    print("Reading correlation data CSV", data_args.correlation_path)
    correlation_df = pd.read_csv(data_args.correlation_path)

    if data_args.use_no_content_topics:
        train_topic_ids = set(topic_df.id.values).difference(set(data_df[data_df["fold"] == fold].topics_ids.values))
    else:
        train_topic_ids = set(data_df[data_df["fold"] != fold].topics_ids.values)
    val_topic_ids = set(data_df[data_df["fold"] == fold].topics_ids.values)

    if data_args.use_translated:
        print("Reading translated topic data CSV", data_args.translated_topic_path)
        translated_topic_df = pd.read_csv(data_args.translated_topic_path)
        print("Reading translated content data CSV", data_args.translated_content_path)
        translated_content_df = pd.read_csv(data_args.translated_content_path)
        print("Reading translated correlation data CSV", data_args.translated_correlation_path)
        translated_correlation_df = pd.read_csv(data_args.translated_correlation_path)

        # add to topic_df, content_df, correlation_df
        # about correlation_df, for training set, we add all correlations
        # but for validation set, we only keep the original ones
        topic_df = pd.concat([topic_df, translated_topic_df.drop(columns=["origin_id", "origin_parent"])], ignore_index=True)
        content_df = pd.concat([content_df, translated_content_df.drop(columns=["origin_id"])], ignore_index=True)
        
        # drop all rows that contains val topic_ids in translated_correlation_df
        train_topic_ids = set(train_topic_ids).union(set(translated_topic_df[translated_topic_df.origin_id.isin(train_topic_ids)].id.values))
        translated_correlation_df = translated_correlation_df[translated_correlation_df.topic_id.isin(train_topic_ids)]
        correlation_df = pd.concat([correlation_df, translated_correlation_df], ignore_index=True)

    tokenizer = init_tokenizer(model_args.tokenizer_name)
    topic_dict, content_dict = get_processed_text_dict(
        topic_df, content_df, tokenizer.sep_token
    )

    train_dataset = LECRDataset(
        supervised_df=data_df[data_df["fold"] != fold],
        topic_df=topic_df,
        content_df=content_df,
        topic_dict=topic_dict,
        content_dict=content_dict,
        correlation_df=correlation_df,
        tokenizer_name=model_args.tokenizer_name,
        max_len=data_args.max_len,
        use_content_pair=data_args.use_content_pair,
        is_training=True,
        use_augmentation=data_args.use_augmentation,
        objective=model_args.objective,
    )

    val_dataset = LECRDataset(
        supervised_df=data_df[data_df["fold"] == fold],
        topic_df=topic_df,
        content_df=content_df,
        topic_dict=topic_dict,
        content_dict=content_dict,
        correlation_df=correlation_df,
        tokenizer_name=model_args.tokenizer_name,
        max_len=data_args.max_len,
        use_content_pair=False,
        is_training=False,
        use_augmentation=data_args.use_augmentation,
        objective=model_args.objective,
    )

    # Initialize trainer
    print("Initializing model...")
    model = Model(
        tokenizer_name=model_args.tokenizer_name,
        model_name=model_args.model_name,
        objective=model_args.objective,
        is_sentence_transformers=model_args.is_sentence_transformers,
    )
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.model.load_state_dict(checkpoint)

        if "fc.weight" in checkpoint:
            model.fc.load_state_dict(
                {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Start training...")
    if model_args.objective == "classification" and data_args.use_sampler:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            pos_neg_ratio=data_args.pn_ratio,
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )

    if model_args.objective == "siamese":
        callback = DatasetUpdateCallback(
            trainer=trainer,
            train_topic_ids=train_topic_ids,
            val_topic_ids=val_topic_ids,
            topic_df=topic_df,
            content_df=content_df,
            topic_dict=topic_dict,
            content_dict=content_dict,
            correlation_df=correlation_df,
            tokenizer_name=model_args.tokenizer_name,
            max_len=data_args.max_len,
            best_score=0,
            top_k=data_args.top_k_neighbors,
            use_translated=data_args.use_translated,
        )
        trainer.add_callback(callback)

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     output = trainer.predict(val_dataset, metric_key_prefix="eval")
    #     metrics = output.metrics
    #     # torch.save(
    #     #     {
    #     #         "label_ids": output.label_ids[0],
    #     #         "patient_ids": output.label_ids[1],
    #     #         "laterality": output.label_ids[2],
    #     #         "predictions": output.predictions,
    #     #         "metrics": output.metrics,
    #     #     },
    #     #     os.path.join(training_args.output_dir, "output.pth"),
    #     # )
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    #     logger.info("*** Optimize MCC ***")
    #     thresholds = np.arange(0.1, 0.9, 0.01)
    #     scores = []
    #     predictions = torch.sigmoid(torch.from_numpy(output.predictions)).numpy()
    #     labels = output.label_ids
    #     for threshold in thresholds:
    #         preds = predictions
    #         preds = preds > threshold
    #         score = matthews_corrcoef(labels, preds)
    #         scores.append(score)
    #     best_threshold = thresholds[np.argmax(scores)]
    #     best_score = np.max(scores)
    #     logger.info(f"Best matthews_corrcoef: {best_score} @ {best_threshold}")


if __name__ == "__main__":
    main()
