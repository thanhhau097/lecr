import gc

import cupy as cp
import pandas as pd
import torch
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformers import TrainerCallback

from csv_samplers import build_new_supervised_df, build_triplet_df
from dataset import InferenceDataset
from tokenizer import init_tokenizer
from utils import f2_score, get_pos_score


class DatasetUpdateCallback(TrainerCallback):
    """
    Trigger re-computing dataset

    A hack that modifies the train/val dataset, pointed by Trainer's dataloader

    0. Calculate new train/val topic/content embeddings, train KNN, get new top-k
    1. Calculate top-k max positive score, compare to current val best, if greater, continue to step 2, else do nothing
    2. Update supervised_df and update dataset:
        self.topic_texts, self.content_texts, self.labels = self.process_csv()
    """

    def __init__(
        self,
        trainer,
        train_topic_ids,
        val_topic_ids,
        topic_df,
        content_df,
        topic_dict,
        content_dict,
        correlation_df: pd.DataFrame,
        tokenizer_name: str,
        max_len: int,
        best_score=0,
        top_k=50,
        use_translated=False,
        use_triplets=False,
        reduce_negatives=False,
    ):
        super(DatasetUpdateCallback, self).__init__()
        self.trainer = trainer
        self.topic_df = topic_df
        self.content_df = content_df
        self.correlation_df = correlation_df
        self.best_score = best_score
        self.top_k = top_k
        self.use_translated = use_translated
        self.use_triplets = use_triplets
        self.reduce_negatives = reduce_negatives

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.topic_dict, self.content_dict = topic_dict, content_dict

        train_topic_texts = [
            topic_dict[topic_id]
            for topic_id in self.topic_df.id.values
            if topic_id in train_topic_ids
        ]
        self.train_topic_ids = [
            topic_id for topic_id in self.topic_df.id.values if topic_id in train_topic_ids
        ]
        self.train_topic_languages = []
        for topic_id, topic_lang in zip(self.topic_df.id.values, self.topic_df.language.values):
            if topic_id in train_topic_ids:
                self.train_topic_languages.append(topic_lang)

        val_topic_texts = [
            topic_dict[topic_id]
            for topic_id in self.topic_df.id.values
            if topic_id in val_topic_ids
        ]
        self.val_topic_ids = [
            topic_id for topic_id in self.topic_df.id.values if topic_id in val_topic_ids
        ]

        content_texts = [
            content_dict[content_id]
            for content_id in self.content_df.id.values
            if content_id.startswith("c_")
        ]

        def inference_collate_fn(inputs):
            inputs = default_collate(inputs)
            mask_len = int(inputs["attention_mask"].sum(axis=1).max())
            for k, v in inputs.items():
                inputs[k] = inputs[k][:, :mask_len]

            return inputs

        train_topic_dataset = InferenceDataset(
            texts=train_topic_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.train_topic_dataloader = DataLoader(
            train_topic_dataset,
            num_workers=16,
            batch_size=trainer.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        val_topic_dataset = InferenceDataset(
            texts=val_topic_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.val_topic_dataloader = DataLoader(
            val_topic_dataset,
            num_workers=16,
            batch_size=trainer.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        content_dataset = InferenceDataset(
            texts=content_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.content_dataloader = DataLoader(
            content_dataset,
            num_workers=16,
            batch_size=trainer.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.on_epoch_end(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        topic_embs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for inputs in tqdm(self.val_topic_dataloader):
            for k, v in inputs.items():
                inputs[k] = inputs[k].to(device)
            out = self.trainer.model.feature(inputs)
            topic_embs.extend(out.cpu().detach().numpy())

        content_embs = []
        # TODO: only use original content embeddings to avoid translation confusing
        for inputs in tqdm(self.content_dataloader):
            for k, v in inputs.items():
                inputs[k] = inputs[k].to(device)
            out = self.trainer.model.feature(inputs)
            content_embs.extend(out.cpu().detach().numpy())

        # Transfer predictions to gpu
        topic_embs_gpu = cp.array(topic_embs)
        content_embs_gpu = cp.array(content_embs)

        # Release memory
        torch.cuda.empty_cache()

        # KNN model
        content_idx_to_id = {}
        for idx, row in self.content_df.iterrows():
            content_idx_to_id[idx] = row.id

        print("Evaluating current score...")
        if self.use_translated:
            # get 500 nearest contents, then select top k contents that is in original contents, just approximate, can't check all
            original_indices = [  # indices of original contents in self.content_df
                i
                for i, emb in enumerate(content_embs)
                if self.content_df.id.values[i].startswith("c_")
            ]
            # original_content_embs = [
            #     emb
            #     for i, emb in enumerate(content_embs)
            #     if self.content_df.id.values[i].startswith("c_")
            # ]
            # original_content_embs_gpu = cp.array(original_content_embs)
            original_content_embs_gpu = content_embs_gpu

            neighbors_model = NearestNeighbors(n_neighbors=500, metric="cosine")
            neighbors_model.fit(original_content_embs_gpu)

            indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
            for selected_k in [5, 10, 20, 50, 100, 200]:
                predictions = []
                for k in tqdm(range(len(indices))):
                    pred = indices[k]
                    # original_contents = [self.content_df.loc[ind, "id"] for ind in pred.get() if self.content_df.loc[ind, "id"].startswith("c_")]
                    # original_contents = [content_idx_to_id[ind] for ind in pred.get() if content_idx_to_id[ind].startswith("c_")]
                    original_contents = [
                        content_idx_to_id[original_indices[ind]] for ind in pred.get()
                    ]
                    p = " ".join(original_contents[:selected_k])
                    predictions.append(p)

                knn_preds = pd.DataFrame(
                    {"topic_id": self.val_topic_ids, "content_ids": predictions}
                ).sort_values("topic_id")

                gt = self.correlation_df[
                    self.correlation_df.topic_id.isin(self.val_topic_ids)
                ].sort_values("topic_id")
                score = get_pos_score(
                    gt["content_ids"],
                    knn_preds.sort_values("topic_id")["content_ids"],
                    selected_k,
                )
                print(
                    "Selecting",
                    selected_k,
                    "nearest contents",
                    "top-k score =",
                    f2_score(
                        gt["content_ids"],
                        knn_preds.sort_values("topic_id")["content_ids"],
                    ),
                    "max positive score =",
                    score,
                )

            print("Training KNN model...")
            print("Generating KNN predictions with top_k =", self.top_k)
            neighbors_model = NearestNeighbors(n_neighbors=self.top_k, metric="cosine")
            neighbors_model.fit(original_content_embs_gpu)

            print("Generating embedding for validation topics")
            indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
            predictions = []
            for k in tqdm(range(len(indices))):
                pred = indices[k]
                # original_contents = [self.content_df.loc[ind, "id"] for ind in pred.get() if self.content_df.loc[ind, "id"].startswith("c_")]
                # original_contents = [content_idx_to_id[ind] for ind in pred.get() if content_idx_to_id[ind].startswith("c_")]
                original_contents = [
                    content_idx_to_id[original_indices[ind]] for ind in pred.get()
                ]
                p = " ".join(original_contents[: self.top_k])
                predictions.append(p)
        else:
            for selected_k in [5, 10, 20, 50, 100, 200]:
                neighbors_model = NearestNeighbors(n_neighbors=selected_k, metric="cosine")
                neighbors_model.fit(content_embs_gpu)

                indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
                predictions = []
                for k in tqdm(range(len(indices))):
                    pred = indices[k]
                    # p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
                    p = " ".join([content_idx_to_id[ind] for ind in pred.get()])
                    predictions.append(p)

                knn_preds = pd.DataFrame(
                    {"topic_id": self.val_topic_ids, "content_ids": predictions}
                ).sort_values("topic_id")

                gt = self.correlation_df[
                    self.correlation_df.topic_id.isin(self.val_topic_ids)
                ].sort_values("topic_id")
                score = get_pos_score(
                    gt["content_ids"],
                    knn_preds.sort_values("topic_id")["content_ids"],
                    selected_k,
                )
                print(
                    "Selecting",
                    selected_k,
                    "nearest contents",
                    "top-k score =",
                    f2_score(
                        gt["content_ids"],
                        knn_preds.sort_values("topic_id")["content_ids"],
                    ),
                    "max positive score =",
                    score,
                )

            print("Training KNN model...")
            print("Generating KNN predictions with top_k =", self.top_k)
            neighbors_model = NearestNeighbors(n_neighbors=self.top_k, metric="cosine")
            neighbors_model.fit(content_embs_gpu)

            print("Generating embedding for validation topics")
            indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
            predictions = []
            for k in tqdm(range(len(indices))):
                pred = indices[k]
                # p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
                p = " ".join([content_idx_to_id[ind] for ind in pred.get()])
                predictions.append(p)

        knn_preds = pd.DataFrame(
            {"topic_id": self.val_topic_ids, "content_ids": predictions}
        ).sort_values("topic_id")

        score = get_pos_score(
            gt["content_ids"],
            knn_preds.sort_values("topic_id")["content_ids"],
            self.top_k,
        )
        print("Current Score:", score, "Best Score:", self.best_score)

        if score > self.best_score:
            self.best_score = score
            print("saving best model to data/ folder")
            # torch.save(self.trainer.model.state_dict(), f"data/siamese_model_{score}.pth")
            knn_preds.to_csv(f"data/knn_preds_{score}.csv", index=False)

        generate_new_dataset_every_epoch = True
        if generate_new_dataset_every_epoch or (score == self.best_score):
            # generate new pairs in dataset
            print("Building new validation supervised df")
            if self.use_triplets:
                new_val_supervised_df = build_triplet_df(knn_preds, self.correlation_df)
            else:
                new_val_supervised_df = build_new_supervised_df(knn_preds, self.correlation_df)
            if score == self.best_score:  # only save for the best checkpoint
                print("saving new_val_supervised_df to data/ folder")
                new_val_supervised_df.to_csv("data/new_val_supervised_df.csv", index=False)

            # get top-k for training set
            # TODO: only get original content neighbors for original topics
            print("Generating embedding for train topics")
            train_topic_embs = []

            for inputs in tqdm(self.train_topic_dataloader):
                for k, v in inputs.items():
                    inputs[k] = inputs[k].to(device)
                out = self.trainer.model.feature(inputs)
                train_topic_embs.extend(out.cpu().detach().numpy())

            train_topic_embs_gpu = cp.array(train_topic_embs)

            train_indices = neighbors_model.kneighbors(train_topic_embs_gpu, return_distance=False)

            # if self.use_translated:
            #     topic_language_df = pd.DataFrame({
            #         "topic_id": self.train_topic_ids,
            #         "language": self.train_topic_languages
            #     })

            train_predictions = []
            for k in tqdm(range(len(train_indices))):
                pred = train_indices[k]
                # p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
                if self.use_translated:
                    p = " ".join([content_idx_to_id[original_indices[ind]] for ind in pred.get()])
                else:
                    p = " ".join([content_idx_to_id[ind] for ind in pred.get()])

                train_predictions.append(p)

            train_knn_preds = pd.DataFrame(
                {
                    "topic_id": self.train_topic_ids,
                    "content_ids": train_predictions,
                    "language": self.train_topic_languages,
                }
            ).sort_values("topic_id")

            print("Building new train supervised df")
            # if self.use_translated:
            #     count_dict = {
            #         "ar": 3701,
            #         "as": 167,
            #         "bg": 2867,
            #         "bn": 2176,
            #         "en": 36161,
            #         "es": 13910,
            #         "fil": 247,
            #         "fr": 3701,
            #         "gu": 2320,
            #         "hi": 1786,
            #         "it": 866,
            #         "km": 121,
            #         "kn": 119,
            #         "mr": 300,
            #         "mul": 4,
            #         "my": 135,
            #         "or": 70,
            #         "pl": 43,
            #         "pnb": 51,
            #         "pt": 4177,
            #         "ru": 34,
            #         "sw": 2860,
            #         "swa": 35,
            #         "ta": 60,
            #         "te": 93,
            #         "tr": 40,
            #         "ur": 66,
            #         "zh": 862,
            #     }

            #     times_positive_samples = 4

            #     # select all original topics and a part of translated topics
            #     translated_knn_preds = (
            #         train_knn_preds[~train_knn_preds.topic_id.str.startswith("t_")]
            #         .groupby("language")
            #         .apply(
            #             lambda x: x.sample(
            #                 n=count_dict[x["language"].iat[0]] * times_positive_samples,
            #                 replace=True,
            #             )
            #         )
            #         .reset_index(drop=True)
            #     )
            #     original_knn_preds = train_knn_preds[
            #         train_knn_preds.topic_id.str.startswith("t_")
            #     ]

            #     train_knn_preds = pd.concat([original_knn_preds, translated_knn_preds])

            if self.use_triplets:
                new_train_supervised_df = build_triplet_df(train_knn_preds, self.correlation_df)
            else:
                new_train_supervised_df = build_new_supervised_df(
                    train_knn_preds, self.correlation_df, self.reduce_negatives
                )

            if self.use_translated:
                # Only add positive cases in training set for translated topics
                translated_supervised_df = new_train_supervised_df[
                    ~new_train_supervised_df.topics_ids.str.startswith("t_")
                    & new_train_supervised_df.target
                    == 1
                ].copy()

                # Only original contents for original topics
                original_supervised_df = new_train_supervised_df[
                    new_train_supervised_df.topics_ids.str.startswith("t_")
                    & new_train_supervised_df.content_ids.str.startswith("c_")
                ].copy()

                # TODO: duplicate number of positive by using translated data
                id_to_language = {}
                for _, row in tqdm(self.topic_df.iterrows()):
                    id_to_language[row.id] = row.language

                original_supervised_df["language"] = original_supervised_df["topics_ids"].apply(
                    lambda x: id_to_language[x]
                )
                count_df = (
                    original_supervised_df[original_supervised_df.target == 1]
                    .groupby("language")
                    .size()
                    .reset_index(name="counts")
                )

                count_dict = {}
                for _, row in count_df.iterrows():
                    count_dict[row.language] = row.counts

                times_positive_samples = 3
                translated_supervised_df["language"] = translated_supervised_df[
                    "topics_ids"
                ].apply(lambda x: id_to_language[x])
                translated_supervised_df = (
                    translated_supervised_df.groupby("language")
                    .apply(
                        lambda x: x.sample(
                            n=count_dict[x["language"].iat[0]] * times_positive_samples,
                            replace=True,
                        )
                    )
                    .reset_index(drop=True)
                )
                original_supervised_df = original_supervised_df.drop(columns=["language"])
                translated_supervised_df = translated_supervised_df.drop(columns=["language"])

                new_train_supervised_df = pd.concat(
                    [translated_supervised_df, original_supervised_df]
                )[["topics_ids", "content_ids", "target"]]

            if score == self.best_score:  # only save for the best checkpoint
                print("saving new_train_supervised_df to data/ folder")
                new_train_supervised_df.to_csv("data/new_train_supervised_df.csv", index=False)

            # update train_dataset and val_dataset
            print("preprocess csv for train/validation topics, contents, labels")
            self.trainer.train_dataset.supervised_df = new_train_supervised_df.dropna()
            self.trainer.eval_dataset.supervised_df = new_val_supervised_df.dropna()
            if self.use_triplets:
                (
                    self.trainer.train_dataset.topic_texts,
                    self.trainer.train_dataset.pos_content_texts,
                    self.trainer.train_dataset.neg_content_texts,
                ) = self.trainer.train_dataset.process_csv()
                (
                    self.trainer.eval_dataset.topic_texts,
                    self.trainer.eval_dataset.pos_content_texts,
                    self.trainer.eval_dataset.neg_content_texts,
                ) = self.trainer.eval_dataset.process_csv()
            else:
                (
                    self.trainer.train_dataset.topic_texts,
                    self.trainer.train_dataset.content_texts,
                    self.trainer.train_dataset.labels,
                ) = self.trainer.train_dataset.process_csv()
                (
                    self.trainer.eval_dataset.topic_texts,
                    self.trainer.eval_dataset.content_texts,
                    self.trainer.eval_dataset.labels,
                ) = self.trainer.eval_dataset.process_csv()

            del (
                train_topic_embs,
                train_topic_embs_gpu,
                train_knn_preds,
                train_indices,
                train_predictions,
            )
            gc.collect()

        del (
            topic_embs,
            content_embs,
            topic_embs_gpu,
            content_embs_gpu,
            knn_preds,
            indices,
            neighbors_model,
            predictions,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # topics_ids, labels = (
        #     self.trainer.train_dataset.supervised_df["topics_ids"].values,
        #     self.trainer.train_dataset.supervised_df["target"].values,
        # )
        # self.trainer.callback_handler.train_dataloader.sampler.initialize(
        #     topics_ids, labels
        # )
