from collections import defaultdict
import gc

import cupy as cp
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainerCallback

from utils import clean_text, f2_score, get_pos_score


LANGUAGE_TOKENS = [
    "<|lang_pnb|>",
    "<|lang_tr|>",
    "<|lang_ur|>",
    "<|lang_bn|>",
    "<|lang_hi|>",
    "<|lang_en|>",
    "<|lang_kn|>",
    "<|lang_km|>",
    "<|lang_zh|>",
    "<|lang_gu|>",
    "<|lang_ta|>",
    "<|lang_my|>",
    "<|lang_fr|>",
    "<|lang_swa|>",
    "<|lang_or|>",
    "<|lang_mul|>",
    "<|lang_fil|>",
    "<|lang_sw|>",
    "<|lang_es|>",
    "<|lang_pt|>",
    "<|lang_pl|>",
    "<|lang_ru|>",
    "<|lang_mr|>",
    "<|lang_it|>",
    "<|lang_ar|>",
    "<|lang_bg|>",
    "<|lang_te|>",
    "<|lang_as|>",
]


CATEGORY_TOKENS = [
    "<|category_supplemental|>",
    "<|category_aligned|>",
    "<|category_source|>",
]

LEVEL_TOKENS = [
    "<|level_0|>",
    "<|level_1|>",
    "<|level_2|>",
    "<|level_3|>",
    "<|level_4|>",
    "<|level_5|>",
    "<|level_6|>",
    "<|level_7|>",
    "<|level_8|>",
    "<|level_9|>",
    "<|level_10|>",
]

KIND_TOKENS = [
    "<|kind_document|>",
    "<|kind_video|>",
    "<|kind_html5|>",
    "<|kind_exercise|>",
    "<|kind_audio|>",
]

OTHER_TOKENS = [
    "<|topic|>",
    "<|content|>",
    "<s_title>",
    "</s_title>",
    "<s_description>",
    "</s_description>",
    "<s_text>",
    "</s_text>",
]

RELATION_TOKENS = [
    "<s_parent>",
    "</s_parent>",
    "<s_children>",
    "</s_children>",
]


def init_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(
        dict(
            additional_special_tokens=LANGUAGE_TOKENS
            + CATEGORY_TOKENS
            + LEVEL_TOKENS
            + KIND_TOKENS
            + OTHER_TOKENS
            + RELATION_TOKENS
        )
    )
    return tokenizer


class LECRDataset(Dataset):
    def __init__(
        self,
        supervised_df,
        topic_df,
        content_df,
        topic_dict,
        content_dict,
        correlation_df,
        tokenizer_name="xlm-roberta-base",
        max_len=512,
        use_content_pair=False,
        is_training=False,
        use_augmentation=False,
        objective="siamese",
    ):
        self.tokenizer = init_tokenizer(tokenizer_name)
        self.max_len = max_len

        self.supervised_df = supervised_df.dropna()
        self.topic_df = topic_df
        self.content_df = content_df
        self.topic_dict, self.content_dict = topic_dict, content_dict
        self.correlation_df = correlation_df
        self.use_content_pair = use_content_pair
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        self.objective = objective
        self.topic_texts, self.content_texts, self.labels = self.process_csv()

    def process_csv(self):
        # get text pairs
        topic_ids = self.supervised_df.topics_ids.values
        content_ids = self.supervised_df.content_ids.values
        labels = list(self.supervised_df.target.values)

        topic_texts = []
        content_texts = []

        for topic_id in topic_ids:
            topic_texts.append(self.topic_dict[topic_id])

        for content_id in content_ids:
            content_texts.append(self.content_dict[content_id])

        set_topic_ids = set(topic_ids)
        use_all_pairs = False  # use all pair, no need to be in the intersection of content_ids of topic ids
        if self.use_content_pair:
            # todo: create content pairs from each topic
            content_to_topic = defaultdict(lambda: [])
            topic_to_content = defaultdict(lambda: [])

            pairs = set()

            for i, row in tqdm(self.correlation_df.iterrows()):
                content_list = row["content_ids"].split(" ")

                if row["topic_id"] not in set_topic_ids:
                    continue

                for content_id in content_list:
                    content_to_topic[content_id].append(row["topic_id"])
                    topic_to_content[row["topic_id"]].append(content_id)

                if len(content_list) <= 1:
                    continue

                if use_all_pairs:
                    for idx1 in range(len(content_list) - 1):
                        for idx2 in range(idx1 + 1, len(content_list)):
                            if (
                                content_list[idx1],
                                content_list[idx2],
                            ) not in pairs and (
                                content_list[idx2],
                                content_list[idx1],
                            ) not in pairs:
                                pairs.add((content_list[idx1], content_list[idx2]))

            if not use_all_pairs:
                for content_id, topics in tqdm(content_to_topic.items()):
                    intersection_contents = list(
                        set.intersection(
                            *[set(topic_to_content[topic_id]) for topic_id in topics]
                        )
                    )
                    for idx1 in range(len(intersection_contents) - 1):
                        for idx2 in range(idx1 + 1, len(intersection_contents)):
                            if (
                                intersection_contents[idx1],
                                intersection_contents[idx2],
                            ) not in pairs and (
                                intersection_contents[idx2],
                                intersection_contents[idx1],
                            ) not in pairs:
                                pairs.add(
                                    (
                                        intersection_contents[idx1],
                                        intersection_contents[idx2],
                                    )
                                )

            for pair in pairs:
                topic_texts.append(self.content_dict[pair[0]])
                content_texts.append(self.content_dict[pair[1]])
                labels.append(1)

        return topic_texts, content_texts, labels

    def __len__(self):
        if self.is_training:
            return len(self.labels)
        else:
            return 1

    def augment(self, inputs):
        probability_matrix = torch.full(inputs["input_ids"].shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_replaced = (
            torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.8)).bool()
            & masked_indices
        )
        inputs["input_ids"][indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        inputs["input_ids"] *= inputs["attention_mask"]

        return inputs

    def __getitem__(self, idx):
        topic_text = self.topic_texts[idx]
        content_text = self.content_texts[idx]
        label = self.labels[idx]

        if self.objective == "siamese":
            # topic
            if isinstance(topic_text, tuple):
                topic_inputs = self.tokenizer.encode_plus(
                    topic_text[0],
                    topic_text[1],
                    return_tensors=None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                )
            else:
                topic_inputs = self.tokenizer.encode_plus(
                    topic_text,
                    return_tensors=None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                )
            for k, v in topic_inputs.items():
                topic_inputs[k] = torch.tensor(v, dtype=torch.long)

            # content
            content_inputs = self.tokenizer.encode_plus(
                content_text,
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
            for k, v in content_inputs.items():
                content_inputs[k] = torch.tensor(v, dtype=torch.long)

            if isinstance(topic_text, tuple):
                topic_text = topic_text[0] + topic_text[1]

            if self.is_training and self.use_augmentation:
                topic_inputs = self.augment(topic_inputs)
                content_inputs = self.augment(content_inputs)

            return topic_inputs, content_inputs, topic_inputs, label
        elif self.objective == "classification":
            combined_inputs = self.tokenizer.encode_plus(
                topic_text,
                content_text,
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
            for k, v in combined_inputs.items():
                combined_inputs[k] = torch.tensor(v, dtype=torch.long)

            if self.is_training and self.use_augmentation:
                combined_inputs = self.augment(combined_inputs)

            return combined_inputs, combined_inputs, combined_inputs, label
        else:
            raise ValueError("Only support siamese/classification for now.")


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer_name="xlm-roberta-base", max_len=512):
        self.texts = texts

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # topic
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        return inputs


def collate_fn(inputs):
    inputs = default_collate(inputs)
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]

    return inputs


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
        correlation_df,
        tokenizer_name,
        max_len,
        best_score=0,
        top_k=50,
        use_translated=False,
        mix_translated=False,
    ):
        super(DatasetUpdateCallback, self).__init__()
        self.trainer = trainer
        self.topic_df = topic_df
        self.content_df = content_df
        self.correlation_df = correlation_df
        self.best_score = best_score
        self.top_k = top_k
        self.use_translated = use_translated
        self.mix_translated = mix_translated

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.topic_dict, self.content_dict = topic_dict, content_dict

        train_topic_texts = [
            topic_dict[topic_id]
            for topic_id in self.topic_df.id.values
            if topic_id in train_topic_ids
        ]
        self.train_topic_ids = [
            topic_id
            for topic_id in self.topic_df.id.values
            if topic_id in train_topic_ids
        ]
        self.train_topic_languages = []
        for topic_id, topic_lang in zip(
            self.topic_df.id.values, self.topic_df.language.values
        ):
            if topic_id in train_topic_ids:
                self.train_topic_languages.append(topic_lang)

        val_topic_texts = [
            topic_dict[topic_id]
            for topic_id in self.topic_df.id.values
            if topic_id in val_topic_ids
        ]
        self.val_topic_ids = [
            topic_id
            for topic_id in self.topic_df.id.values
            if topic_id in val_topic_ids
        ]

        content_texts = [
            content_dict[content_id] for content_id in self.content_df.id.values if content_id.startswith("c_")
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
            batch_size=32,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        val_topic_dataset = InferenceDataset(
            texts=val_topic_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.val_topic_dataloader = DataLoader(
            val_topic_dataset,
            num_workers=16,
            batch_size=32,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        content_dataset = InferenceDataset(
            texts=content_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.content_dataloader = DataLoader(
            content_dataset,
            num_workers=16,
            batch_size=32,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.on_epoch_end(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        print("Callback on local_rank =", args.local_rank)
        self.trainer.model.eval()
        print("On Epoch Begin")
        topic_embs = []
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        
        with torch.no_grad():
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
        with cp.cuda.Device(args.local_rank):
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
            original_indices = [ # indices of original contents in self.content_df
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
                neighbors_model = NearestNeighbors(
                    n_neighbors=selected_k, metric="cosine"
                )
                neighbors_model.fit(content_embs_gpu)

                indices = neighbors_model.kneighbors(
                    topic_embs_gpu, return_distance=False
                )
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
            torch.save(
                self.trainer.model.state_dict(), f"data/siamese_model_{score}.pth"
            )

        generate_new_dataset_every_epoch = True
        if generate_new_dataset_every_epoch or (score == self.best_score):
            # generate new pairs in dataset
            print("Building new validation supervised df")
            new_val_supervised_df = build_new_supervised_df(
                knn_preds, self.correlation_df
            )[["topics_ids", "content_ids", "target"]].sort_values(["topics_ids", "content_ids"])
            if score == self.best_score:  # only save for the best checkpoint
                print("saving new_val_supervised_df to data/ folder")
                new_val_supervised_df.to_csv("data/new_val_supervised_df.csv")

            # get top-k for training set
            # TODO: only get original content neighbors for original topics
            print("Generating embedding for train topics")
            train_topic_embs = []

            with torch.no_grad():
                for inputs in tqdm(self.train_topic_dataloader):
                    for k, v in inputs.items():
                        inputs[k] = inputs[k].to(device)
                    out = self.trainer.model.feature(inputs)
                    train_topic_embs.extend(out.cpu().detach().numpy())

            with cp.cuda.Device(args.local_rank):
                train_topic_embs_gpu = cp.array(train_topic_embs)

            train_indices = neighbors_model.kneighbors(
                train_topic_embs_gpu, return_distance=False
            )

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

            new_train_supervised_df = build_new_supervised_df(
                train_knn_preds, self.correlation_df
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
                
                original_supervised_df["language"] = original_supervised_df["topics_ids"].apply(lambda x: id_to_language[x])
                count_df = original_supervised_df[original_supervised_df.target == 1].groupby("language").size().reset_index(name='counts')

                count_dict = {}
                for _, row in count_df.iterrows():
                    count_dict[row.language] = row.counts

                times_positive_samples = 3
                translated_supervised_df["language"] = translated_supervised_df["topics_ids"].apply(lambda x: id_to_language[x])
                translated_supervised_df = (
                    translated_supervised_df
                    .groupby("language")
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
                )[["topics_ids", "content_ids", "target"]].sort_values(["topics_ids", "content_ids"])

            if score == self.best_score:  # only save for the best checkpoint
                print("saving new_train_supervised_df to data/ folder")
                new_train_supervised_df.to_csv("data/new_train_supervised_df.csv")

            # update train_dataset and val_dataset
            print("preprocess csv for train/validation topics, contents, labels")
            self.trainer.train_dataset.supervised_df = new_train_supervised_df.dropna()
            (
                self.trainer.train_dataset.topic_texts,
                self.trainer.train_dataset.content_texts,
                self.trainer.train_dataset.labels,
            ) = self.trainer.train_dataset.process_csv()

            self.trainer.eval_dataset.supervised_df = new_val_supervised_df.dropna()
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

        if self.mix_translated:
            self.use_translated = not self.use_translated


def build_new_supervised_df(knn_df, correlations):
    # Create lists for training
    topics_ids = []
    content_ids = []
    targets = []

    # Iterate over each topic in df
    mapping = set()

    # get all class 1 in correlations
    topic_ids = set(knn_df.topic_id.values)
    filtered_correlations = correlations[
        correlations["topic_id"].isin(topic_ids)
    ]
    for i, row in tqdm(filtered_correlations.iterrows()):
        if str(row["content_ids"]) and str(row["content_ids"]) != "nan":
            content_ids = str(row["content_ids"]).split(" ")
            for content_id in content_ids:
                mapping.add((row["topic_id"], content_id, 1))

    for i, row in tqdm(knn_df.iterrows()):
        if str(row["content_ids"]) and str(row["content_ids"]) != "nan":
            content_ids = str(row["content_ids"]).split(" ")
            for content_id in content_ids:
                if (
                    row["topic_id"],
                    content_id,
                    1,
                ) not in mapping:  # because mapping already contains all positive cases
                    mapping.add((row["topic_id"], content_id, 0))

    # Build training dataset
    mapping = list(mapping)
    new_df = pd.DataFrame(
        {
            "topics_ids": [item[0] for item in mapping if item[1]],
            "content_ids": [item[1] for item in mapping if item[1]],
            "target": [item[2] for item in mapping if item[1]],
        }
    )
    # Release memory
    del topics_ids, content_ids
    gc.collect()
    return new_df


def collate_fn(batch):
    batch = default_collate(batch)

    topic_inputs, content_inputs, combined_inputs, labels = batch
    mask_len = int(topic_inputs["attention_mask"].sum(axis=1).max())
    for k, v in topic_inputs.items():
        topic_inputs[k] = topic_inputs[k][:, :mask_len]

    mask_len = int(content_inputs["attention_mask"].sum(axis=1).max())
    for k, v in content_inputs.items():
        content_inputs[k] = content_inputs[k][:, :mask_len]

    mask_len = int(combined_inputs["attention_mask"].sum(axis=1).max())
    for k, v in combined_inputs.items():
        combined_inputs[k] = combined_inputs[k][:, :mask_len]

    return {
        "topic_inputs": topic_inputs,
        "content_inputs": content_inputs,
        "combined_inputs": combined_inputs,
        "labels": labels,
    }
