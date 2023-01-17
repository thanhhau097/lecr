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


def init_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(
        dict(
            additional_special_tokens=LANGUAGE_TOKENS
            + CATEGORY_TOKENS
            + LEVEL_TOKENS
            + KIND_TOKENS
            + OTHER_TOKENS
        )
    )
    return tokenizer


def get_processed_text_dict(topic_df, content_df):
    # Fillna titles
    topic_df["title"].fillna("", inplace=True)
    content_df["title"].fillna("", inplace=True)

    # Fillna descriptions
    topic_df["description"].fillna("", inplace=True)
    content_df["description"].fillna("", inplace=True)

    # clean text
    print("Cleaning text data for topics")
    topic_df["title"] = topic_df["title"].apply(clean_text)
    topic_df["description"] = topic_df["description"].apply(clean_text)

    print("Cleaning text data for content")
    content_df["title"] = content_df["title"].apply(clean_text)
    content_df["description"] = content_df["description"].apply(clean_text)
    # self.content_df["text"] = self.content_df["text"].apply(clean_text)

    # get concatenated texts
    topic_dict = {}
    for i, (index, row) in tqdm(enumerate(topic_df.iterrows())):
        text = (
            "<|topic|>"
            + f"<|lang_{row['language']}|>"
            + f"<|category_{row['category']}|>"
            + f"<|level_{row['level']}|>"
        )
        text += (
            "<s_title>"
            + row["title"]
            + "</s_title>"
            + "<s_description>"
            + row["description"]
            + "</s_description>"
        )
        topic_dict[row["id"]] = text

    content_dict = {}
    for i, (index, row) in tqdm(enumerate(content_df.iterrows())):
        text = "<|content|>" + f"<|lang_{row['language']}|>" + f"<|kind_{row['kind']}|>"
        text += (
            "<s_title>"
            + row["title"]
            + "</s_title>"
            + "<s_description>"
            + row["description"]
            + "</s_description>"
        )  # + "<s_text>" + row["text"] + "</s_text>"
        content_dict[row["id"]] = text[:2048]

    return topic_dict, content_dict


class LECRDataset(Dataset):
    def __init__(
        self,
        supervised_df,
        topic_df,
        content_df,
        correlation_df,
        tokenizer_name="xlm-roberta-base",
        max_len=512,
        use_content_pair=False,
    ):
        self.supervised_df = supervised_df
        self.topic_df = topic_df
        self.content_df = content_df
        self.correlation_df = correlation_df
        self.use_content_pair = use_content_pair
        self.topic_texts, self.content_texts, self.labels = self.process_csv()

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.max_len = max_len

    def process_csv(self):
        topic_dict, content_dict = get_processed_text_dict(
            self.topic_df, self.content_df
        )

        # get text pairs
        topic_ids = self.supervised_df.topics_ids.values
        content_ids = self.supervised_df.content_ids.values
        labels = list(self.supervised_df.target.values)

        topic_texts = []
        content_texts = []

        for topic_id in topic_ids:
            topic_texts.append(topic_dict[topic_id])

        for content_id in content_ids:
            content_texts.append(content_dict[content_id])

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
                topic_texts.append(content_dict[pair[0]])
                content_texts.append(content_dict[pair[1]])
                labels.append(1)

        return topic_texts, content_texts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        topic_text = self.topic_texts[idx]
        content_text = self.content_texts[idx]
        label = self.labels[idx]

        # topic
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

        return topic_inputs, content_inputs, combined_inputs, label


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
        correlation_df,
        tokenizer_name,
        max_len,
        best_score=0,
    ):
        super(DatasetUpdateCallback, self).__init__()
        self.trainer = trainer
        self.topic_df = topic_df
        self.content_df = content_df
        self.correlation_df = correlation_df
        self.best_score = best_score

        topic_dict, content_dict = get_processed_text_dict(
            self.topic_df, self.content_df
        )

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
            content_dict[content_id] for content_id in self.content_df.id.values
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
            batch_size=64,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        val_topic_dataset = InferenceDataset(
            texts=val_topic_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.val_topic_dataloader = DataLoader(
            val_topic_dataset,
            num_workers=16,
            batch_size=64,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

        content_dataset = InferenceDataset(
            texts=content_texts, tokenizer_name=tokenizer_name, max_len=max_len
        )
        self.content_dataloader = DataLoader(
            content_dataset,
            num_workers=16,
            batch_size=64,
            shuffle=False,
            collate_fn=inference_collate_fn,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("On Epoch Begin")
        topic_embs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for inputs in tqdm(self.val_topic_dataloader):
            for k, v in inputs.items():
                inputs[k] = inputs[k].to(device)
            out = self.trainer.model.feature(inputs)
            topic_embs.extend(out.cpu().detach().numpy())

        content_embs = []

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
        print("Evaluating current score...")
        for selected_k in [5, 10, 20, 50]:
            neighbors_model = NearestNeighbors(n_neighbors=selected_k, metric="cosine")
            neighbors_model.fit(content_embs_gpu)

            indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
            predictions = []
            for k in tqdm(range(len(indices))):
                pred = indices[k]
                p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
                predictions.append(p)

            knn_preds = pd.DataFrame(
                {"topic_id": self.val_topic_ids, "content_ids": predictions}
            ).sort_values("topic_id")

            gt = self.correlation_df[
                self.correlation_df.topic_id.isin(self.val_topic_ids)
            ].sort_values("topic_id")
            score = get_pos_score(
                gt["content_ids"], knn_preds.sort_values("topic_id")["content_ids"], selected_k
            )
            print("Selecting", selected_k, "nearest contents", "top-k score =", f2_score(gt["content_ids"], knn_preds.sort_values("topic_id")["content_ids"]), "max positive score =", score)

        print("Training KNN model...")
        top_k = 50
        print("Generating KNN predictions with top_k =", top_k)
        neighbors_model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
        neighbors_model.fit(content_embs_gpu)

        print("Generating embedding for validation topics")
        indices = neighbors_model.kneighbors(topic_embs_gpu, return_distance=False)
        predictions = []
        for k in tqdm(range(len(indices))):
            pred = indices[k]
            p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
            predictions.append(p)

        knn_preds = pd.DataFrame(
            {"topic_id": self.val_topic_ids, "content_ids": predictions}
        ).sort_values("topic_id")

        if score > self.best_score:
            self.best_score = score
            # generate new pairs in dataset
            print("Building new validation supervised df")
            new_val_supervised_df = build_new_supervised_df(
                knn_preds, self.correlation_df
            )

            # get top-k for training set
            print("Generating embedding for train topics")
            train_topic_embs = []

            for inputs in tqdm(self.train_topic_dataloader):
                for k, v in inputs.items():
                    inputs[k] = inputs[k].to(device)
                out = self.trainer.model.feature(inputs)
                train_topic_embs.extend(out.cpu().detach().numpy())

            train_topic_embs_gpu = cp.array(train_topic_embs)

            train_indices = neighbors_model.kneighbors(
                train_topic_embs_gpu, return_distance=False
            )
            train_predictions = []
            for k in tqdm(range(len(train_indices))):
                pred = train_indices[k]
                p = " ".join([self.content_df.loc[ind, "id"] for ind in pred.get()])
                train_predictions.append(p)

            train_knn_preds = pd.DataFrame(
                {"topic_id": self.train_topic_ids, "content_ids": train_predictions}
            ).sort_values("topic_id")

            print("Building new train supervised df")
            new_train_supervised_df = build_new_supervised_df(
                train_knn_preds, self.correlation_df
            )
            # update train_dataset and val_dataset
            print("preprocess csv for train/validation topics, contents, labels")
            self.trainer.train_dataset.supervised_df = new_train_supervised_df
            (
                self.trainer.train_dataset.topic_texts,
                self.trainer.train_dataset.content_texts,
                self.trainer.train_dataset.labels,
            ) = self.trainer.train_dataset.process_csv()

            self.trainer.eval_dataset.supervised_df = new_val_supervised_df
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


def build_new_supervised_df(knn_df, correlations):
    # Create lists for training
    topics_ids = []
    content_ids = []
    targets = []

    # Iterate over each topic in df
    mapping = set()

    # get all class 1 in correlations
    topic_ids = set(knn_df.topic_id.values)
    filtered_correlations = correlations[correlations["topic_id"].isin(topic_ids)]
    for i, row in tqdm(filtered_correlations.iterrows()):
        content_ids = row["content_ids"].split(" ")
        if content_ids:
            for content_id in content_ids:
                mapping.add((row["topic_id"], content_id, 1))

    for i, row in tqdm(knn_df.iterrows()):
        content_ids = row["content_ids"].split(" ")
        if content_ids:
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
            "topics_ids": [item[0] for item in mapping],
            "content_ids": [item[1] for item in mapping],
            "target": [item[2] for item in mapping],
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
