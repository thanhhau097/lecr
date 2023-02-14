from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm
from transformers import AutoTokenizer

from tokenizer import init_tokenizer


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
        use_all_pairs = (
            False  # use all pair, no need to be in the intersection of content_ids of topic ids
        )
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
                            if (content_list[idx1], content_list[idx2],) not in pairs and (
                                content_list[idx2],
                                content_list[idx1],
                            ) not in pairs:
                                pairs.add((content_list[idx1], content_list[idx2]))

            if not use_all_pairs:
                for content_id, topics in tqdm(content_to_topic.items()):
                    intersection_contents = list(
                        set.intersection(*[set(topic_to_content[topic_id]) for topic_id in topics])
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
        return len(self.labels)

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
                topic_inputs = augment(topic_inputs, self.tokenizer)
                content_inputs = augment(content_inputs, self.tokenizer)

            topic_id = self.supervised_df.topics_ids.values[idx]
            return topic_inputs, content_inputs, topic_id, label
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
                combined_inputs = augment(combined_inputs, self.tokenizer)

            return combined_inputs, combined_inputs, combined_inputs, label
        else:
            raise ValueError("Only support siamese/classification for now.")


class LECRTripletDataset(Dataset):
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
        self.topic_texts, self.pos_content_texts, self.neg_content_texts = self.process_csv()

    def process_csv(self):
        topic_ids = self.supervised_df["topics_ids"].values
        pos_content_ids = self.supervised_df["pos_content_ids"].values
        neg_content_ids = self.supervised_df["neg_content_ids"].values

        topic_texts, pos_content_texts, neg_content_texts = [], [], []
        for topic_id, pos_id, neg_id in zip(topic_ids, pos_content_ids, neg_content_ids):
            topic_texts.append(self.topic_dict[topic_id])
            pos_content_texts.append(self.content_dict[pos_id])
            neg_content_texts.append(self.content_dict[neg_id])

        return topic_texts, pos_content_texts, neg_content_texts

    def __len__(self):
        return len(self.topic_texts)

    def __getitem__(self, idx):
        topic_text = self.topic_texts[idx]
        pos_content_text = self.pos_content_texts[idx]
        neg_content_text = self.neg_content_texts[idx]
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
        pos_content_inputs = self.tokenizer.encode_plus(
            pos_content_text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        for k, v in pos_content_inputs.items():
            pos_content_inputs[k] = torch.tensor(v, dtype=torch.long)
        neg_content_inputs = self.tokenizer.encode_plus(
            neg_content_text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        for k, v in neg_content_inputs.items():
            neg_content_inputs[k] = torch.tensor(v, dtype=torch.long)
        return topic_inputs, pos_content_inputs, neg_content_inputs


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


def augment(inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer):
    probability_matrix = torch.full(inputs["input_ids"].shape, 0.15)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    indices_replaced = (
        torch.bernoulli(torch.full(inputs["input_ids"].shape, 0.8)).bool() & masked_indices
    )
    inputs["input_ids"][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs["input_ids"] *= inputs["attention_mask"]
    return inputs


def truncate_inputs(inputs: Dict[str, torch.Tensor]):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, _ in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def collate_fn(batch):
    batch = default_collate(batch)
    topic_inputs, content_inputs, topic_ids, labels = batch
    # create labels for MultipleNegativesRankingLoss
    topic_ids_to_labels = {topic_id: label for label, topic_id in enumerate(np.unique(topic_ids))}
    mnrl_labels = torch.from_numpy(np.vectorize(topic_ids_to_labels.get)(topic_ids))

    return {
        "topic_inputs": truncate_inputs(topic_inputs),
        "content_inputs": truncate_inputs(content_inputs),
        "mnrl_labels": mnrl_labels,
        "labels": labels,
    }


def triplet_collate_fn(batch):
    batch = default_collate(batch)
    topic_inputs, pos_content_inputs, neg_content_inputs = batch
    return {
        "topic_inputs": truncate_inputs(topic_inputs),
        "content_inputs": truncate_inputs(pos_content_inputs),
        "neg_content_inputs": truncate_inputs(neg_content_inputs),
    }


def build_dataset_and_collator(
    supervised_df,
    topic_df,
    content_df,
    topic_dict,
    content_dict,
    correlation_df,
    tokenizer_name,
    max_len,
    use_content_pair,
    is_training,
    use_augmentation,
    objective,
):
    if objective in ["siamese", "both"]:
        return (
            LECRDataset(
                supervised_df,
                topic_df,
                content_df,
                topic_dict,
                content_dict,
                correlation_df,
                tokenizer_name,
                max_len,
                use_content_pair,
                is_training,
                use_augmentation,
                objective,
            ),
            collate_fn,
        )
    if objective == "triplet":
        return (
            LECRTripletDataset(
                supervised_df,
                topic_df,
                content_df,
                topic_dict,
                content_dict,
                correlation_df,
                tokenizer_name,
                max_len,
                use_content_pair,
                is_training,
                use_augmentation,
            ),
            triplet_collate_fn,
        )
