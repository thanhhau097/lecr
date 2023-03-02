from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from dataset import init_tokenizer


class LECRerankDataset(Dataset):
    def __init__(
        self,
        tokenizer_name: str,
        max_len: int,
        topics_dict: Dict[str, str],
        content_embs_dict: Dict[str, np.ndarray],
        knn_df: pd.DataFrame,
        topic2relevantcontents: Dict[str, str],
    ):
        self.topics_dict, self.content_embs_dict = topics_dict, content_embs_dict
        self.tokenizer = init_tokenizer(tokenizer_name)
        self.max_len = max_len
        self.knn_df = knn_df
        self.topic2relevantcontents = topic2relevantcontents

    def __len__(self):
        return len(self.knn_df)

    def __getitem__(self, idx):
        row = self.knn_df.iloc[idx]

        topic_inputs = self.tokenizer.encode_plus(
            self.topics_dict.get(row["topic_id"]),
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        for k, v in topic_inputs.items():
            topic_inputs[k] = torch.tensor(v, dtype=torch.long)

        relevant_content_ids = self.topic2relevantcontents.get(row["topic_id"]).split(" ")

        content_embs = []
        labels = []
        for idx, content_id in enumerate(row["content_ids"].split(" ")):
            content_embs.append(self.content_embs_dict.get(content_id))
            # # rank labels
            # if content_id in relevant_content_ids:
            #     labels.append(idx)
            # else:
            #     labels.append(-1)
            if content_id in relevant_content_ids:
                labels.append(1)
            else:
                labels.append(0)

        content_embs = torch.from_numpy(np.stack(content_embs)).float()
        # labels = torch.tensor(labels)

        # labels, sorted_indices = torch.sort(labels, descending=True)
        # content_embs = content_embs[sorted_indices]

        return topic_inputs, content_embs, torch.as_tensor(labels)


def collate_fn(batch):
    batch = default_collate(batch)

    topic_inputs, content_embs, labels = batch

    mask_len = int(topic_inputs["attention_mask"].sum(axis=1).max())
    for k, v in topic_inputs.items():
        topic_inputs[k] = topic_inputs[k][:, :mask_len]

    return {"topic_inputs": topic_inputs, "content_embs": content_embs, "labels": labels}
