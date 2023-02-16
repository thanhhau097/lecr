import random

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


class ProportionalTwoClassesBatchSampler(Sampler):
    """
    dataset: DataSet class that returns torch tensors
    batch_size: Size of mini-batches
    minority_size_in_batch: Number of minority class samples in each mini-batch
    majority_priority: If it is True, iterations will include all majority
    samples in the data. Otherwise, it will be completed after all minority samples are used.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        minority_size_in_batch: int,
        majority_priority=True,
    ):
        super().__init__(labels)
        self.labels = labels
        self.minority_size_in_batch = minority_size_in_batch
        self.batch_size = batch_size
        self.priority = majority_priority
        self._num_batches = (labels == 0).sum() // (batch_size - minority_size_in_batch)
        self._num_samples = (len(self.labels) // self.batch_size + 1) * self.batch_size

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        if self.minority_size_in_batch > self.batch_size:
            raise ValueError(
                "Number of minority samples in a batch must be lower than batch size!"
            )
        y_indices = [np.where(self.labels == label)[0] for label in np.unique(self.labels)]
        y_indices = sorted(y_indices, key=lambda x: x.shape)

        # y_indices[0]: minority
        # y_indices[1]: majority
        # create num_batch pairs

        y_indices[0] = np.random.permutation(y_indices[0])
        y_indices[1] = np.random.permutation(y_indices[1])

        minority = np.split(y_indices[0][:(len(y_indices[0]) // self.minority_size_in_batch) * self.minority_size_in_batch], len(y_indices[0]) // self.minority_size_in_batch)
        minority = minority * (self._num_batches // len(minority) + 1)
        minority = minority[:self._num_batches]
        random.shuffle(minority)

        majority_size_in_batch = self.batch_size - self.minority_size_in_batch
        majority = np.split(y_indices[1][:(len(y_indices[1]) // majority_size_in_batch) * majority_size_in_batch], len(y_indices[1]) // majority_size_in_batch)
        majority = majority * (self._num_batches // len(majority) + 1)
        majority = majority[:self._num_batches]
        random.shuffle(majority)

        indices = np.concatenate([np.random.permutation(np.concatenate([a, b])) for a, b in zip(minority, majority)]).tolist()

        return iter(indices[: self._num_samples])



class TopicSampler(Sampler):
    def __init__(self, topics_ids, labels, batch_size, per_topic_batch_size=32):

        self.topics_ids = topics_ids
        self.labels = labels
        self.batch_size = batch_size
        self.per_topic_batch_size = per_topic_batch_size
        self.initialize(topics_ids, labels)

    def initialize(self, topics_ids, labels):
        self.topics_dict = {}
        topics2indices = {}
        for i, topic_id in enumerate(topics_ids):
            if topic_id not in topics2indices:
                topics2indices[topic_id] = []
            topics2indices[topic_id].append(i)

        for topic_id in tqdm(np.unique(topics_ids)):
            indices = topics2indices.get(topic_id)
            self.topics_dict[topic_id] = (np.array(indices), labels[indices])

        self.topics_ids_list = np.unique(topics_ids)
        self.num_topics_per_batch = self.batch_size // self.per_topic_batch_size
        self.per_topic_batch_size = self.per_topic_batch_size
        self.batch_size = self.batch_size
        self._dataset_length = (
            len(self.topics_ids_list) // self.num_topics_per_batch + 1
        ) * self.batch_size

    def __len__(self):
        return self._dataset_length

    def __iter__(self):
        topics_ids_list = np.random.permutation(self.topics_ids_list)

        indices = []
        for i in tqdm(range(0, len(topics_ids_list), self.num_topics_per_batch)):
            topics_ids = topics_ids_list[i : i + self.num_topics_per_batch]
            batch_indices = []
            for topic_id in topics_ids:
                topic_indices = self.topics_dict[topic_id][0]
                topic_labels = self.topics_dict[topic_id][1]
                num_positives = (topic_labels == 1).sum()
                num_negatives = (topic_labels == 0).sum()
                if num_positives < self.per_topic_batch_size:
                    topic_indices = np.append(
                        topic_indices[topic_labels == 1],
                        np.random.permutation(topic_indices[topic_labels == 0]),
                    )[: self.per_topic_batch_size]
                else:
                    num_negatives = max(num_negatives, self.per_topic_batch_size // 2)
                    num_positives = self.per_topic_batch_size - num_negatives
                    topic_indices = np.append(
                        np.random.permutation(topic_indices[topic_labels == 1])[:num_positives],
                        np.random.permutation(topic_indices[topic_labels == 0])[:num_negatives],
                    )
                batch_indices.extend(topic_indices.tolist())
            batch_indices = batch_indices[: self.batch_size]
            random.shuffle(batch_indices)
            indices.extend(batch_indices)

        return iter(indices[: self._dataset_length])


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/new_train_supervised_df.csv")
    sampler = TopicSampler(df.topics_ids.values, df.target.values, 128)
    inds = list(iter(sampler))