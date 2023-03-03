import random

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


class ProportionalTwoClassesBatchSamplerĐDP(Sampler):
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
        world_size: int,
        local_rank: int,
        majority_priority=True,
    ):
        super().__init__(labels)
        self.labels = labels
        self.minority_size_in_batch = minority_size_in_batch
        self.batch_size = batch_size
        self.priority = majority_priority
        self._num_batches = (labels == 0).sum() // (batch_size - minority_size_in_batch)
        self._num_samples = (len(self.labels) // self.batch_size + 1) * self.batch_size
        self.world_size = world_size
        self.local_rank = local_rank

    def __len__(self):
        return self._num_samples // self.world_size

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
        minor_per_device = len(y_indices[0])//self.world_size
        major_per_device = len(y_indices[1])//self.world_size
        y_indices[0] = np.random.permutation(y_indices[0][self.local_rank*minor_per_device : (self.local_rank + 1)*minor_per_device])
        y_indices[1] = np.random.permutation(y_indices[1][self.local_rank*major_per_device : (self.local_rank + 1)*major_per_device])

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

        return iter(indices[: self._num_samples  // self.world_size])


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
        **kwargs,
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

