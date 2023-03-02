import gc
from enum import Enum
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import fbeta_score
from torch import Tensor, nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import Scorer


class RelevanceListnetLoss(nn.Module):
    """
    ListNet loss
    """

    def __init__(self):
        super(RelevanceListnetLoss, self).__init__()

    def forward(self, predictions, labels):
        """
        :param predictions: (batch_size, num_docs) relevance scores (arb. range) for each candidate document and query.
        :param labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices (positions) of the
                relevant documents within its corresponding pool of candidates (docinds). If n relevant documents exist,
                then labels[0:n] are the positions of these documents inside `docinds`, and labels[n:] == -1,
                indicating non-relevance.
        :return: loss: scalar tensor. Mean loss per query
        """
        # WARNING: works assuming that `labels` aren't scores but integer indices of relevant documents padded with -1, e.g. [0, 1, 2, -1, ..., -1]

        _labels_values = labels.new_zeros(labels.shape, dtype=torch.float32)
        is_relevant = labels > -1
        _labels_values[is_relevant] = 1
        # NOTE: _labels_values = _labels_values / torch.sum(is_relevant, dim=1).unsqueeze(dim=1)
        # is equivalent but interestingly much slower than setting -Inf and computing Softmax; maybe due to CUDA Softmax code
        _labels_values[labels == -1] = float("-Inf")
        labels_probs = torch.nn.Softmax(dim=1)(_labels_values)

        predictions_logprobs = torch.nn.LogSoftmax(dim=1)(
            predictions
        )  # (batch, num_docs) log-distribution over docs
        # KLDivLoss expects predictions ('inputs') as log-probabilities and 'targets' as probabilities
        loss = torch.nn.KLDivLoss(reduction="batchmean")(predictions_logprobs, labels_probs)

        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model: Scorer, inputs: Dict, return_outputs=False):
        # try:
        #     device = f"cuda:{model.module.local_rank}" if torch.cuda.is_available() else "cpu"
        # except:
        #     device = f"cuda:{model.local_rank}" if torch.cuda.is_available() else "cpu"
        inputs = self._prepare_inputs(inputs)
        # loss_fct = RelevanceListnetLoss()
        loss_fct = nn.BCEWithLogitsLoss()
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = loss_fct(outputs, labels.float())

        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if type(outputs) == tuple:
            outputs = outputs[0]  # return only classification outputs
        outputs = outputs.float()
        outputs = nested_detach(outputs)

        gc.collect()
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds, **kwargs):
    labels = eval_preds.label_ids
    threshold = kwargs.get("threshold", 0.1)
    predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions)).numpy()

    f2 = np.array(
        [fbeta_score(labels[i], predictions[i] > threshold, beta=2) for i in range(len(labels))]
    ).mean()
    return {"f2": round(f2, 4)}

    # thresholds = np.arange(0.05, 0.2, 0.01)
    # scores = []
    # for threshold in thresholds:
    #     f2 = np.array(
    #         [
    #             fbeta_score(labels[i], predictions[i] > threshold, beta=2)
    #             for i in range(len(labels))
    #         ]
    #     ).mean()
    #     scores.append(f2)
    # f2 = np.max(scores)
    # return {"f2": round(f2, 4)}
