from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import Model


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    0 if dissimilar;
    1 if similar.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) + 1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model: Model, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for k, v in inputs["topic_inputs"].items():
            inputs["topic_inputs"][k] = inputs["topic_inputs"][k].to(device)
        for k, v in inputs["content_inputs"].items():
            inputs["content_inputs"][k] = inputs["content_inputs"][k].to(device)

        outputs = model(inputs["topic_inputs"], inputs["content_inputs"])
        loss_fct = ContrastiveLoss()
        labels = inputs.get("labels")
        loss = loss_fct(outputs.view(-1), 1 - labels.float())
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
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        outputs = outputs.float()
        outputs = nested_detach(outputs)
        del inputs["images"]
        del inputs["features"]
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds):
    # predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions)).numpy()
    # fbeta_score = pfbeta_torch(
    #     eval_preds.label_ids[0],
    #     predictions,
    # )

    # auc = roc_auc_score(eval_preds.label_ids, predictions)
    # score = matthews_corrcoef(eval_preds.label_ids, predictions > 0.5)
    # return {"pF1": fbeta_score, "AUC": auc, "matthews_corrcoef": score}
    return {"f1": 0}

# =========================================================================================
# F2 score metric
# =========================================================================================
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)