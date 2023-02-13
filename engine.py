import gc
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from losses import OnlineContrastiveLoss, TripletLoss
from model import Model


class CustomTrainer(Trainer):
    def compute_loss(
        self, model: Model, inputs: Dict[str, Dict[str, Tensor]], return_outputs=False
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        labels = inputs.get("labels")
        for k, _ in inputs["topic_inputs"].items():
            inputs["topic_inputs"][k] = inputs["topic_inputs"][k].to(device)
        for k, _ in inputs["content_inputs"].items():
            inputs["content_inputs"][k] = inputs["content_inputs"][k].to(device)

        if model.objective == "siamese":
            outputs = model(inputs["topic_inputs"], inputs["content_inputs"])
            loss_fct = OnlineContrastiveLoss()
            loss = loss_fct(outputs, labels.float())

        elif model.objective == "triplet":
            for k, _ in inputs["neg_content_inputs"].items():
                inputs["neg_content_inputs"][k] = inputs["neg_content_inputs"][k].to(device)
            outputs = model(
                inputs["topic_inputs"],
                inputs["content_inputs"],
                neg_content_inputs=inputs["neg_content_inputs"],
            )
            loss_fct = TripletLoss()
            loss = loss_fct(outputs[0], outputs[1], outputs[2])

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
        del inputs["topic_inputs"]
        del inputs["content_inputs"]
        del inputs["combined_inputs"]

        gc.collect()
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds):
    try:
        predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions)).numpy()
        auc = roc_auc_score(eval_preds.label_ids, predictions)
        accuracy = accuracy_score(eval_preds.label_ids, predictions > 0.5)
        f1 = f1_score(eval_preds.label_ids, predictions > 0.5)
        return {"AUC": auc, "acc": accuracy, "f1": f1}
    except:
        return {"f1": 0}
