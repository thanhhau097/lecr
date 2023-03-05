import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoConfig, AutoModel

from dataset import init_tokenizer


class MeanPooling(nn.Module):
    def __init__(self, is_sentence_transformers=False):
        super(MeanPooling, self).__init__()
        self.is_sentence_transformers = is_sentence_transformers

    def forward(self, outputs, attention_mask):
        if self.is_sentence_transformers:
            # First element of outputs contains all token embeddings
            token_embeddings = outputs[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sentence_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        return mean_embeddings


class SentenceTransformerModel(nn.Module):
    def __init__(
        self,
        tokenizer_name="sentence-transformers/sentence-t5-base",
        model_name="sentence-transformers/sentence-t5-base",
        objective="siamese",
        is_sentence_transformers=True,
    ):
        super(SentenceTransformerModel, self).__init__()

        self.objective = objective
        self.is_sentence_transformers = is_sentence_transformers
        self.tokenizer = init_tokenizer(tokenizer_name)
        self.model = SentenceTransformer(model_name)
        self.model[0].auto_model.resize_token_embeddings(len(self.tokenizer))

    def feature(self, inputs):
        outputs = self.model(inputs)
        return F.normalize(outputs["sentence_embedding"])

    def forward(self, topic_inputs, content_inputs, combined_inputs=None, labels=None):
        topic_features = self.feature(topic_inputs)
        content_features = self.feature(content_inputs)
        return topic_features, content_features


class Model(nn.Module):
    def __init__(
        self,
        tokenizer_name="xlm-roberta-base",
        model_name="xlm-roberta-base",
        objective="classification",
        is_sentence_transformers=False,
    ):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pool = MeanPooling(is_sentence_transformers=is_sentence_transformers)

        self.objective = objective
        if self.objective in ["classification", "both"]:
            self.fc = nn.Linear(self.model.config.hidden_size, 1)

    def feature(self, inputs):
        outputs = self.model(**inputs)

        feature = self.pool(outputs, inputs["attention_mask"])
        return feature

    def forward(
        self,
        topic_inputs,
        content_inputs,
        combined_inputs=None,
        neg_content_inputs=None,
        labels=None,
    ):
        if self.objective == "classification":
            combined_features = self.feature(combined_inputs)
            return self.fc(combined_features)

        topic_features = self.feature(topic_inputs)
        content_features = self.feature(content_inputs)

        if self.objective == "siamese":
            return topic_features, content_features

        neg_content_features = self.feature(neg_content_inputs)
        return topic_features, content_features, neg_content_features
