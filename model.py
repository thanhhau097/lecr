import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import cosine_similarity
from transformers import AutoConfig, AutoModel

from dataset import init_tokenizer


class MeanPooling(nn.Module):
    def __init__(self, is_sentence_transformers=False):
        super(MeanPooling, self).__init__()
        self.is_sentence_transformers = is_sentence_transformers

    def forward(self, outputs, attention_mask):
        if self.is_sentence_transformers:
            token_embeddings = outputs[0]  # First element of outputs contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sentence_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings
        else:
            last_hidden_state = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings


class Model(nn.Module):
    def __init__(
        self,
        tokenizer_name="xlm-roberta-base",
        model_name="xlm-roberta-base",
        objective="classification",
        is_sentence_transformers=False,
        local_rank=-1,
    ):
        super(Model, self).__init__()
        self.local_rank = local_rank
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

    def forward(self, topic_inputs, content_inputs, combined_inputs, labels=None):
        if self.objective == "classification":
            combined_features = self.feature(combined_inputs)
            return self.fc(combined_features)
        elif self.objective == "siamese":
            topic_features = self.feature(topic_inputs)
            content_features = self.feature(content_inputs)
            return topic_features, content_features
        elif self.objective == "both":
            topic_features = self.feature(topic_inputs)
            content_features = self.feature(content_inputs)
            combined_features = self.feature(combined_inputs)
            return self.fc(combined_features), (topic_features, content_features)
        else:
            raise ValueError("objective should be classification/siamese/both")


class Scorer(nn.Module):
    def __init__(
        self,
        d_model,
        tokenizer_name="xlm-roberta-base",
        model_name="xlm-roberta-base",
        objective="classification",
        is_sentence_transformers=False,
    ):
        super(Scorer, self).__init__()

        self.query_encoder = Model(tokenizer_name, model_name, objective, is_sentence_transformers)

        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, content_embs, topic_inputs, labels=None):
        topic_embs = F.normalize(self.query_encoder.feature(topic_inputs), dim=-1)
        content_embs = F.normalize(content_embs, dim=-1)

        scores = torch.matmul(content_embs, topic_embs.unsqueeze(-1))
        topic_embs = topic_embs[:, None].expand_as(content_embs)
        scores = torch.cat([scores, content_embs, topic_embs], dim=-1)
        scores = self.scorer(scores)
        scores = scores.squeeze(-1)
        return scores

        # attn_output = self.multihead_attn(
        #     content_embs, topic_embs.unsqueeze(1), topic_embs.unsqueeze(1)
        # )[0]
        # attn_output = self.scorer(attn_output)
        # attn_output = attn_output.squeeze(-1)
        # return attn_output
