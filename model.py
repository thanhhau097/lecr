import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from dataset import init_tokenizer


class MeanPooling(nn.Module):
    def __init__(self, is_sentence_transformers=False):
        super(MeanPooling, self).__init__()
        self.is_sentence_transformers = is_sentence_transformers

    def forward(self, outputs, attention_mask):
        if self.is_sentence_transformers:
            token_embeddings = outputs[0] #First element of outputs contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings
        else:
            last_hidden_state = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings
    

class Model(nn.Module):
    def __init__(self, tokenizer_name="xlm-roberta-base", model_name="xlm-roberta-base", objective="classification", is_sentence_transformers=False):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name, config = self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pool = MeanPooling(is_sentence_transformers=is_sentence_transformers)

        self.objective = objective
        if self.objective in ["classification", "both"]:
            self.fc = nn.Linear(self.model.config.hidden_size * 3, 1)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        feature = self.pool(outputs, inputs['attention_mask'])
        return feature

    def forward(self, topic_inputs, content_inputs, labels=None):
        topic_features = self.feature(topic_inputs)
        content_features = self.feature(content_inputs)

        if self.objective == "classification":
            return self.fc(torch.cat([topic_features, content_features, topic_features - content_features], -1))
        elif self.objective == "siamese":
            return topic_features, content_features
        elif self.objective == "both":
            return self.fc(torch.cat([topic_features, content_features, topic_features - content_features], -1)), (topic_features, content_features)
        else:
            raise ValueError("objective should be classification/siamese/both")