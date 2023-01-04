import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch.nn.functional import cosine_similarity

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class Model(nn.Module):
    def __init__(self, tokenizer, model_name="xlm-roberta-base"):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0

        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(model_name, config = self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, topic_inputs, content_inputs, labels=None):
        topic_features = self.feature(topic_inputs)
        content_features = self.feature(content_inputs)

        return cosine_similarity(topic_features, content_features)