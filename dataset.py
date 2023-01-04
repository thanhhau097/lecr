import torch
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer


LANGUAGE_TOKENS = [
    '<|lang_pnb|>',
    '<|lang_tr|>',
    '<|lang_ur|>',
    '<|lang_bn|>',
    '<|lang_hi|>',
    '<|lang_en|>',
    '<|lang_kn|>',
    '<|lang_km|>',
    '<|lang_zh|>',
    '<|lang_gu|>',
    '<|lang_ta|>',
    '<|lang_my|>',
    '<|lang_fr|>',
    '<|lang_swa|>',
    '<|lang_or|>',
    '<|lang_mul|>',
    '<|lang_fil|>',
    '<|lang_sw|>',
    '<|lang_es|>',
    '<|lang_pt|>',
    '<|lang_pl|>',
    '<|lang_ru|>',
    '<|lang_mr|>',
    '<|lang_it|>',
    '<|lang_ar|>',
    '<|lang_bg|>',
    '<|lang_te|>',
    '<|lang_as|>'
]


CATEGORY_TOKENS = ['<|category_supplemental|>', '<|category_aligned|>', '<|category_source|>']

LEVEL_TOKENS = [
    '<|level_0|>',
    '<|level_1|>',
    '<|level_2|>',
    '<|level_3|>',
    '<|level_4|>',
    '<|level_5|>',
    '<|level_6|>',
    '<|level_7|>',
    '<|level_8|>',
    '<|level_9|>',
    '<|level_10|>'
]

KIND_TOKENS = [
    '<|kind_document|>',
    '<|kind_video|>',
    '<|kind_html5|>',
    '<|kind_exercise|>',
    '<|kind_audio|>'
]

OTHER_TOKENS = [
    "<|topic|>", "<|content|>", "<s_title>", "</s_title>", "<s_description>", "</s_description>", "<s_text>", "</s_text>"
]


class LECRDataset(Dataset):
    def __init__(self, df, tokenizer_name='xlm-roberta-base', max_len=512):
        self.df = df
        self.topic_texts = self.df.topic_text.values
        self.content_texts = self.df.content_text.values
        self.labels = self.df.target.values

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens(dict(additional_special_tokens=LANGUAGE_TOKENS + CATEGORY_TOKENS + LEVEL_TOKENS + KIND_TOKENS + OTHER_TOKENS))
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        topic_text = self.topic_texts[idx]
        content_text = self.content_texts[idx]
        label = self.labels[idx]
        
        # topic
        topic_inputs = self.tokenizer.encode_plus(
            topic_text, 
            return_tensors = None, 
            add_special_tokens = True, 
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation = True
        )
        for k, v in topic_inputs.items():
            topic_inputs[k] = torch.tensor(v, dtype = torch.long)
            
        # content
        content_inputs = self.tokenizer.encode_plus(
            content_text, 
            return_tensors = None, 
            add_special_tokens = True, 
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation = True
        )
        for k, v in content_inputs.items():
            content_inputs[k] = torch.tensor(v, dtype = torch.long)
            
        return topic_inputs, content_inputs, label


def collate_fn(batch):
    batch = default_collate(batch)
    
    topic_inputs, content_inputs, labels = batch
    mask_len = int(topic_inputs["attention_mask"].sum(axis=1).max())
    for k, v in topic_inputs.items():
        topic_inputs[k] = topic_inputs[k][:,:mask_len]
        
    mask_len = int(content_inputs["attention_mask"].sum(axis=1).max())
    for k, v in content_inputs.items():
        content_inputs[k] = content_inputs[k][:,:mask_len]

    return {
        "topic_inputs": batch[0],
        "content_inputs": batch[1],
        "labels": batch[2]
    }
