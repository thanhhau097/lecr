import torch
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import clean_text


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


def init_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(dict(additional_special_tokens=LANGUAGE_TOKENS + CATEGORY_TOKENS + LEVEL_TOKENS + KIND_TOKENS + OTHER_TOKENS))
    return tokenizer


class LECRDataset(Dataset):
    def __init__(self, correlation_df, topic_df, content_df, tokenizer_name='xlm-roberta-base', max_len=512):
        self.correlation_df = correlation_df
        self.topic_df = topic_df
        self.content_df = content_df
        self.labels = self.correlation_df.target.values
        self.topic_texts, self.content_texts = self.process_csv()

        self.tokenizer = init_tokenizer(tokenizer_name)
        self.max_len = max_len

    def process_csv(self):
        # Fillna titles
        self.topic_df['title'].fillna("", inplace = True)
        self.content_df['title'].fillna("", inplace = True)

        # Fillna descriptions
        self.topic_df['description'].fillna("", inplace = True)
        self.content_df['description'].fillna("", inplace = True)

        # clean text
        print("Cleaning text data for topics")
        self.topic_df["title"] = self.topic_df["title"].apply(clean_text)
        self.topic_df["description"] = self.topic_df["description"].apply(clean_text)

        print("Cleaning text data for content")
        self.content_df["title"] = self.content_df["title"].apply(clean_text)
        self.content_df["description"] = self.content_df["description"].apply(clean_text)
        # self.content_df["text"] = self.content_df["text"].apply(clean_text)

        # get concatenated texts
        topic_dict = {}
        for i, (index, row) in tqdm(enumerate(self.topic_df.iterrows())):
            text = "<|topic|>" + f"<|lang_{row['language']}|>" + f"<|category_{row['category']}|>" + f"<|level_{row['level']}|>"
            text += "<s_title>" + row["title"] + "</s_title>" + "<s_description>" + row["description"] + "</s_description>"
            topic_dict[row["id"]] = text

        content_dict = {}
        for i, (index, row) in tqdm(enumerate(self.content_df.iterrows())):
            text = "<|content|>" + f"<|lang_{row['language']}|>" + f"<|kind_{row['kind']}|>"
            text += "<s_title>" + row["title"] + "</s_title>" + "<s_description>" + row["description"] + "</s_description>" # + "<s_text>" + row["text"] + "</s_text>"
            content_dict[row["id"]] = text[:2048]

        # get text pairs
        topic_ids = self.correlation_df.topics_ids.values
        content_ids = self.correlation_df.content_ids.values

        topic_texts = []
        content_texts = []

        for topic_id in topic_ids:
            topic_texts.append(topic_dict[topic_id])

        for content_id in content_ids:
            content_texts.append(content_dict[content_id])

        return topic_texts, content_texts

    def __len__(self):
        return len(self.labels)

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
