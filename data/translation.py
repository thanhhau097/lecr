import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--index', type=int)

args = parser.parse_args()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

import time
from tqdm import tqdm

import pandas as pd
import numpy as np

import uuid


TASK = "translation"
CKPT = "facebook/nllb-200-distilled-600M"

model = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
tokenizer = AutoTokenizer.from_pretrained(CKPT)

device = 0 if torch.cuda.is_available() else -1
max_length = 512

lang_mapping = {
    'en': "eng_Latn",
    'es': "spa_Latn",
    'pt': "por_Latn",
    'ar': "arb_Arab",
    'fr': "fra_Latn",
    'bg': "bul_Cyrl",
    'sw': "swh_Latn",
    'gu': "guj_Gujr",
    'bn': "ben_Beng",
    'hi': "hin_Deva",
    'it': "ita_Latn",
    'zh': "zho_Hans",
}

lang_mapping_reverse = {v: k for k, v in lang_mapping.items()}

def translate(text, src_lang, tgt_lang):
    translation_pipeline = pipeline(
        TASK,
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        device=device
    )
    result = translation_pipeline(text)
    
    return result[0]["translation_text"] if result else ""


topic_df = pd.read_csv("topics.csv")
content_df = pd.read_csv("content.csv")

topic_df = topic_df.fillna('')
content_df = content_df.fillna('')


# # topics
# origin_ids = []
# ids = []
# titles = []
# descriptions = []
# channels = []
# categories = []
# levels = []
# languages = []
# parents = []
# has_contents = []

# for i, row in tqdm(topic_df[(args.index * 16000): ((args.index + 1) * 16000)].iterrows()):
#     src_lang = row["language"]
#     for lang_code, tgt_lang in lang_mapping.items():
#         if lang_code == src_lang or src_lang not in lang_mapping:
#             continue
#         origin_ids.append(row["id"])
#         ids.append(str(uuid.uuid4()))
        
#         if row["title"]:
#             translated_title = translate(row["title"], lang_mapping[src_lang], tgt_lang)
#             titles.append(translated_title)
#         else:
#             titles.append(row["title"])

#         if row["description"]:
#             translated_description = translate(row["description"], lang_mapping[src_lang], tgt_lang)
#             descriptions.append(translated_description)
#         else:
#             descriptions.append(row["description"])
            
#         channels.append(row["channel"])
#         categories.append(row["category"])
#         levels.append(row["level"])
#         languages.append(lang_mapping_reverse[tgt_lang])
#         parents.append(row["parent"])
#         has_contents.append(row["has_content"])
        
# new_df = pd.DataFrame({
#     "id": ids,
#     "origin_id": origin_ids,
#     "title": titles,
#     "description": descriptions,
#     "channel": channels,
#     "category": categories,
#     "level": levels,
#     "language": languages,
#     "origin_parent": parents,
#     "has_content": has_contents,
# })

# new_df.to_csv(f"translated_topics_{args.index}.csv")


# content
origin_ids = []
ids = []
titles = []
descriptions = []
texts = []
kinds = []
languages = []

for i, row in tqdm(content_df[(args.index * 16000): ((args.index + 1) * 16000)].iterrows()):
    src_lang = row["language"]
    for lang_code, tgt_lang in lang_mapping.items():
        if lang_code == src_lang or src_lang not in lang_mapping:
            continue
        origin_ids.append(row["id"])
        ids.append(str(uuid.uuid4()))
        
        if row["title"]:
            translated_title = translate(row["title"], lang_mapping[src_lang], tgt_lang)
            titles.append(translated_title)
        else:
            titles.append(row["title"])

        if row["description"]:
            translated_description = translate(row["description"], lang_mapping[src_lang], tgt_lang)
            descriptions.append(translated_description)
        else:
            descriptions.append(row["description"][:256])
        
        if row["text"]:
            translated_text = translate(row["text"][:256], lang_mapping[src_lang], tgt_lang)
            texts.append(translated_text)
        else:
            texts.append(row["text"])
            
        kinds.append(row["kind"])
        languages.append(lang_mapping_reverse[tgt_lang])

new_df = pd.DataFrame({
    "id": ids,
    "origin_id": origin_ids,
    "title": titles,
    "description": descriptions,
    "text": texts,
    "kind": kinds,
    "language": languages,
})

new_df.to_csv(f"translated_contents_{args.index}.csv")