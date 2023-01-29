# clean text
import re
import string
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def decontracted(phrase):

    # Specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # ..

    # General
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    # ..

    return phrase


def remove_punctuations(text):
    for punctuation in list(string.punctuation):
        text = text.replace(punctuation, "")
    return text


def clean_number(text):
    text = re.sub(r"(\d+)([a-zA-Z])", "\g<1> \g<2>", text)
    text = re.sub(r"(\d+) (th|st|nd|rd) ", "\g<1>\g<2> ", text)
    text = re.sub(r"(\d+),(\d+)", "\g<1>\g<2>", text)
    return text


def clean_whitespace(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_repeat_words(text):
    return re.sub(r"(\w*)(\w)\2(\w*)", r"\1\2\3", text)


def clean_text(text):
    text = str(text)
    text = decontracted(text)
    text = remove_punctuations(text)
    text = clean_number(text)
    text = clean_whitespace(text)

    return text


def get_pos_score(y_true, y_pred, top_k):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()[:top_k]))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


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



def get_processed_text_dict(topic_df, content_df, sep_token):
    # Fillna titles
    topic_df["title"].fillna("", inplace=True)
    content_df["title"].fillna("", inplace=True)

    # Fillna descriptions
    topic_df["description"].fillna("", inplace=True)
    content_df["description"].fillna("", inplace=True)
    content_df["text"].fillna("", inplace=True)

    # clean text
    print("Cleaning text data for topics")
    topic_df["title"] = topic_df["title"].apply(clean_text)
    topic_df["description"] = topic_df["description"].apply(clean_text)

    print("Cleaning text data for content")
    content_df["title"] = content_df["title"].apply(clean_text)
    content_df["description"] = content_df["description"].apply(clean_text)
    # content_df["text"] = content_df["text"].apply(clean_text)

    # parent and children information
    parents = defaultdict(lambda: [])
    children = defaultdict(lambda: [])
    topic_title_dict = {}

    all_topic_ids = set(topic_df.id.values)
    for i, row in tqdm(topic_df.iterrows()):
        if row["parent"] in all_topic_ids:
            parents[row["id"]].append(row["parent"])
            children[row["parent"]].append(row["id"])

        topic_title_dict[row["id"]] = row["title"]
    
    # get concatenated texts
    topic_dict = {}
    for i, (index, row) in tqdm(enumerate(topic_df.iterrows())):
        text = (
            "<|topic|>"
            + f"<|lang_{row['language']}|>"
            + f"<|category_{row['category']}|>"
            + f"<|level_{row['level']}|>"
        )
        text += (
            "<s_title>"
            + row["title"]
            + "</s_title>"
            + "<s_description>"
            + row["description"]
            + "</s_description>"
        )

        context_text = "<s_parent>" 
        max_successor = 10
        parent_id = parents.get(row["id"], [None])[0]

        i = 0
        while parent_id and i < max_successor:
            context_text += topic_title_dict[parent_id] + sep_token
            parent_id = parents.get(parent_id, [None])[0]
            i += 1

        context_text += "</s_parent>"
        
        if children.get(row["id"]):
            children_text = "<s_children>"
            for child_topic_id in children.get(row["id"]):
                children_text += topic_title_dict[child_topic_id] + sep_token
            children_text = children_text[:-(len(sep_token))] + "</s_children>"
        else:
            children_text = ""
        
        context_text += children_text
        topic_dict[row["id"]] = text + context_text

    content_dict = {}
    for i, (index, row) in tqdm(enumerate(content_df.iterrows())):
        text = "<|content|>" + f"<|lang_{row['language']}|>" + f"<|kind_{row['kind']}|>"
        text += (
            "<s_title>"
            + row["title"]
            + "</s_title>"
            + "<s_description>"
            + row["description"]
            + "</s_description>"
            + "<s_text>" + str(row["text"]) + "</s_text>"
        )
        content_dict[row["id"]] = text

    return topic_dict, content_dict

