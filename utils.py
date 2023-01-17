# clean text
import re
import string

import numpy as np


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
