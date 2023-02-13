import gc
import random

import pandas as pd
from tqdm import tqdm


def build_new_supervised_df(knn_df, correlations):
    # Create lists for training
    topics_ids = []
    content_ids = []
    targets = []

    # Iterate over each topic in df
    mapping = set()

    # get all class 1 in correlations
    topic_ids = set(knn_df.topic_id.values)
    filtered_correlations = correlations[correlations["topic_id"].isin(topic_ids)]
    for i, row in tqdm(filtered_correlations.iterrows()):
        if str(row["content_ids"]) and str(row["content_ids"]) != "nan":
            content_ids = str(row["content_ids"]).split(" ")
            for content_id in content_ids:
                mapping.add((row["topic_id"], content_id, 1))

    for i, row in tqdm(knn_df.iterrows()):
        if str(row["content_ids"]) and str(row["content_ids"]) != "nan":
            content_ids = str(row["content_ids"]).split(" ")
            for content_id in content_ids:
                if (
                    row["topic_id"],
                    content_id,
                    1,
                ) not in mapping:  # because mapping already contains all positive cases
                    mapping.add((row["topic_id"], content_id, 0))

    # Build training dataset
    mapping = list(mapping)
    new_df = pd.DataFrame(
        {
            "topics_ids": [item[0] for item in mapping if item[1]],
            "content_ids": [item[1] for item in mapping if item[1]],
            "target": [item[2] for item in mapping if item[1]],
        }
    )
    # Release memory
    del topics_ids, content_ids
    gc.collect()
    return new_df


def build_triplet_df(knn_df, correlations):
    # Iterate over each topic in df
    mapping = set()

    # get all class 1 in correlations
    topic_ids = set(knn_df.topic_id.values)
    filtered_correlations = correlations[correlations["topic_id"].isin(topic_ids)]
    filtered_correlations["content_ids"].fillna(" ", inplace=True)

    topic2positivecontents = filtered_correlations.set_index("topic_id")["content_ids"].to_dict()
    topic2topkcontents = knn_df.set_index("topic_id")["content_ids"].to_dict()
    for topic_id, pos_content_ids in topic2positivecontents.items():
        pos_content_ids = pos_content_ids.split(" ")
        content_ids = topic2topkcontents.get(topic_id).split(" ")
        neg_content_ids = set(content_ids).difference(pos_content_ids)
        for neg_content_id in neg_content_ids:
            triplet = (topic_id, random.choice(pos_content_ids), neg_content_id)
            if triplet not in mapping:
                mapping.add(triplet)

    # Build training dataset
    mapping = list(mapping)
    new_df = pd.DataFrame(mapping, columns=["topics_ids", "pos_content_ids", "neg_content_ids"])

    # Release memory
    del mapping, topic2positivecontents, topic2topkcontents
    gc.collect()
    return new_df
