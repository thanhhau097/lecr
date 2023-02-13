from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_path: str = field(
        default="./data/supervised_correlations.csv", metadata={"help": "data path"}
    )
    topic_path: str = field(
        default="./data/topics.csv", metadata={"help": "topic csv path"}
    )
    content_path: str = field(
        default="./data/content.csv", metadata={"help": "content csv path"}
    )
    correlation_path: str = field(
        default="./data/correlations.csv", metadata={"help": "correlation csv path"}
    )
    fold: int = field(default=0, metadata={"help": "Fold"})
    max_len: int = field(default=512, metadata={"help": "max input length"})
    use_content_pair: bool = field(
        default=False, metadata={"help": "Use content pair in data"}
    )
    top_k_neighbors: int = field(default=50, metadata={"help": "select top_k nearest neighbors for training and valiation set"})
    use_translated: bool = field(
        default=False, metadata={"help": "Use translated data while training"}
    )
    mix_translated: bool = field(
        default=False, metadata={"help": "Mix translated data and original data while training"}
    )
    translated_topic_path: str = field(
        default="./data/translated_topics.csv", metadata={"help": "translated topic csv path"}
    )
    translated_content_path: str = field(
        default="./data/translated_content.csv", metadata={"help": "translated content csv path"}
    )
    translated_correlation_path: str = field(
        default="./data/translated_correlations.csv", metadata={"help": "translated correlation csv path"}
    )
    use_sampler: bool = field(
        default=False, metadata={"help": "Use custom sampler"}
    )
    pn_ratio: int = field(default=1, metadata={"help": "Positive/negative ratio"})
    use_augmentation: bool = field(
        default=False, metadata={"help": "Use augmentation for text"}
    )
    use_no_content_topics: bool = field(
        default=False, metadata={"help": "Use augmentation for text"}
    )