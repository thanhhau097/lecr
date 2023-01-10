from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_path: str = field(default="./data/supervised_correlations.csv", metadata={"help": "data path"})
    topic_path: str = field(default="./data/topics.csv", metadata={"help": "topic csv path"})
    content_path: str = field(default="./data/content.csv", metadata={"help": "content csv path"})
    fold: int = field(default=0, metadata={"help": "Fold"})
    max_len: int = field(default=512, metadata={"help": "max input length"})
