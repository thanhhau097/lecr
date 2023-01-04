from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_path: str = field(default="./data/siamese_train.csv", metadata={"help": "data path"})
    fold: int = field(default=0, metadata={"help": "Fold"})
    max_len: int = field(default=512, metadata={"help": "max input length"})
