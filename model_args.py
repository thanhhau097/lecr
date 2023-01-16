from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """

    model_name: str = field(
        default="xlm-roberta-base", metadata={"help": "timm model name"}
    )
    tokenizer_name: str = field(
        default="xlm-roberta-base", metadata={"help": "timm model name"}
    )
    resume: Optional[str] = field(
        default=None, metadata={"help": "Path of model checkpoint"}
    )
    objective: str = field(
        default="classification", metadata={"help": "classification or siamese"}
    )
    is_sentence_transformers: bool = field(
        default=False, metadata={"help": "Use sentence transformers"}
    )
