from transformers import AutoTokenizer

LANGUAGE_TOKENS = [
    "<|lang_pnb|>",
    "<|lang_tr|>",
    "<|lang_ur|>",
    "<|lang_bn|>",
    "<|lang_hi|>",
    "<|lang_en|>",
    "<|lang_kn|>",
    "<|lang_km|>",
    "<|lang_zh|>",
    "<|lang_gu|>",
    "<|lang_ta|>",
    "<|lang_my|>",
    "<|lang_fr|>",
    "<|lang_swa|>",
    "<|lang_or|>",
    "<|lang_mul|>",
    "<|lang_fil|>",
    "<|lang_sw|>",
    "<|lang_es|>",
    "<|lang_pt|>",
    "<|lang_pl|>",
    "<|lang_ru|>",
    "<|lang_mr|>",
    "<|lang_it|>",
    "<|lang_ar|>",
    "<|lang_bg|>",
    "<|lang_te|>",
    "<|lang_as|>",
]


CATEGORY_TOKENS = [
    "<|category_supplemental|>",
    "<|category_aligned|>",
    "<|category_source|>",
]

LEVEL_TOKENS = [
    "<|level_0|>",
    "<|level_1|>",
    "<|level_2|>",
    "<|level_3|>",
    "<|level_4|>",
    "<|level_5|>",
    "<|level_6|>",
    "<|level_7|>",
    "<|level_8|>",
    "<|level_9|>",
    "<|level_10|>",
]

KIND_TOKENS = [
    "<|kind_document|>",
    "<|kind_video|>",
    "<|kind_html5|>",
    "<|kind_exercise|>",
    "<|kind_audio|>",
]

OTHER_TOKENS = [
    "<|topic|>",
    "<|content|>",
    "<s_title>",
    "</s_title>",
    "<s_description>",
    "</s_description>",
    "<s_text>",
    "</s_text>",
]

RELATION_TOKENS = [
    "<s_parent>",
    "</s_parent>",
    "<s_children>",
    "</s_children>",
]


def init_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(
        dict(
            additional_special_tokens=LANGUAGE_TOKENS
            + CATEGORY_TOKENS
            + LEVEL_TOKENS
            + KIND_TOKENS
            + OTHER_TOKENS
            + RELATION_TOKENS
        )
    )
    return tokenizer
