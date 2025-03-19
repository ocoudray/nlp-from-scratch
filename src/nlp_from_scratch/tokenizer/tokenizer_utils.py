from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from tqdm.auto import tqdm


def instanciate_blank_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),  # Decompose Unicode characters
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(
                individual_digits=True
            ),  # Split digits into individual tokens
        ]
    )
    return tokenizer


def is_latin_character(char):
    # Define Unicode ranges for Latin, Cyrillic, and Greek scripts
    latin_range = (0x0000, 0x024F)

    code_point = ord(char)
    return latin_range[0] <= code_point <= latin_range[1]


def filter_non_latin_characters(text):
    return "".join(char for char in text if is_latin_character(char))


def iterate_other_dataset(
    name: str,
    max_iter: int,
    split: str = "train",
    text_key: str = "text",
):
    dataset = load_dataset(
        name,
        split=split,
        streaming=True,
    )
    iterator = dataset.__iter__()
    k = 0
    for item in tqdm(iterator, total=max_iter):
        if k >= max_iter:
            break
        k += 1
        filtered_text = filter_non_latin_characters(item[text_key])
        yield filtered_text
