import string

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from tqdm.auto import tqdm

from nlp_from_scratch.utils import retry_on_timeout


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


def is_latin_character(char: str) -> bool:
    """Check if a character is a Latin character or punctuation.

    Args:
        char (str): input character

    Returns:
        bool: whether the character is a Latin character or punctuation

    Examples:
        >>> is_latin_character("a")
        True
        >>> is_latin_character("世")
        False
        >>> is_latin_character("é")
        True
        >>> is_latin_character("!")
        True
        >>> is_latin_character(".")
        True
    """
    # Define Unicode ranges for Latin characters
    latin_range = (0x0000, 0x024F)

    # Check if the character is a Latin character or punctuation
    code_point = ord(char)
    return latin_range[0] <= code_point <= latin_range[1] or char in string.punctuation


def filter_non_latin_characters(text: str) -> str:
    """Filter out non-Latin characters from a given text, keeping punctuation.

    Args:
        text (str): input text

    Returns:
        str: text with only Latin characters and punctuation

    Examples:
        >>> filter_non_latin_characters("Hello, 世界!")
        'Hello, !'
        >>> filter_non_latin_characters("Café au lait.")
        'Café au lait.'
        >>> filter_non_latin_characters("Привет, мир!")
        ', !'
        >>> filter_non_latin_characters("123abc.")
        '123abc.'
    """
    return "".join(char for char in text if is_latin_character(char))


@retry_on_timeout(retries=5, delay=5)
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
    k = 0
    for item in tqdm(dataset, total=max_iter):
        if k >= max_iter:
            break
        k += 1
        filtered_text = filter_non_latin_characters(item[text_key])
        yield filtered_text
