from loguru import logger
from tokenizers import trainers

from nlp_from_scratch.tokenizer.tokenizer_utils import (
    instanciate_blank_tokenizer,
    iterate_other_dataset,
)


class TokenizerTrainer:
    def __init__(
        self,
        vocab_size: int,
        max_inputs: int = 1000000,
    ):
        self.tokenizer = instanciate_blank_tokenizer()
        self.vocab_size = vocab_size
        self.max_inputs = max_inputs
        self.trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,  # Vocabulary size
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
            ],  # Special tokens
        )

    def train(self, dataset_name: str, split: str = "train", text_key: str = "text"):
        logger.info("Load dataset in streaming mode")
        iterator = iterate_other_dataset(
            name=dataset_name,
            max_iter=self.max_inputs,
            split=split,
            text_key=text_key,
        )
        logger.success("OK")
        logger.info("Start training tokenizer")
        self.tokenizer.train_from_iterator(iterator=iterator, trainer=self.trainer)
        logger.success("OK")

    def to_json(self, path: str):
        self.tokenizer.save(path=path)
