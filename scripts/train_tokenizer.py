import click
from loguru import logger

from nlp_from_scratch.constants import TOKENIZER_SAVE_PATH
from nlp_from_scratch.tokenizer.tokenizer_trainer import TokenizerTrainer


@click.command()
@click.option(
    "--vocab_size",
    default=50000,
    help="Vocabulary size for the tokenizer.",
)
@click.option(
    "--max_inputs",
    default=1000000,
    help="Max number of texts to train the tokenizer on (one text = one item of dataset)",
)
@click.option(
    "--dataset_name",
    default="monology/pile-uncopyrighted",
    help="Name of the dataset to use.",
)
@click.option(
    "--split",
    default="train",
    help="Split of the dataset to use.",
)
@click.option(
    "--text_key",
    default="text",
    help="Key for the text data in the dataset.",
)
@click.option(
    "--path",
    default=TOKENIZER_SAVE_PATH,
    help="Path to save the trained tokenizer.",
)
def train_tokenizer(
    vocab_size,
    max_inputs,
    dataset_name,
    split,
    text_key,
    path,
):
    # Step 1: Initialize the WordPiece tokenizer
    logger.info("Set up tokenizer and trainer")
    tokenizer_trainer = TokenizerTrainer(vocab_size=vocab_size, max_inputs=max_inputs)
    logger.success("OK")

    # Step 2: Train tokenizer
    logger.info("Train tokenizer")
    tokenizer_trainer.train(
        dataset_name=dataset_name,
        split=split,
        text_key=text_key,
    )
    logger.success("OK")

    # Step 3: Save tokenizer
    logger.info("Save tokenizer")
    tokenizer_trainer.to_json(path)
    logger.success("OK")


if __name__ == "__main__":
    train_tokenizer()
