import click
from tokenizers import Tokenizer

from nlp_from_scratch.constants import TOKENIZER_SAVE_PATH, VOCAB_SIZE
from nlp_from_scratch.pre_training.dataset import DatasetMLM
from nlp_from_scratch.pre_training.model import BertMLM
from nlp_from_scratch.pre_training.pre_trainer import MLMTrainer
from nlp_from_scratch.pre_training.training_params import TrainingParams


@click.command()
@click.option("--model_name", default="bert_mlm_test", help="Name of the model.")
@click.option(
    "--dataset_name", default="monology/pile-uncopyrighted", help="Name of the dataset."
)
@click.option("--model_version", default=0, help="Version of the model.")
@click.option("--chunks_per_epoch", default=2**20, help="Number of chunks per epoch.")
@click.option("--offset", default=0, help="Offset for the dataset.")
@click.option(
    "--accumulate_grad_batches",
    default=8,
    help="Number of batches to accumulate gradients.",
)
@click.option("--log_every_n_steps", default=5, help="Log every n steps.")
@click.option("--batch_size", default=32, help="Batch size for training.")
@click.option("--n_iterations", default=10, help="Number of training iterations.")
@click.option("--vocab_size", default=VOCAB_SIZE, help="Vocabulary size.")
@click.option("--d_model", default=128, help="Dimension of the model.")
@click.option("--max_len", default=256, help="Maximum length of the input sequence.")
@click.option("--num_heads", default=4, help="Number of attention heads.")
@click.option("--num_layers", default=6, help="Number of layers in the model.")
def main(
    model_name,
    dataset_name,
    model_version,
    chunks_per_epoch,
    offset,
    accumulate_grad_batches,
    log_every_n_steps,
    batch_size,
    n_iterations,
    vocab_size,
    d_model,
    max_len,
    num_heads,
    num_layers,
):
    tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)
    dataset = DatasetMLM(
        name=dataset_name,
        tokenizer=tokenizer,
        max_len=max_len,
        start_offset=offset,
    )
    model = BertMLM(
        vocab_size=vocab_size,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
    )
    training_params = TrainingParams(
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        batch_size=batch_size,
        n_iterations=n_iterations,
    )
    trainer = MLMTrainer(
        model=model,
        dataset=dataset,
        model_name=model_name,
        model_version=model_version,
        chunks_per_epoch=chunks_per_epoch,
        max_len=max_len,
    )
    trainer.train(training_params=training_params)


if __name__ == "__main__":
    main()
