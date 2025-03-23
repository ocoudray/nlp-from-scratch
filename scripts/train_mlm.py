import click
from tokenizers import Tokenizer

from nlp_from_scratch.constants import TOKENIZER_SAVE_PATH
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
@click.option(
    "--accumulate_grad_batches",
    default=8,
    help="Number of batches to accumulate gradients.",
)
@click.option("--log_every_n_steps", default=5, help="Log every n steps.")
@click.option("--batch_size", default=32, help="Batch size for training.")
@click.option("--n_iterations", default=10, help="Number of training iterations.")
def main(
    model_name,
    dataset_name,
    model_version,
    chunks_per_epoch,
    accumulate_grad_batches,
    log_every_n_steps,
    batch_size,
    n_iterations,
):
    tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)
    dataset = DatasetMLM(name=dataset_name, tokenizer=tokenizer)
    model = BertMLM(vocab_size=50000)
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
    )
    trainer.train(training_params=training_params)


if __name__ == "__main__":
    main()
