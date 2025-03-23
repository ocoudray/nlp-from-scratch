from tokenizers import Tokenizer

from nlp_from_scratch.constants import TOKENIZER_SAVE_PATH
from nlp_from_scratch.pre_training.dataset import DatasetMLM
from nlp_from_scratch.pre_training.model import BertMLM
from nlp_from_scratch.pre_training.pre_trainer import MLMTrainer
from nlp_from_scratch.pre_training.training_params import TrainingParams

MODEL_NAME = "BERT_256_4_6_freq_mlm_wikipedia_complete_sentences2"
DATASET_NAME = "monology/pile-uncopyrighted"
VERSION = 0
DEFAULT_EPOCH = -1

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)
    dataset = DatasetMLM(name=DATASET_NAME, tokenizer=tokenizer)
    model = BertMLM(vocab_size=50000)
    training_params = TrainingParams(
        accumulate_grad_batches=8,
        log_every_n_steps=5,
        batch_size=32,
        n_iterations=10,
    )
    trainer = MLMTrainer(
        model=model,
        dataset=dataset,
        model_name="bert_mlm_test",
        model_version=0,
        chunks_per_epoch=2**20,
    )
    trainer.train(training_params=training_params)
