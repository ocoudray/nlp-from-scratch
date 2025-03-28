import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch.utils.data import DataLoader

from nlp_from_scratch.constants import TB_LOGS_PATH_PREFIX
from nlp_from_scratch.pre_training.dataset import DatasetMLM
from nlp_from_scratch.pre_training.model import BertMLM
from nlp_from_scratch.pre_training.training_params import TrainingParams
from nlp_from_scratch.utils import path_to_last_checkpoint


class MLMTrainer:
    def __init__(
        self,
        dataset: DatasetMLM,
        model: BertMLM,
        model_name: str,
        model_version: int,
        chunks_per_epoch: int,
        max_len: int = 256,
    ):
        self.dataset = dataset
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.chunks_per_epoch = chunks_per_epoch
        self.logger = TensorBoardLogger(
            TB_LOGS_PATH_PREFIX,
            name=self.model_name,
            version=self.model_version,
        )
        self.trainer = None
        self.max_len = max_len

    def set_up_new_trainer(self, training_params: TrainingParams, max_epochs: int = 1):
        self.trainer = Trainer(
            logger=self.logger,
            accumulate_grad_batches=training_params.accumulate_grad_batches,
            log_every_n_steps=training_params.log_every_n_steps,
            max_epochs=max_epochs,
        )

    def get_last_checkpoint(self) -> str:
        folder_path, filename = path_to_last_checkpoint(
            self.model_name,
            self.model_version,
        )
        return f"{folder_path}{filename}"

    def save_checkpoint(self, checkpoint_path: str):
        """Save the model checkpoint manually."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(
        self,
        training_params: TrainingParams,
    ):
        for k in range(training_params.n_iterations):
            if k >= 1:
                self.model = BertMLM.load_from_checkpoint(
                    **self.model.hparams,
                    checkpoint_path=self.get_last_checkpoint(),
                )
                self.model.optimizer_state_path = self.get_last_checkpoint()
            logger.info(f"Prepare chunks for subepoch {k}")
            self.dataset.prepare_chunks(
                n_chunks=self.chunks_per_epoch,
                offset=self.dataset.offset,
            )
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=training_params.batch_size,
                shuffle=True,
            )
            logger.success("OK")
            logger.info("Train model")
            self.set_up_new_trainer(training_params, max_epochs=1)
            self.trainer.fit(
                model=self.model,
                train_dataloaders=dataloader,
            )
            logger.success("OK")
