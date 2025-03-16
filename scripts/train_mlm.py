from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from nlp_from_scratch.dataset import TextDataset
from nlp_from_scratch.model import SimpleTransformer
from nlp_from_scratch.utils import get_epoch_step, path_to_last_checkpoint

MODEL_NAME = "BERT_256_4_6_freq_mlm_wikipedia_complete_sentences2"
VERSION = 0

if __name__ == "__main__":
    logger.info("Load dataset")
    text_dataset = TextDataset.load_from_save()
    logger.success("OK")
    # logger.info("Log model")
    # cpkt_folder, filename = path_to_last_checkpoint(MODEL_NAME, VERSION)
    # logger.success("OK")
    for k in range(1):
        logger.info(f"Prepare chunks for subepoch {k}")
        text_dataset.prepare_chunks(n_chunks=3000000)
        logger.success("OK")
        dataloader = DataLoader(
            text_dataset,
            batch_size=64,
            shuffle=True,
        )
        logger.info("Define trainer")
        if k == -1:
            model = SimpleTransformer(d_model=256)
            epoch = -1
        else:
            cpkt_folder, filename = path_to_last_checkpoint(
                MODEL_NAME,
                VERSION,
            )
            epoch, step = get_epoch_step(filename)
            model = SimpleTransformer.load_from_checkpoint(
                f"{cpkt_folder}{filename}",
                d_model=256,
            )
            model.optimizer_state_path = f"{cpkt_folder}{filename}"
            print(epoch, step)
        tb_logger = TensorBoardLogger(
            "tb_logs",
            name=MODEL_NAME,
            version=VERSION,
        )
        trainer = Trainer(
            max_epochs=1,
            accumulate_grad_batches=4,
            logger=tb_logger,
            log_every_n_steps=1,
        )
        logger.success("OK")
        logger.info("Train model")
        trainer.fit(model=model, train_dataloaders=dataloader)
        logger.success("OK")
        model.cpu()
