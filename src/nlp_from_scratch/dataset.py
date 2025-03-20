import json
import re

import numpy as np
import numpy.random as npr
import torch
from datasets import load_dataset
from loguru import logger
from tokenizers import Tokenizer
from torch.utils.data.dataset import Dataset

from nlp_from_scratch.constants import (
    DUMMY_JSON_FILE_PATH,
    FREQUENCIES_SAVE_PATH,
    MAX_LEN,
    TOKENIZER_SAVE_PATH,
)
from nlp_from_scratch.utils import get_chunks_from_text


# Function to filter text
def filter_text(text):
    # Keep only alphabet and punctuation
    return re.sub(r"[^\da-zA-Z.,!?;:()'\"]+", " ", text)


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        corpus: Dataset,
        frequencies: np.ndarray,
        chunks: list[list[int]],
    ):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.frequencies = frequencies
        self.chunks = chunks

    @property
    def cls_token(self):
        return self.tokenizer.token_to_id("[CLS]")

    @property
    def sep_token(self):
        return self.tokenizer.token_to_id("[SEP]")

    @property
    def mask_token(self):
        return self.tokenizer.token_to_id("[MASK]")

    @property
    def pad_token(self):
        return self.tokenizer.token_to_id("[PAD]")

    def prepare_chunks(self, n_chunks: int = 1000):
        logger.info("Preparing chunks")
        chunks = []
        while len(chunks) < n_chunks:
            k = npr.randint(0, len(self.corpus))
            chunks += get_chunks_from_text(
                self.corpus[k]["text"], self.tokenizer, self.cls_token, self.pad_token
            )
        self.chunks = torch.stack(chunks)
        del chunks
        logger.success(f"OK: {len(self.chunks)} prepared")

    @staticmethod
    def load_from_save(tokenizer_path: str = TOKENIZER_SAVE_PATH):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        corpus = load_dataset("wikipedia", "20220301.en", split="train")
        frequencies = np.load(FREQUENCIES_SAVE_PATH)
        frequencies[:5] = 1.0
        # try:
        #     with open(CHUNKS_POSITIONS_PATH, "r") as f:
        #         chunks = json.load(f)
        # except:
        #     chunks = []
        chunks = []
        return TextDataset(
            tokenizer=tokenizer, corpus=corpus, frequencies=frequencies, chunks=chunks
        )

    @staticmethod
    def load_dummy(tokenizer_path: str = TOKENIZER_SAVE_PATH):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(DUMMY_JSON_FILE_PATH, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        corpus = data * 32
        frequencies = np.load(FREQUENCIES_SAVE_PATH)
        frequencies[:5] = 1.0
        return TextDataset(
            tokenizer=tokenizer, corpus=corpus, frequencies=frequencies, chunks=[]
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_index):
        tokens = self.chunks[chunk_index]
        return self.apply_mask(tokens)

    def apply_mask(self, input_ids, mask_prob=0.15):
        labels = input_ids.clone()  # Keep a copy of the original for labels (target)
        attention_mask_vector = labels == self.pad_token
        # Masking process: randomly mask a portion of the tokens
        masked = (
            (torch.rand(MAX_LEN) <= mask_prob)
            * (input_ids != self.cls_token)
            * (input_ids != self.sep_token)
            * (input_ids != self.pad_token)
        )
        input_ids = input_ids * (1 - masked.long()) + masked.long() * self.mask_token

        # Return the modified input_ids (masked) and the labels (original)
        return input_ids, masked, attention_mask_vector, labels
