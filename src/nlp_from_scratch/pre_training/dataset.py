import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from nlp_from_scratch.constants import MAX_LEN
from nlp_from_scratch.tokenizer.tokenizer_utils import iterate_other_dataset
from nlp_from_scratch.utils import get_chunks_from_text


class DatasetMLM(Dataset):
    def __init__(
        self,
        name: str = "monology/pile-uncopyrighted",
        split: str = "train",
        text_key: str = "text",
        start_offset: int = 0,
        tokenizer: Tokenizer | None = None,
        max_len: int = MAX_LEN,
    ):
        self.name = name
        self.split = split
        self.text_key = text_key
        self.chunks = []
        self.offset = start_offset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def prepare_chunks(
        self,
        n_chunks: int = 1000,
        offset: int = 0,
    ):
        chunks = []
        k = 0
        for text in iterate_other_dataset(
            name=self.name,
            split=self.split,
            text_key=self.text_key,
            max_iter=1000000,
        ):
            if k < offset:
                k += 1
                continue
            else:
                if len(chunks) >= n_chunks:
                    break
                chunks += get_chunks_from_text(
                    text,
                    self.tokenizer,
                    self.tokenizer.token_to_id("[CLS]"),
                    self.tokenizer.token_to_id("[PAD]"),
                    self.tokenizer.token_to_id("."),
                    max_len=self.max_len,
                )
                k += 1
        self.chunks = torch.stack(chunks)
        del chunks
        self.offset = k

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_index):
        tokens = self.chunks[chunk_index]
        return self.apply_mask(tokens)

    def apply_mask(self, input_ids, mask_prob=0.15):
        labels = input_ids.clone()  # Keep a copy of the original for labels (target)
        attention_mask_vector = labels == self.tokenizer.token_to_id("[PAD]")
        # Masking process: randomly mask a portion of the tokens
        masked = (
            (torch.rand(self.max_len) <= mask_prob)
            * (input_ids != self.tokenizer.token_to_id("[CLS]"))
            * (input_ids != self.tokenizer.token_to_id("[SEP]"))
            * (input_ids != self.tokenizer.token_to_id("[PAD]"))
        )
        input_ids = input_ids * (
            1 - masked.long()
        ) + masked.long() * self.tokenizer.token_to_id("[MASK]")

        # Return the modified input_ids (masked) and the labels (original)
        return input_ids, masked, attention_mask_vector, labels
