import os
import re
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from nlp_from_scratch.constants import (
    REGEX_EPOCH_STEPS,
    TB_LOGS_PATH_PREFIX,
    VOCAB_SIZE,
)


# Function to filter text
def filter_text(text):
    # Keep only alphabet and punctuation
    return re.sub(
        r"[^a-zA-Z0-9\s.,!?;:\"'(){}\[\]<>–—_‘’“”]+",
        " ",
        text,
    )


# Apply this to the dataset
def get_filtered_texts(data):
    for entry in data:
        yield filter_text(entry["text"])


def convert_counter_to_npy(c: Counter) -> np.ndarray:
    vect = np.zeros(VOCAB_SIZE)
    for key in c.keys():
        vect[key] = c[key]
    return vect / vect.sum()


def get_chunks_from_text(text, tokenizer, cls_token, pad_token):
    tokens = torch.tensor([11] + tokenizer.encode(text).ids, dtype=torch.long)
    dot_positions = torch.where(tokens == 11)[0]
    chunks = []
    for id_start in range(len(dot_positions)):
        id_end = id_start
        while (
            id_end + 1 < len(dot_positions)
            and dot_positions[id_end + 1] - dot_positions[id_start] < 255
        ):
            id_end += 1
        # chunks.append((text_id, int(dot_positions[id_start]), int(dot_positions[id_end])))
        encoded = F.pad(
            tokens[int(dot_positions[id_start]) + 1 : int(dot_positions[id_end]) + 1],
            pad=(1, 0),
            mode="constant",
            value=cls_token,
        )
        chunks.append(
            F.pad(
                encoded,
                pad=(0, 255 - dot_positions[id_end] + dot_positions[id_start]),
                mode="constant",
                value=pad_token,
            )
        )
    return chunks


def path_to_last_checkpoint(
    model_name: str,
    version: int,
):
    cpkt_folder = f"{TB_LOGS_PATH_PREFIX}{model_name}/version_{version}/checkpoints/"
    checkpoint_files = sorted(os.listdir(cpkt_folder))
    print(checkpoint_files)
    cpkt_filename = checkpoint_files[-1]
    print(cpkt_filename)
    return cpkt_folder, cpkt_filename


def get_epoch_step(cpkt_filename: str):
    re_match = re.findall(REGEX_EPOCH_STEPS, cpkt_filename)[0]
    epoch, step = int(re_match[0]), int(re_match[1])
    return epoch, step
