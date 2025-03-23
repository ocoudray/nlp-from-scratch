import os
import re
import time
from collections import Counter
from functools import wraps

import numpy as np
import torch
from torch.nn import functional as F

from nlp_from_scratch.constants import (
    MAX_LEN,
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


def get_chunks_from_text(
    text,
    tokenizer,
    cls_token,
    pad_token,
    dot_token,
    max_len=MAX_LEN,
):
    tokens = torch.tensor([11] + tokenizer.encode(text).ids, dtype=torch.long)
    dot_positions = torch.where(tokens == dot_token)[0]
    chunks = []
    for id_start, _ in enumerate(dot_positions):
        id_end = id_start
        while (
            id_end + 1 < len(dot_positions)
            and dot_positions[id_end + 1] - dot_positions[id_start] < max_len - 1
        ):
            id_end += 1
        encoded = F.pad(
            tokens[int(dot_positions[id_start]) + 1 : int(dot_positions[id_end]) + 1],
            pad=(1, 0),
            mode="constant",
            value=cls_token,
        )
        chunks.append(
            F.pad(
                encoded,
                pad=(0, max_len - 1 - dot_positions[id_end] + dot_positions[id_start]),
                mode="constant",
                value=pad_token,
            )
        )
    return chunks


def path_to_last_checkpoint(model_name: str, version: int):
    """
    Returns the path to the last modified checkpoint file in the given folder.

    Args:
        model_name (str): The name of the model.
        version (int): The version of the model.

    Returns:
        tuple: A tuple containing the folder path and the filename of
        the last modified checkpoint.
    """
    cpkt_folder = f"{TB_LOGS_PATH_PREFIX}{model_name}/version_{version}/checkpoints/"

    # List all files in the checkpoint folder
    checkpoint_files = os.listdir(cpkt_folder)

    # Filter out non-files (e.g., directories) and get the last modified file
    last_modified_file = max(
        (
            os.path.join(cpkt_folder, f)
            for f in checkpoint_files
            if os.path.isfile(os.path.join(cpkt_folder, f))
        ),
        key=os.path.getmtime,
        default=None,
    )

    if last_modified_file is None:
        raise FileNotFoundError("No checkpoint files found in the directory.")

    # Extract the filename from the full path
    cpkt_filename = os.path.basename(last_modified_file)

    return cpkt_folder, cpkt_filename


def get_epoch_step(cpkt_filename: str):
    re_match = re.findall(REGEX_EPOCH_STEPS, cpkt_filename)[0]
    epoch, step = int(re_match[0]), int(re_match[1])
    return epoch, step


def retry_on_timeout(retries: int = 3, delay: int = 5):
    """
    Decorator to retry a function upon encountering a ReadTimeout error.

    Args:
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(
                        f"""Attempt {attempt + 1} failed with error: {e}.
                        Retrying in {delay} seconds..."""
                    )
                    time.sleep(delay)
            # If all retries fail, raise an exception
            raise Exception(
                f"Failed to execute {func.__name__} after {retries} attempts."
            )

        return wrapper

    return decorator
