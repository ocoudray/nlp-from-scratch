from nlp_from_scratch.dataset import TextDataset
from collections import Counter
from tqdm.auto import tqdm
import numpy as np
from nlp_from_scratch.constants import FREQUENCIES_SAVE_PATH
from nlp_from_scratch.utils import convert_counter_to_npy

text_dataset = TextDataset.load_from_save()
counter = Counter()
for k in tqdm(range(len(text_dataset))):
    counter.update(text_dataset.tokenizer.encode(text_dataset.corpus[k]["text"]).ids)
frequencies = convert_counter_to_npy(counter)
np.save(FREQUENCIES_SAVE_PATH, frequencies)