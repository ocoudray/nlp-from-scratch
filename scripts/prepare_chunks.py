from nlp_from_scratch.constants import CHUNKS_POSITIONS_PATH
from nlp_from_scratch.dataset import TextDataset
from nlp_from_scratch.utils import get_chunks_from_text, filter_text
from tqdm.auto import tqdm
import json

text_dataset = TextDataset.load_from_save()
chunks = []
for k in tqdm(range(len(text_dataset.corpus))):
    chunks += get_chunks_from_text(filter_text(text_dataset.corpus[k]["text"]), k, text_dataset.tokenizer)
with open(CHUNKS_POSITIONS_PATH, "w") as f:
    json.dump(chunks, f)