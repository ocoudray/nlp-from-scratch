from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from nlp_from_scratch.utils import get_filtered_texts
from nlp_from_scratch.constants import VOCAB_SIZE, TOKENIZER_SAVE_PATH
from loguru import logger


# * Load Wikipedia dataset
logger.info("Load Wikipedia dataset")
dataset = load_dataset("wikipedia", "20220301.en", split="train")
logger.success("OK")

# Step 1: Initialize the WordPiece tokenizer
logger.info("Set up tokenizer and trainer")
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Step 2: Set up normalization (e.g., lowercase, strip accents)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),  # Decompose Unicode characters
    normalizers.Lowercase(),  # Convert to lowercase
    normalizers.StripAccents()  # Remove accents
])

# Step 3: Define pre-tokenization (e.g., split by whitespace)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Step 4: Set up the trainer
trainer = trainers.WordPieceTrainer(
    vocab_size=VOCAB_SIZE,  # Vocabulary size
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # Special tokens
)
logger.success("OK")

# Step 5: Train the tokenizer on your text files
logger.info("Train tokenizer")
tokenizer.train_from_iterator(get_filtered_texts(dataset), trainer)
logger.success("OK")

# Step 6: Save as json
logger.info("Save as json")
tokenizer.save(TOKENIZER_SAVE_PATH)
logger.success("OK")