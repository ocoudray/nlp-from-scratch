TOKENIZER_SAVE_PATH = "/Users/ocoudray/Documents/Projects/nlp-from-scratch/notebooks/wordpiece_tokenizer.json"
FREQUENCIES_SAVE_PATH = (
    "/Users/ocoudray/Documents/Projects/nlp-from-scratch/notebooks/frequencies.npy"
)
CHUNKS_POSITIONS_PATH = (
    "/Users/ocoudray/Documents/Projects/nlp-from-scratch/notebooks/chunk_positions.json"
)
TB_LOGS_PATH_PREFIX = "/Users/ocoudray/Documents/Projects/nlp-from-scratch/tb_logs/"
REGEX_EPOCH_STEPS = r"epoch=(?P<epoch>\d+)-step=(?P<step>\d+)\.ckpt"
VOCAB_SIZE = 10_000
MAX_LEN = 256
STRIDE = 32
N_STEPS = 100_000
START_LR = 5e-4
END_LR = 5e-5
