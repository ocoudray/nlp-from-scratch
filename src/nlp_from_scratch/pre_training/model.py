import math

import torch
from lightning import LightningModule
from torch import nn

from nlp_from_scratch.constants import (
    D_MODEL,
    DIM_FF,
    MAX_LEN,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(VOCAB_SIZE) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))  # Add & Norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))  # Add & Norm
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dim_ff=DIM_FF,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=max_len, vocab_size=vocab_size
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dim_ff) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x


class BertMLM(LightningModule):
    def __init__(
        self,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dim_ff=DIM_FF,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_len=max_len,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)  # Output layer for classification
        self.dropout = nn.Dropout(0.1)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.training_step_losses = []
        self.optimizer_state_path = None
        self.save_hyperparameters()

    def get_embeddings(self, x, mask=None):
        return self.transformer_encoder(x, mask=mask)

    def _classify_output(self, x):
        return self.fc_out(x)

    def forward(self, x, mask=None):  # pylint: disable=arguments-differ
        x = self.get_embeddings(x, mask=mask)
        return self._classify_output(x)  # Final output

    def training_step(self, batch):  # pylint: disable=arguments-differ
        # training_step defines the train loop.
        inputs, mask, attention_mask, labels = batch
        output = self.forward(inputs, mask=attention_mask)
        loss_matrix = self.loss_fn(
            output.view(-1, self.vocab_size), labels.view(-1)
        ).view(labels.size())
        loss = loss_matrix[mask].mean()
        self.training_step_losses.append(loss.item())
        if (
            self.trainer.fit_loop.epoch_loop.batch_progress.current.processed + 1
        ) % self.trainer.accumulate_grad_batches == 0 and len(
            self.training_step_losses
        ) > 0:
            avg_loss = sum(self.training_step_losses) / len(self.training_step_losses)
            self.log("train_loss", avg_loss, prog_bar=True)
            self.training_step_losses.clear()  # Reset after logging
        return loss

    def configure_optimizers(self):
        if self.optimizer_state_path is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
            optimizer.load_state_dict(
                torch.load(self.optimizer_state_path)["optimizer_states"][0]
            )
            print("Restored optimizer state")
        return optimizer
