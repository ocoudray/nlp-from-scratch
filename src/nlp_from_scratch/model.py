import math

import torch
from lightning import LightningModule
from torch import nn

from nlp_from_scratch.constants import MAX_LEN, VOCAB_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(MAX_LEN, d_model)
        position = torch.arange(0, MAX_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(VOCAB_SIZE) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.0):
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


class SimpleTransformer(LightningModule):
    def __init__(self, d_model=128, num_heads=4, dim_ff=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dim_ff) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)  # Output layer for classification
        self.dropout = nn.Dropout(0.0)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.training_step_losses = []
        self.optimizer_state_path = None

    def get_embeddings(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x

    def _classify_output(self, x):
        return self.fc_out(x)

    def forward(self, x, mask=None):
        x = self.get_embeddings(x, mask=mask)
        return self._classify_output(x)  # Final output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        inputs, mask, attention_mask, labels = batch
        output = self.forward(inputs, mask=attention_mask)
        loss_matrix = self.loss_fn(output.view(-1, VOCAB_SIZE), labels.view(-1)).view(
            labels.size()
        )
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
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
            optimizer.load_state_dict(
                torch.load(self.optimizer_state_path)["optimizer_states"][0]
            )
            print("Restored optimizer state")
        return optimizer
