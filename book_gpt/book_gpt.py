from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(
            self,
            head_size,
            n_embd: int,
            block_size: int,
            dropout: float,
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        batch_dim_size, time_dim_size, channel_dim_size = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:time_dim_size, :time_dim_size] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 block_size: int,
                 dropout: float,
                 n_embd: int,
                 ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(
                head_size=head_size,
                block_size=block_size,
                dropout=dropout,
                n_embd=n_embd,
            ) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
            self,
            n_embd_in: int,
            n_embd_out: int,
            dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_in, 4 * n_embd_out),
            nn.ReLU(),
            nn.Linear(4 * n_embd_out, n_embd_out),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
            self,
            n_embd_in,
            n_head_in,
            n_embd_b_in,
            block_size: int,
            dropout: float
    ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd_in // n_head_in
        self.sa = MultiHeadAttention(
            num_heads=n_head_in,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
            n_embd=n_embd_in
        )
        self.ffwd = FeedForward(n_embd_in=n_embd_in + n_embd_b_in, n_embd_out=n_embd_in, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd_in)
        self.ln2 = nn.LayerNorm(n_embd_in)

    def forward(self, input_tuple: Tuple[Tensor, Tensor]):
        x, b = input_tuple
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(torch.cat([self.ln2(x), b], dim=2))
        return [x, b]


class GPTLanguageModel(nn.Module):

    def __init__(
            self,
            n_embd: int,
            n_head: int,
            block_size: int,
            n_embd_b: int,
            dropout: float,
            n_layer: int,
            vocab_size: int,
            num_books: int,
            device: str,
    ):
        super().__init__()
        self.device = device
        self.n_embd_b = n_embd_b
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.book_embedding_table = nn.Embedding(num_books, n_embd_b)
        self.blocks = nn.Sequential(
            *[Block(
                n_embd_in=n_embd,
                n_head_in=n_head,
                n_embd_b_in=n_embd_b,
                block_size=block_size,
                dropout=dropout,
            ) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def m_to_device(self):
        return self.to(self.device)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, ibx: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_dim_size, time_dim_size = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # ibx is (B) tensor of integers

        tok_emb: Tensor = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb: Tensor = self.position_embedding_table(torch.arange(time_dim_size, device=self.device))  # (T,C)
        book_emb: Tensor = self.book_embedding_table(ibx)  # (B,C)
        book_emb_t: Tensor = torch.reshape(book_emb, (batch_dim_size, 1, self.n_embd_b)).expand(-1, time_dim_size, -1)
        x: Tensor = tok_emb + pos_emb  # (B,T,C)
        x, _ = self.blocks([x, book_emb_t])  # (B,T,C), (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits: Tensor = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            batch_dim_size, time_dim_size, channels_dim_size = logits.shape
            logits = logits.view(batch_dim_size * time_dim_size, channels_dim_size)
            targets = targets.view(batch_dim_size * time_dim_size)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, book_id: int, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            probs = self.get_next_prob(idx=idx, book_id=book_id)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def get_next_prob(self, idx: Tensor, book_id: int):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -self.block_size:]
        ibx_cond = torch.full(size=(idx_cond.shape[0],), fill_value=book_id, device=self.device)
        # get the predictions
        logits, loss = self(idx_cond, ibx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        return probs

    def generate_stream(self, idx: Tensor, book_id: int):
        # idx is (B, T) array of indices in the current context
        while True:
            probs = self.get_next_prob(idx=idx, book_id=book_id)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            yield idx_next
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
