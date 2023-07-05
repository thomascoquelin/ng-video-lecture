from __future__ import annotations
import signal
import glob
import sys
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 512  # what is the maximum context length for predictions?
max_iters = 5000
default_eval_interval = 500
learning_rate = 3e-4
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() and sys.gettrace() is None:
    device = 'mps'
else:
    device = 'cpu'
eval_iters = 250
n_embd = 416
n_embd_b = 32
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# torch.manual_seed(1337)

text_by_book = {}
for filename in glob.glob("zola/v1/*.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        text_by_book[filename] = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set().union(*[set(x) for x in text_by_book.values()])))
vocab_size = len(chars)
num_books = len(text_by_book)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
# create a mapping for books
btoi = {b: i for i, b in enumerate(text_by_book.keys())}

# Train and test splits
data: Dict[int, torch.Tensor] = {}
for book in text_by_book.keys():
    data[btoi[book]] = torch.tensor(encode(text_by_book[book]), dtype=torch.long)

b_itol = {i: len(d) for i, d in data.items()}


# data loading
def get_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    ibx = torch.randint(len(data), (batch_size,))

    idx = [(i, torch.randint(b_itol[int(i)] - block_size, (1,))) for i in ibx]
    x = torch.stack([data[int(b)][i:i + block_size] for b, i in idx])
    y = torch.stack([data[int(b)][i + 1:i + block_size + 1] for b, i in idx])
    x, b, y = x.to(device), ibx.to(device), y.to(device)
    return x, b, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd_in, n_embd_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_in, 4 * n_embd_out),
            nn.ReLU(),
            nn.Linear(4 * n_embd_out, n_embd_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd_in, n_head_in, n_embd_b_in):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd_in // n_head_in
        self.sa = MultiHeadAttention(n_head_in, head_size)
        self.ffwd = FeedFoward(n_embd_in=n_embd_in + n_embd_b_in, n_embd_out=n_embd_in)
        self.ln1 = nn.LayerNorm(n_embd_in)
        self.ln2 = nn.LayerNorm(n_embd_in)

    def forward(self, input_tuple):
        x, b = input_tuple
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(torch.cat([self.ln2(x), b], dim=2))
        return [x, b]


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.book_embedding_table = nn.Embedding(num_books, n_embd_b)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head_in=n_head, n_embd_b_in=n_embd_b) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, ibx: Tensor, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # ibx is (B) tensor of integers

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        book_emb = self.book_embedding_table(ibx)  # (B,C)
        book_emb_t = torch.reshape(book_emb, (B, 1, n_embd_b)).expand(-1, T, -1)
        x = tok_emb + pos_emb  # (B,T,C)
        x, _ = self.blocks([x, book_emb_t])  # (B,T,C), (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, book_id: int, max_new_tokens):
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
        idx_cond = idx[:, -block_size:]
        ibx_cond = torch.full(size=(idx_cond.shape[0],), fill_value=book_id, device=device)
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


class Gen:
    def __init__(self, model_state_path=None):
        self.batch_count = 0
        self.loss = 0
        self.loss_history = {0: 0.}
        checkpoint = torch.load(model_state_path, map_location=device) if model_state_path else None
        self.model = GPTLanguageModel()
        if checkpoint:
            self.batch_count = checkpoint['batch_count']
            self.loss = checkpoint['loss']
            self.loss_history = checkpoint['loss_history']
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.m = self.model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in self.m.parameters()) / 1e6, 'M parameters')
        print("device:", device)

        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        if checkpoint:
            self.model.eval()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.interrupted = False

        def signal_handler(sig, frame):
            if not self.interrupted:
                self.interrupted = True
            else:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

    @torch.no_grad()
    def estimate_loss(self, batch_count=eval_iters):
        losses = torch.zeros(batch_count)
        for k in range(batch_count):
            X, B, Y = get_batch()
            logits, loss = self.model(X, B, Y)
            losses[k] = loss.item()
        self.loss = losses.mean()
        return self.loss

    def train(self, count=max_iters, eval_interval=default_eval_interval):
        self.model.train()
        for iteration in range(self.batch_count + 1, self.batch_count + count + 1):
            # every once in a while evaluate the loss on train and val sets
            if iteration % eval_interval == 0:
                self.model.eval()
                print(f"step {iteration}: ", end="")
                losses = self.estimate_loss()
                print(f"train loss {losses:.4f}")
                self.loss_history[iteration] = float(losses)
                self.model.train()

            # sample a batch of data
            xb, bb, yb = get_batch()

            # evaluate the loss
            logits, loss = self.model(xb, bb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.batch_count = iteration
            if self.interrupted:
                self.interrupted = False
                break
        self.model.eval()
        print(f"step {self.batch_count}: ", end="")
        if count != 1:
            losses = self.estimate_loss()
            print(f"train loss {losses:.4f}")
        else:
            print("no loss eval")

    @torch.no_grad()
    # generate from the model
    def generate(self, book_id=0, length=500):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(self.m.generate(context, max_new_tokens=length, book_id=book_id)[0].tolist()))
        # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    @torch.no_grad()
    def stream(self, input_text: str, book_name: str):
        input_int = encode(input_text)
        book_id = btoi[book_name]
        context = torch.tensor([[0] + input_int], dtype=torch.long, device=device)
        generator = self.m.generate_stream(context, book_id=book_id)
        while True:
            yield itos[int(next(generator)[0][0])]

    @torch.no_grad()
    def get_next_word_list(self, input_text: str, book_name: str, count: int) -> List[str]:
        input_int = encode(input_text)
        book_id = btoi[book_name]
        completions = [Completion()]
        it = 0
        while any(completions) and it < 20:
            active_comps = [c for c in completions if c]
            nb_ac = len(active_comps)
            done_comps = [c for c in completions if not c]
            py_context = [input_int + ac.string_int for ac in active_comps]
            context = torch.tensor(py_context, dtype=torch.long, device=device)
            top_probs = self.get_top_n_probs(book_id, context, count)
            new_comps: List[Completion] = []
            for n in range(nb_ac):
                for p in range(count):
                    c_i = int(top_probs.indices[n][p])
                    new_comp = active_comps[n].add_char(char_int=c_i, prob=float(top_probs.values[n][p]))
                    if new_comp in new_comps + done_comps:
                        # print(f"duplicate comp: {str(new_comp)}")
                        continue
                    if new_comp.prob < 0.001:
                        continue
                    new_comps.append(
                        new_comp
                    )
                    # print(
                    #     f"it: {it}, n: {n}, p: {p}, "
                    #     f"pred: {itos[c_i]}: {float(top_probs.values[n][p])}"
                    #     f" => {str(new_comps[-1])}: {new_comps[-1].prob}"
                    # )
            completions = sorted(done_comps + new_comps, reverse=True)[0:count]
            it += 1

        print(f"{input_text}| => {book_name}")
        print(" ".join([f"'{str(c)}'({int(c.prob*100)}%)" for c in completions]))
        print()
        return [str(x) for x in completions]

    def get_top_n_probs(self, book_id, context, count):
        probs: torch = self.m.get_next_prob(idx=context, book_id=book_id)
        top_probs = torch.topk(probs, count, sorted=True)
        return top_probs

    def save(self, path: str):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                'loss_history': self.loss_history,
                'batch_count': self.batch_count,
            },
            path,
        )


class Completion:
    def __init__(self):
        self.string = ""
        self.prob = 1.
        self.string_int: List[int] = []
        self.open = True

    def add_char(self, char_int: int, prob: float) -> Completion:
        ret = self.__new__(self.__class__)
        ret.__init__()
        char_string = itos[char_int]
        if char_string.isalpha() and len(ret.string) < 20:
            ret.string = self.string + char_string
            ret.prob = self.prob * prob
            ret.string_int = self.string_int + [char_int]
            ret.open = True
        else:
            ret.string = self.string or char_string
            ret.prob = self.prob * prob
            ret.string_int = self.string_int
            ret.open = False
        return ret

    def __gt__(self, other: Completion) -> bool:
        return self.prob > other.prob

    def __lt__(self, other: Completion) -> bool:
        return self.prob < other.prob

    def __bool__(self) -> bool:
        return self.open

    def __str__(self) -> str:
        return self.string

    def __eq__(self, other: Completion) -> bool:
        return self.string == other.string


def rprint(nc, g):
    for _ in range(nc):
        print(next(g), end="", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    pass
