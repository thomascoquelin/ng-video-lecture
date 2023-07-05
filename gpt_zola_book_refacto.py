from __future__ import annotations

import signal
import sys
from typing import Dict, List, Optional

import torch

from book_gpt import GPTLanguageModel, HyperParams
from codec import Codec
from data_store import BookStore


def get_default_device():
    if torch.cuda.is_available():
        default_device = 'cuda'
    elif torch.backends.mps.is_available() and sys.gettrace() is None:
        default_device = 'mps'
    else:
        default_device = 'cpu'
    return default_device


class Gen:
    def __init__(
            self,
            model_state_path=None,
            hyper_params: Optional[HyperParams] = None,
            book_store: BookStore = None,
            device: Optional[str] = None,
    ):
        self.eval_interval = 500
        self.device = device if device else get_default_device()
        self.ds: Optional[BookStore] = book_store
        self.batch_count = 0
        self.loss = 0
        self.loss_history = {0: 0.}
        self.hp: HyperParams = hyper_params or HyperParams()

        checkpoint: Optional[Dict] = torch.load(model_state_path, map_location=device) if model_state_path else None
        if checkpoint:
            if "codec" in checkpoint.keys():
                codec = Codec().from_dict(codec_dict=checkpoint['codec'])
                self.ds = BookStore(codec=codec)
            if "hyper_params" in checkpoint.keys():
                self.hp = checkpoint["hyper_params"]
            self.batch_count = checkpoint['batch_count']
            self.loss = checkpoint['loss']
            self.loss_history = checkpoint['loss_history']
        self.model = GPTLanguageModel(
            block_size=self.hp.block_size,
            dropout=self.hp.dropout,
            n_embd=self.hp.n_embd,
            n_embd_b=self.hp.n_embd_b,
            n_layer=self.hp.n_layer,
            n_head=self.hp.n_head,
            device=self.device,
            vocab_size=self.ds.codec.vocab_size,
            num_books=self.ds.num_books,
        )
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.m = self.model.m_to_device()
        # print the number of parameters in the model
        print(sum(p.numel() for p in self.m.parameters()) / 1e6, 'M parameters')
        print("device:", device)

        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.learning_rate)
        if checkpoint:
            self.model.eval()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.interrupted = False

        def sigint_handler(*_):
            if not self.interrupted:
                self.interrupted = True
            else:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, sigint_handler)

    @torch.no_grad()
    def estimate_loss(self, batch_count: Optional[int] = None):
        if not batch_count:
            batch_count = self.hp.batch_size
        losses = torch.zeros(batch_count)
        for k in range(batch_count):
            X, B, Y = self.ds.get_batch(
                batch_size=self.hp.batch_size,
                block_size=self.hp.block_size,
                to_device=self.device,
            )
            logits, loss = self.model(X, B, Y)
            losses[k] = loss.item()
        self.loss = losses.mean()
        return self.loss

    def train(self, count: int):
        self.model.train()
        for iteration in range(self.batch_count + 1, self.batch_count + count + 1):
            # every once in a while evaluate the loss on train and val sets
            if iteration % self.eval_interval == 0:
                self.model.eval()
                print(f"step {iteration}: ", end="")
                losses = self.estimate_loss()
                print(f"train loss {losses:.4f}")
                self.loss_history[iteration] = float(losses)
                self.model.train()

            # sample a batch of data
            xb, bb, yb = self.ds.get_batch(
                batch_size=self.hp.batch_size,
                block_size=self.hp.block_size,
                to_device=self.device,
            )

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
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        print(self.ds.codec.decode(self.m.generate(context, max_new_tokens=length, book_id=book_id)[0].tolist()))
        # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    @torch.no_grad()
    def stream(self, input_text: str, book_name: str):
        input_int = self.ds.codec.encode(input_text)
        book_id = self.ds.codec.btoi[book_name]
        context = torch.tensor([[0] + input_int], dtype=torch.long, device=self.device)
        generator = self.m.generate_stream(context, book_id=book_id)
        while True:
            yield self.ds.codec.itos[int(next(generator)[0][0])]

    @torch.no_grad()
    def get_next_word_list(self, input_text: str, book_name: str, count: int) -> List[str]:
        input_int = self.ds.codec.encode(input_text)
        book_id = self.ds.codec.btoi[book_name]
        completions = [Completion(codec=self.ds.codec)]
        it = 0
        while any(completions) and it < 20:
            active_comps = [c for c in completions if c]
            nb_ac = len(active_comps)
            done_comps = [c for c in completions if not c]
            py_context = [input_int + ac.string_int for ac in active_comps]
            context = torch.tensor(py_context, dtype=torch.long, device=self.device)
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
        print(" ".join([f"'{str(c)}'({int(c.prob * 100)}%)" for c in completions]))
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
                'hyper_params': self.hp,
                'codec': self.ds.codec.to_dict(),
            },
            path,
        )


class Completion:
    def __init__(self, codec: Codec):
        self.string = ""
        self.prob = 1.
        self.string_int: List[int] = []
        self.open = True
        self.codec = codec

    def add_char(self, char_int: int, prob: float) -> Completion:
        ret = self.__new__(self.__class__)
        ret.__init__(codec=self.codec)
        char_string = self.codec.decode([char_int])
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
