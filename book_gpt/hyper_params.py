from __future__ import annotations


class HyperParams:
    def __init__(
            self,
            batch_size=32,  # how many independent sequences will we process in parallel?
            block_size=512,  # what is the maximum context length for predictions?
            learning_rate=3e-4,
            eval_iters=250,
            n_embd=416,
            n_embd_b=32,
            n_head=6,
            n_layer=6,
            dropout=0.2,

    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_embd_b = n_embd_b
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
