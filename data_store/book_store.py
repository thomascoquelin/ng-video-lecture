import glob
from typing import Dict, Tuple, Optional, Any

import torch

from codec import Codec, BookCharCodec
from data_store.data_store import DataStore


class BookStore(DataStore):
    def __init__(self, codec: Optional[Dict[str, Any]] = None):
        if codec:
            loaded_codec = BookCharCodec()
        super().__init__(codec=codec)
        self.data: Dict[int, torch.Tensor] = {}
        self.num_books = 0

    def load_books(self, book_glob: str):
        text_by_book = {}
        for filename in glob.glob(book_glob):
            with open(filename, 'r', encoding='utf-8') as f:
                text_by_book[filename] = f.read()

        if not isinstance(self.codec, BookCharCodec):
            chars = sorted(list(set().union(*[set(x) for x in text_by_book.values()])))
            self.codec = BookCharCodec(chars)
            self.codec.set_books(list(text_by_book.keys()))

        for book in text_by_book.keys():
            self.data[self.codec.encode_book(book)] = torch.tensor(
                self.codec.encode(text_by_book[book]),
                dtype=torch.long,
            )
        self.num_books = len(text_by_book)

    def get_batch(
            self,
            batch_size: int,
            block_size: int,
            to_device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate a small batch of data of inputs x and targets y
        # data = train_data if split == 'train' else val_data
        ibx = torch.randint(len(self.data), (batch_size,))

        idx = [(i, torch.randint(self.b_itol[int(i)] - block_size, (1,))) for i in ibx]
        x = torch.stack([self.data[int(b)][i:i + block_size] for b, i in idx])
        y = torch.stack([self.data[int(b)][i + 1:i + block_size + 1] for b, i in idx])
        x, b, y = x.to(to_device), ibx.to(to_device), y.to(to_device)
        return x, b, y