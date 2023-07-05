from typing import List, Dict, Iterable, Any

from codec import Codec
from codec.char_codec import CharCodec


class BookCharCodec(CharCodec):
    def __init__(self, chars: List[str]):
        super().__init__(chars)
        self.btoi: Dict[str, int] = {}
        self.b_itol: Dict[int, int] = {}
        self.num_books = 0

    def to_dict(self) -> Dict[str,Any]:
        ret = super().to_dict()
        ret["btoi"] = self.btoi
        ret["b_itol"] = self.b_itol
        ret["num_books"] = self.num_books
        return ret

    def from_dict(self, codec_dict: Dict[str,Any]) -> Codec:
        super().from_dict(codec_dict=codec_dict)
        self.btoi = codec_dict["btoi"]
        self.b_itol = codec_dict["b_itol"]
        self.num_books = codec_dict["num_books"]
        return self

    def set_books(self, books: List[str]):
        self.btoi = {b: i for i, b in enumerate(books)}
        self.num_books = len(books)

    def encode_book(self, book: str) -> int:
        return self.btoi[book]

