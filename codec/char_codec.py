from typing import List, Any, Dict

from codec.codec import Codec


class CharCodec(Codec):
    def __init__(self, chars: List[str]):
        self.vocab_size = len(chars)
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def to_dict(self) -> Dict[str,Any]:
        return {
            "vocab_size": self.vocab_size,
            "chars": self.chars,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    def from_dict(self, codec_dict: Dict[str,Any]) -> Codec:
        self.vocab_size = codec_dict["vocab_size"]
        self.chars = codec_dict["chars"]
        self.itos = codec_dict["stoi"]
        self.stoi = codec_dict["stoi"]
        return self

    def encode(self, s: Any, **kwargs) -> List[int]:
        if not isinstance(s, str):
            return super().encode(s, **kwargs)
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int], **kwargs) -> str:
        return ''.join([self.itos[i] for i in l])

