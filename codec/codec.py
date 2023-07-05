from typing import List, Any, Dict


class Codec:
    def to_dict(self) -> Dict[str,Any]:
        raise NotImplementedError("Codec interface can't be exported")

    def from_dict(self, codec_dict: Dict[str,Any]) -> None:
        raise NotImplementedError("Loading not implemented for interface")

    def encode(self, s: Any, **kwargs) -> List[int]:
        raise NotImplementedError(f"decode not implemented for type {type(s)}")

    def decode(self, l:List[int], **kwargs) -> Any:
        raise NotImplementedError
