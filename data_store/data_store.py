import glob
from typing import Optional, Tuple

import torch

from codec import Codec


class DataStore:
    def __init__(self, codec: Optional[Codec] = None):
        self.codec = codec

    def get_batch(
            self,
            batch_size: int,
            block_size: int,
            to_device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
