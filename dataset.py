from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class TokenDataset:
    def __init__(
        self,
        data_dir: Path,
        block_size: int,
        device: str,
        token_dtype: str = "uint16",
    ):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.device = device
        self.token_dtype = np.dtype(token_dtype)

        self.train_data = np.memmap(
            self.data_dir / "train.bin",
            dtype=self.token_dtype,
            mode="r",
        )

        self.val_data = np.memmap(
            self.data_dir / "val.bin",
            dtype=self.token_dtype,
            mode="r",
        )

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError("split must be either 'train' or 'val'")

        max_start = len(data) - self.block_size - 1

        if max_start <= 0:
            raise ValueError(
                f"{split}.bin is too small for block_size={self.block_size}"
            )

        indices = torch.randint(
            low=0,
            high=max_start,
            size=(batch_size,),
        )

        x = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        data[i : i + self.block_size],
                        dtype=np.int64,
                    )
                )
                for i in indices
            ]
        )

        y = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        data[i + 1 : i + 1 + self.block_size],
                        dtype=np.int64,
                    )
                )
                for i in indices
            ]
        )

        x = x.to(self.device)
        y = y.to(self.device)

        return x, y