import os

import torchaudio

from torch.utils.data import Dataset

from typing import Callable


class GTZAN(Dataset):
    def __init__(
        self,
        root: str,
        preprocessor: Callable | None = None
    ):
        super().__init__()

        self.root = root
        self._genre_to_id = {
            genre: i
            for i, genre in enumerate(os.listdir(self.root))
        }
        self._files = [
            (file, genre)
            for genre in os.listdir(self.root)
            for file in os.listdir(f"{self.root}/{genre}")
        ]

        if preprocessor is None:
            preprocessor = lambda wf, sr: wf
        self._preprocessor = preprocessor

        self._cache = [None] * len(self._files)

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if self._cache is None:
            file, genre = self._files[index]
            file = f"{self.root}/{genre}/{file}"
            self._cache = (
                self._preprocessor(torchaudio.load(file)),
                self._genre_to_id[genre]
            )
        return self._cache[index]