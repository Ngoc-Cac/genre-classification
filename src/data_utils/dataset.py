import os
import torch

from scipy.io import wavfile
from torch.utils.data import Dataset

from .processing_utils import clip_signal

from typing import Callable


class GTZAN(Dataset):
    def __init__(
        self,
        root: str,
        n_seconds: float = -1,
        *,
        preprocessor: Callable | None = None
    ):
        super().__init__()

        self.root = root
        self._n_secs = n_seconds
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

        self._data_cache = [None] * len(self._files)
        self._label_cache = [None] * len(self._files)

    def _build_cache(self, index):
        file, genre = self._files[index]
        file = f"{self.root}/{genre}/{file}"

        sr, wave = wavfile.read(file)
        wave = clip_signal(wave, sr, 0, self._n_secs)

        self._data_cache[index] = self._preprocessor(torch.tensor(wave), sr)
        self._label_cache[index] = self._genre_to_id[genre]

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            for idx in range(*index):
                if self._data_cache[idx] is None:
                    self._build_cache(idx)
        elif isinstance(index, int) and self._data_cache[index] is None:
                self._build_cache(index)

        return self._data_cache[index], self._label_cache[index]