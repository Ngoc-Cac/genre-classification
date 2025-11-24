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
        random_crops: int = 0,
        *,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ):
        super().__init__()

        self._n_secs = n_seconds
        self._rand_crops = int(random_crops)

        self._genre_to_id = {genre: i for i, genre in enumerate(os.listdir(root))}
        self._files = [
            (f"{root}/{genre}/{file}", genre)
            for genre in os.listdir(root) for file in os.listdir(f"{root}/{genre}")
        ]
        self._size = len(self._files) * (self._rand_crops if self._rand_crops else 1)

        self._preprocessor = (lambda wf, _: wf) if preprocessor is None else preprocessor
        self._data_cache = [None] * self._size
        self._label_cache = [None] * self._size

    @property
    def id_to_genre(self):
        return {i: genre for genre, i in self._genre_to_id.values()}

    def _build_cache(self, index: int):
        file_index = index // self._rand_crops if self._rand_crops else index
        file, genre = self._files[file_index]
        sr, wave = wavfile.read(file)

        if self._rand_crops and self._n_secs > 0:
            index = file_index * self._rand_crops
            for i in range(index, index + self._rand_crops):
                self._data_cache[i] = self._preprocessor(
                    torch.tensor(clip_signal(wave, sr, self._n_secs)), sr
                )
                self._label_cache[i] = self._genre_to_id[genre]
        else:
            if self._n_secs > 0:
                wave = clip_signal(wave, sr, self._n_secs, 0)
            self._data_cache[index] = self._preprocessor(torch.tensor(wave), sr)
            self._label_cache[index] = self._genre_to_id[genre]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        if isinstance(index, tuple):
            for idx in range(*index):
                if self._data_cache[idx] is None:
                    self._build_cache(idx)
        elif isinstance(index, int) and self._data_cache[index] is None:
                self._build_cache(index)

        return self._data_cache[index], self._label_cache[index]
