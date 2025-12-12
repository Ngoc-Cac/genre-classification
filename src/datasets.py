import os
import itertools

import pandas as pd
import librosa
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Dataset, Subset

from data_utils.processing_utils import crop_signal

from typing import Callable, Iterable


class _MGRDataset(Dataset):
    def __init__(
        self,
        audio_files: Iterable[str],
        genres: Iterable[str],
        n_seconds: float = -1,
        random_crops: int = 0,
        *,
        sampling_rate: int | None = None,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ):
        super().__init__()

        self._n_secs = n_seconds
        self._rand_crops = int(random_crops)
        self._sr = sampling_rate

        genres = list(genres)
        self._genre_to_id = {genre: i for i, genre in enumerate(set(genres))}
        self._audios = tuple(zip(audio_files, genres, strict=True))
        self._size = len(self._audios) * (self._rand_crops if self._rand_crops else 1)

        self._preprocessor = (lambda wf, _: wf) if preprocessor is None else preprocessor
        self._data_cache = [None] * self._size
        self._label_cache = [None] * self._size

    @property
    def id_to_genre(self):
        return {i: genre for genre, i in self._genre_to_id.values()}

    @property
    def num_genres(self):
        return len(self._genre_to_id)

    def random_split(self, ratios: list[float]) -> tuple[Subset, Subset]:
        # random_split doesnt actually check if dataset is a Dataset,
        # it just needs an object with len method
        splits = [split.indices for split in random_split(self._audios, ratios)]

        if self._rand_crops > 1:
            splits = [
                [
                    idx * self._rand_crops + i
                    for idx in split for i in range(self._rand_crops)
                ]
                for split in splits
            ]
        return [Subset(self, indices) for indices in splits]

    def _build_cache(self, index: int):
        file_index = index // self._rand_crops if self._rand_crops else index
        file, genre = self._audios[file_index]
        wave, sr = librosa.load(file, mono=True, sr=self._sr)

        if self._rand_crops and self._n_secs > 0:
            index = file_index * self._rand_crops
            for i in range(index, index + self._rand_crops):
                self._data_cache[i] = self._preprocessor(
                    torch.tensor(crop_signal(wave, sr, self._n_secs)), sr
                )
                self._label_cache[i] = self._genre_to_id[genre]
        else:
            if self._n_secs > 0:
                wave = crop_signal(wave, sr, self._n_secs, 0)
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


class GTZAN(_MGRDataset):
    def __init__(
        self,
        root: str,
        n_seconds: float = -1,
        random_crops: int = 0,
        *,
        sampling_rate: int | None = None,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ):
        file_iter, genre_iter = itertools.tee(
            (f"{root}/{genre}/{file}", genre)
            for genre in os.listdir(root) for file in os.listdir(f"{root}/{genre}")
        )
        super().__init__(
            [file for file, _ in file_iter],
            [genre for _, genre in genre_iter],
            n_seconds, random_crops,
            sampling_rate=sampling_rate,
            preprocessor=preprocessor
        )


class FMA(_MGRDataset):
    def __init__(self,
        meta_root: str,
        audio_root: str,
        n_seconds: int = -1,
        random_crops: int = 0,
        subset: str = 'small',
        *,
        subset_ratio: float | None = None,
        sampling_rate: int | None = None,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ):
        no_load_ids = pd.read_csv(f"{os.path.dirname(__file__)}/fma_no_load.csv")
        metadata = pd.read_csv(f"{meta_root}/tracks.csv", index_col=0, header=[0, 1])
        metadata = metadata[metadata['set', 'subset'] == subset]
        metadata = metadata[~metadata.index.isin(no_load_ids['id'])]

        test_data = metadata['set', 'split'] == 'test'
        train_data, test_data = metadata[~test_data], metadata[test_data]

        if subset_ratio:
            subset_idx, _ = train_test_split(
                train_data.index.to_numpy(),
                train_size=subset_ratio,
                stratify=train_data['track', 'genre_top'].to_numpy()
            )
            train_data = train_data[train_data.index.isin(subset_idx)]

        track_ids = (f"{track_id:0>6}" for track_id in train_data.index)
        super().__init__(
            (f"{audio_root}/{id[:3]}/{id}.mp3" for id in track_ids),
            train_data['track', 'genre_top'].tolist(),
            n_seconds, random_crops,
            sampling_rate=sampling_rate,
            preprocessor=preprocessor
        )

        track_ids = (f"{track_id:0>6}" for track_id in test_data.index)
        self._test_set = _MGRDataset(
            (f"{audio_root}/{id[:3]}/{id}.mp3" for id in track_ids),
            test_data['track', 'genre_top'].tolist(),
            n_seconds, 0,
            sampling_rate=sampling_rate,
            preprocessor=preprocessor
        )

        train_data.reset_index(inplace=True)
        splits = {
            split: train_data[train_data['set', 'split'] == split].index.tolist()
            for split in ['training', 'validation']
        }
        num_crops = self._rand_crops
        self._splits = {
            split: Subset(
                self,
                [idx * num_crops + i for idx in indices for i in range(num_crops)]
                if num_crops > 1 else indices
            )
            for split, indices in splits.items()
        }

    def random_split(self, ratios: list[float] | None = None):
        if ratios is None:
            return tuple(self._splits.values())
        return super().random_split(ratios)
