import random

from numpy.typing import ArrayLike


def clip_signal(
    signal: ArrayLike,
    sampling_rate: int,
    duration: int | float,
    start: int | float | None = None,
) -> ArrayLike:
    if not len(signal):
        return signal[:0]

    num_samples = duration * sampling_rate
    # quirky round up because n % 1 > 0 iff n has a decimal part
    num_samples = min(int(num_samples) + bool(num_samples % 1), len(signal))

    start_idx = (
        random.randint(0, len(signal) - num_samples)
        if start is None else int(start * sampling_rate)
    )
    end_idx = min(start_idx + num_samples - 1, len(signal) - 1)

    return (
        signal[start_idx:(end_idx + 1)]
        if (start_idx < end_idx and start_idx < len(signal))
        else signal[:0]  # return nothing
    )