import random

from numpy.typing import ArrayLike


def crop_signal(
    signal: ArrayLike,
    sampling_rate: int,
    duration: int | float,
    start: int | float | None = None,
) -> ArrayLike:
    r"""
    Crop the signal starting at the `start`-th second and ending at the
    `(start + duration)`-second. If `start` is not given, this will randomize
    a starting position.

    Internally, the cropping begins at the sample that contains the sub-sample
    start time `start`. Likewise, the croppings spans the samples that contains
    the sub-sample duration `duration`.
    .. math:
        \text{start_idx}=\lfloor s\cdot F\rfloor
    .. math:
        \text{n_samples}=\lceil d\cdot F\rceil

    where :math:`s` is the start time `start`, :math:`d` is the duration `duration`
    and :math:`F` is the sampling rate `sampling_rate`. The cropped signal includes
    all samples starting at start_idx and ending at
    :math:`\text{start_idx} + \text{n_samples} - 1`.

    :param ArrayLike signal: The signal to crop.
    :param int sampling_rate: The sampling rate of the given signal.
    :param int or float duration: The duration (seconds) of the cropped signal.
    :param int, float or None: Where (in seconds) to start the cropping. If `start=None`,
    a random starting index will be chosen.
    """
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