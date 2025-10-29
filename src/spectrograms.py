import torch

from itertools import chain

from torchaudio.transforms import Spectrogram

from data_utils.pitch_utils import (
    get_fourier_coef_frequency,
    get_midi_note_frequency
)

from typing import Callable


class LogFreqSpectrogram:
    r"""
    Class to compute pitch-based log-frequency spectrograms.

    ## What is a pitch-based log-frequency spectrogram?
    A pitch-based log-frequency spectrogram is a modified spectrogram
    that has been binned in the frequency axis to corresponds to our MIDI
    note system. You can think of it as segmenting the original spectrogram
    into notes on the piano.

    In this implementation, the MIDI note `p` includes all `k`-th Fourier coefficients
    satisfying the inequality :math:`F(p - .5)\leq F_{\text{coef}}(k)\leq F(p + .5)`,
    where :math:`F` and :math:`F_{\text{coef}}` denotes the physical frequency of
    a MIDI note and the `k`-th Fourier coefficients respectively.

    To see how to compute these frequencies, please see the `data_utils.pitch_utils`
    module.

    ## Caveats
    In order for this computation to work, all input signals must have the same
    sampling rate,

    :param int n_fft: The number of samples to compute the FFT. This is used
        to compute the STFT for the spectrograms.
    :param int sampling_rate: The sampling rate of all signals to be given
        to this class.
    """
    _BANDWIDTHS = tuple(
        (
            get_midi_note_frequency(p - .5),
            get_midi_note_frequency(p + .5),
        )
        for p in range(128)
    )
    __slots__ = (
        '_pitch_class',
        '_sampling_rate',
        '_spec',
    )

    def __init__(self,
        sampling_rate,
        n_fft: int = 2048,
        win_length: int | None= None,
        hop_length: int | None= None,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window
    ):
        """"""
        self._spec = Spectrogram(
            n_fft,
            win_length,
            hop_length,
            window_fn=window_fn,
            power=2
        )

        self.sampling_rate = sampling_rate

    @property
    def n_fft(self) -> int:
        """The number of samples used to compute the FFT for STFT."""
        return self._spec.n_fft

    @property
    def sampling_rate(self) -> int:
        """The sampling rate of all signals to be processed."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, new_sr: int):
        self._sampling_rate = new_sr
        self._compute_pitch_class()


    def _compute_pitch_class(self):
        get_coef = lambda k: get_fourier_coef_frequency(k, self._sampling_rate, self.n_fft)
        self._pitch_class = [
            list(filter(
                lambda k: lower <= get_coef(k) <= upper,
                range(self.n_fft // 2 + 1)
            ))
            for lower, upper in self._BANDWIDTHS
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the pitch-based log-frequency spectrogram. This will bin the
        STFT results along the frequency axis and sum the squared magnitudes
        of the Fourier coefficients in each bin.

        The result will be a new axis with 128 channels, corresponding to the
        MIDI note system.

        :param torch.Tensor waveform: The signal waveform. The shape of
            this tensor should be `(..., samples)`, where `...` is any
            variable number of dimensions.

        :return torch.Tensor: A tensor with shape `(..., 128, frames)`.
            The last dimension corresponds to each time frame, determined by
            the specified `win_length` and `hop_length`. The second last
            dimension corresponds to each MIDI note, with the 69th note
            being A4=440Hz.
        """
        spectrogram = self._spec(waveform)

        if len(spectrogram.shape) == 2:
            binned_specs = [spectrogram[ks].sum(axis=0) for ks in self._pitch_class]
            dim = 0
        else:
            binned_specs = [spectrogram[:, ks].sum(axis=1) for ks in self._pitch_class]
            dim = 1

        return torch.stack(binned_specs, dim=dim)
        

class Chromagram(LogFreqSpectrogram):
    """
    Class to compute Chromagrams from signals. This class extends the
    `LogFreqSpectrogram` class and works in a similar way.

    Unlike its predecessor, this further bins the frequency axis into notes of
    the 12-TET system, starting at C being 0.
    """
    def _compute_pitch_class(self):
        super()._compute_pitch_class()

        self._pitch_class = [
            list(chain(
                *(self._pitch_class[p] for p in range(128) if p % 12 == chroma)
            ))
            for chroma in range(12)
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the chromagram. This will bin the STFT results along the frequency axis
        and sum the squared magnitudes of the Fourier coefficients in each bin.

        The result will be a new axis with 12 channels, representing the
        12-TET system, with the 0th index corresponding to the chroma C.

        :param torch.Tensor waveform: The signal waveform. The shape of
            this tensor should be `(..., samples)`, where `...` is any
            variable number of dimensions.

        :return torch.Tensor: A tensor with shape `(..., 12, frames)`.
            The last dimension corresponds to each time frame, determined by
            the specified `win_length` and `hop_length`. The second last
            dimension corresponds to each chroma, with the 0th index
            being C.
        """
        return super().__call__(waveform)
