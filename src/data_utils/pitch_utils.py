def get_fourier_coef_frequency(
    k: int,
    sampling_rate: int | float,
    n_fft: int
) -> float:
    r"""
    Get the physical frequency of the k-th Fourier Coefficients. Mathematically,
    this is computed as:
    .. math::
        F(k) = \frac{k\cdot\text{sampling_rate}}{\text{n_fft}}

    :param int k: The k-th Fourier coefficient to get the frequency of.
    :param int or float sampling_rate: The sampling rate of the signal.
    :param int n_fft: The number of samples used for computing the FFT.

    :return float: The frequency corresponding to the k-th coefficient.
    """
    return k * sampling_rate / n_fft

def get_midi_note_frequency(
    p: int,
    ref_note: int = 69,
    ref_frequency: int | float = 440,
) -> float:
    r"""
    Get the physical frequency of the p-th MIDI note. Mathematically,
    the frequency for the note :math:`p\in\{0, 1,\ldots, 127\}` is computed
    as:
    .. math::
        F(p) = \text{ref_note}\cdot2^{\frac{p-\text{ref_frequency}}12}

    :param int p: The MIDI note number to compute the frequency.
    :param int ref_note: The reference note number of the reference
        frequency. The default is 69 for A4=440Hz.
    :param int or float ref_frequency: The reference frequency for
        the reference MIDI note. The default is 440Hz, corresponding to A4.
    """
    return ref_frequency * 2 ** ((p - ref_note) / 12)