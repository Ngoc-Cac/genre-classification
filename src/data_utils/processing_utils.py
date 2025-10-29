def clip_signal(
    signal,
    sampling_rate: int,
    start: int | float,
    duration: int | float
):
    start_idx = int(start * sampling_rate)
    end_idx = (start + duration) * sampling_rate
    end_idx = int(end_idx) + bool(end_idx % 1)  # quirky round up
    if end_idx >= len(signal):
        end_idx = len(signal) - 1

    return (
        signal[start_idx:(end_idx + 1)]
        if (start_idx < end_idx and start_idx < len(signal))
        else signal[:0]  # return nothing
    )