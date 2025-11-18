# Music Genre Classification

This is an in-progress repository for music genre classification task.

The aim of this project is to investigate different methods of
feature engineering for audio data, as well as different models to perform
classification on said feature representation.

# The Current State
As of now, this project is a work-in-progress and, thus, is not very well documented. So please excuse me for that.\
The project has currently implemented three different spectrogram features:
- MIDI pitch-based spectrogram: This is a binned-version of the power spectrogram
    where frequencies are binned to their closest MIDI note.
- Chroma-based spectrogram: This extends the previous format by further binning
    each MIDI into the respective tones of the 12-TET system.
- Mel-frequency spectrogram: MFC are spectrograms that uses Mel-frequency cepstral coefficients (MFCC), based on a series of binning and transforms of the signal's original power spectrogram. For more information, see [here](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).

# The Training Script
There is a training script available at [`train.py`](./train.py), which encapsulates the data loading, feature extraction, model building, training and validation.

To use the train script, install the necessary requirements:
1. Python 3.13+
2. `pip install -r requirements.txt`
3. Appropriate installation of PyTorch with CUDA, see [here](https://pytorch.org/get-started/locally/).

Then, specify your training configuration in [`train_config.yml`](./train_config.yml). Finally, run training with:
```bash
python train.py
```

You may also specify a different YAML configuration file with:
```bash
python train.py -cf path_to_file
```
