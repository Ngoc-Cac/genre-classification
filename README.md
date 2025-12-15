# Music Genre Classification

This is an in-progress repository for music genre classification task.

The aim of this project is to investigate different methods of
feature engineering for audio data, as well as different models to perform
classification on said feature representation.

### Table of Contents
- [**The Current State**](#the-current-state)
- [**About The Datasets**](#about-the-datasets)
- [**The Training Script**](#the-training-script)
    - [**Dependencies**](#dependencies)
    - [**Training Configurations**](#training-configurations)
    - [**Run Training**](#run-training)
    - [**See The Training Progress**](#see-the-training-progress)

# The Current State
As of now, this project is a work-in-progress and, thus, is not very well documented.
So please excuse me for that.\
The project has currently implemented four different spectrogram features:
- MIDI pitch-based spectrogram: This is a binned-version of the power spectrogram
    where frequencies are binned to their closest MIDI note.
- Chroma-based spectrogram: This extends the previous format by further binning each
    MIDI into the respective tones of the 12-TET system.
- Mel-scale spectrogram: Similar to the MIDI spectrogram, this spectrogram also bins
    the frequency axis according to the Mel scale. For more information on the Mel scale,
    see [here](https://en.wikipedia.org/wiki/Mel_scale).
- Mel-frequency cepstrum coefficients: The MFCC extends the idea of Mel-scale spectrograms
    by performing an additional transform on a logarithmic Mel-scale spectrograms, creating a
    sepctrum-of-a-spectrum (hence the name 'cepstrum'). For more information, see
    [here](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).

# About The Datasets
This project currently uses two datasets:
- The [GTZAN dataset](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals).
    You can unzip the dataset by running the following command:
```bash
unzip assets/gtzan.zip -d assets
```
- The [FMA dataset](https://arxiv.org/abs/1612.01840).
    To download this dataset, please navigate to this
    [repository](https://github.com/mdeff/fma) and follow their instructions. Note
    that the `fma_metadata.zip` MUST be downloaded as well, apart from the audio files.

# The Training Script
There is a training script available at [`train.py`](./train.py), which encapsulates
the data loading, feature extraction, model building, training and validation.

## Dependencies
To use the train script, install the necessary requirements:
1. Python 3.13+
2. `pip install -r requirements.txt`
3. Appropriate installation of PyTorch with CUDA, see [here](https://pytorch.org/get-started/locally/).

## Training Configurations
Then, specify your training configuration in [`train_config.yml`](./train_config.yml).

There are a variety of parameters you can configure. Some of which are required, some of which
are not. The details of what each parameter means/does is described at the very bottom of the file.

Note that it is not required for you to use the exact same filename. In the case where you
have multiple configurations you want to test out, you can specify them in different files
with different filenames.

### The Dataset Paths
For `gtzan`, you only need to provide the root directory that contains the audios.
However, for `fma`, you must specify the root as two paths: the path to the metadata
directory and the path to the audio directory.

You may also use your own dataset, in which case, just specify the root path in `train_config.yml`.
**You must ensure that all audio files in your dataset can be read by** `librosa.load`.

Note that the implemented spectrogram transforms needs to know the sampling
rate beforehand. Thus, any deviation in the given sampling rate will not give
a correct result due to how frequency-binning works.

Furthermore, any deviation in the sampling rate will also cause the spectrograms to
differ in the temporal dimension due to how STFT works, which will cause errors when
collating samples into batches when training. 

To deal with this, you need to specify a sampling rate to which the loaded
waveforms are normalized to in `train_config.yml`.

## Run Training
Finally, run training with:
```bash
python train.py
```
This will set everything up according to the configurations given in `train_config.yml`.

You may also specify a different YAML configuration file with:
```bash
python train.py -cf path_to_file
```

## See The Training Progress
All metrics and hyperparameters are logged to [Tensorboard](https://www.tensorflow.org/tensorboard).

During training (or even after training), you can view the reported metrics by running:
```bash
tensorboard --logdir path/to/your/ckpt_dir
```
and then visiting `http://localhost:6006`. This URL will also be logged to your stdout when running
tensorboard, so you can also click there to get redirected instead.
