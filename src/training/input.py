import os, yaml

from .structs import (
    DATASETS,
    FEATURE_TYPES,
    WINDOW_FUNCTIONS,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    SCHEDULERS
)


def parse_yml_config(filepath: str):
    with open(filepath) as file:
        configs = yaml.safe_load(file)

    # convert to list for easier argument passing later on
    if not isinstance(configs['data_args']['root'], list):
        configs['data_args']['root'] = [configs['data_args']['root']]
    # in case no kwargs is passed
    if configs['lr_schedulers']['decay']['kwargs'] is None:
        configs['lr_schedulers']['decay']['kwargs'] = {}

    _validate_data_args(configs['data_args'])
    _validate_inout_args(configs['inout'])
    _validate_feat_args(configs['feature_args'])
    _validate_training_args(configs['training_args'])
    _validate_optimizer_args(configs['optimizer'])
    _validate_scheduler_args(configs['lr_schedulers'])

    return configs


def _validate_data_args(data_args: dict):
    if data_args['type'] not in DATASETS:
        raise ValueError(f"Dataset type should be one of: {list(DATASETS.keys())}")
    elif data_args['type'] == 'fma' and len(data_args['root']) != 2:
        raise ValueError(
            "Invalid 'root' argument for FMA dataset! Specify as [metadata_root, audio_root]."
        )
    elif not all(os.path.exists(p) for p in data_args['root']):
        raise FileNotFoundError(
            "Cannot find the given root path! Please check if the "
            "given path is correct."
        )

    if not isinstance(data_args['sampling_rate'], int):
        if data_args['sampling_rate'] is not None:
            raise TypeError("sampling_rate should be a positive integer")
    elif data_args['sampling_rate'] <= 0:
        raise ValueError(
            "Found invalid value for sampling_rate! Please specify as "
            "a positive integer."
        )

    if data_args['type'] == 'fma' and data_args['subset_ratio'] is not None:
        if not isinstance(data_args['subset_ratio'], float):
            raise TypeError("subset_ratio should be a float between 0 and 1.")
        elif not 0 < data_args['subset_ratio'] < 1:
            raise ValueError(
                "Found invalid value for subset ratio! Please specify as a "
                "number between 0 and 1."
            )

    if not isinstance(data_args['train_ratio'], float):
        if data_args['train_ratio'] is not None or data_args['type'] != 'fma':
            raise TypeError('train_ratio should be a float between 0 and 1.')
    elif not 0 <= data_args['train_ratio'] <= 1:
        raise ValueError(
            "Found invalid value for train_ratio! Please specify as a number "
            "between 0 and 1."
        )

    if not isinstance(data_args['first_n_secs'], (int, float)):
        raise TypeError("first_n_secs should be a positive number!")
    elif data_args['first_n_secs'] <= 0:
        data_args['first_n_secs'] = -1

    if not isinstance(data_args['random_crops'], int):
        raise TypeError("random_crops must be a non-negative integer!")
    elif data_args['random_crops'] < 0:
        raise ValueError(
            "Found invalid value for random_crops! Please specify as a "
            "non-negative integer."
        )

def _validate_feat_args(feat_args):
    if feat_args['feature_type'] not in FEATURE_TYPES:
        raise ValueError(f"feature_type should be one of: {list(FEATURE_TYPES.keys())}")

    if feat_args['feature_type'] in ('mel', 'mfcc'):
        if not isinstance(feat_args['n_mels'], int):
            raise TypeError("n_mels should be a positive integer!")
        elif feat_args['n_mels'] <= 0:
            raise ValueError("n_mels should be a positive integer!")
    if feat_args['feature_type'] == 'mfcc':
        if not isinstance(feat_args['n_mfcc'], int):
            raise TypeError("n_mfcc should be a positive integer!")
        elif feat_args['n_mfcc'] <= 0:
            raise ValueError("n_mfcc should be a positive integer!")

    if not isinstance(feat_args['n_fft'], int):
        raise TypeError("n_fft should be a positive integer!")
    elif feat_args['n_fft'] <= 0:
        raise ValueError(
            "Found invalid value for n_fft! Please specify as a positive integer."
        )

    if feat_args['window_type'] not in WINDOW_FUNCTIONS:
        raise ValueError(f"window_type should be one of : {list(WINDOW_FUNCTIONS.keys())}")

def _validate_inout_args(inout: dict):
    if not os.path.exists(inout['ckpt_dir']):
        os.makedirs(inout['ckpt_dir'])

    if not os.path.exists(inout['model_path']):
        raise FileNotFoundError(
            "Model configuration file does not exist! "
            "Please check if the given path is correct."
        )

    if inout['checkpoint'] == 'latest':
        ckpts = list(filter(
            lambda path: path[-4:] == '.pth', os.listdir(inout['ckpt_dir'])
        ))
        inout['checkpoint'] =  f"{inout['ckpt_dir']}/{ckpts[-1]}" if ckpts else ''
    elif inout['checkpoint'] and not os.path.exists(inout['checkpoint']):
        raise FileNotFoundError(
            "Checkpoint does not exist! Please check if the given path is correct."
        )

def _validate_training_args(training_args: dict):
    if not isinstance(training_args['epochs'], int):
        raise TypeError("epochs should be a positive integer!")
    elif training_args['epochs'] <= 0:
        raise ValueError(
            "Found invalid value for epochs! Please specify as a positive integer."
        )

    if not isinstance(training_args['batch_size'], int):
        raise TypeError("batch_size should be a positive integer!")
    elif training_args['batch_size'] <= 0:
        raise ValueError(
            "Found invalid value for batch_size! Please specify as a positive integer."
        )

    if not isinstance(training_args['distributed_training'], bool):
        raise TypeError(
            'Found invalid type for distributed_training parameter! '
            'Please specify as either true or false'
        )

    if not isinstance(training_args['mixed_precision'], bool):
        raise TypeError(
            'Found invalid type for mixed_precision parameter! '
            'Please specify as either true or false'
        )

def _validate_optimizer_args(optimizer_args: dict):
    if optimizer_args['use_8bit_optimizer']:
        if OPTIMIZERS_8BIT is None:
            raise ModuleNotFoundError(
                "Cannot find bitsandbytes! Make sure bitsandbytes is "
                "installed in order to use 8bit-optimization."
            )
        avail_opts = OPTIMIZERS_8BIT
    else:
        avail_opts = OPTIMIZERS
    if optimizer_args['type'] not in avail_opts:
        raise ValueError(f'optimizer should be one of: {list(avail_opts.keys())}')

    if optimizer_args['kwargs'].get('lr') is None:
        raise KeyError("Missing required argument 'lr' for optimizer!")
    elif not isinstance(optimizer_args['kwargs']['lr'], (int, float)):
        raise TypeError("learning_rate should be a positive number!")
    elif optimizer_args['kwargs']['lr'] <= 0:
        raise ValueError(
            "Found invalid value for learning_rate! "
            "Please specify as a positive number."
        )

def _validate_scheduler_args(scheduler_args: dict):
    if not isinstance(scheduler_args['warmup']['total_steps'], int):
        raise TypeError(
            "total_steps in warmup configuration should be a non-negative integer!"
        )
    elif scheduler_args['warmup']['total_steps'] < 0:
        raise ValueError(
            "Found negative value for total_steps in warmup configuration! "
            "Please specify as a non-negative integer."
        )
    elif scheduler_args['warmup']['total_steps'] != 0:
        # ignore this validation if total_steps == 0
        if not isinstance(scheduler_args['warmup']['start_factor'], (int, float)):
            raise TypeError(
                "start_factor in warmup configuration should "
                "be a number in the range (0, 1)!"
            )
        elif not 0 < scheduler_args['warmup']['start_factor'] < 1:
            raise ValueError(
                "Found invalid value for start_factor in warmup configuration! "
                "Please specify as a number in the range (0, 1)."
            )

    if scheduler_args['decay']['type'] not in SCHEDULERS:
        raise ValueError(
            f"Decay scheduler type should be one of: {list(SCHEDULERS.keys())}"
        )
