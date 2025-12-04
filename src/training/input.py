import os, yaml

from .structs import (
    DATASETS,
    FEATURE_TYPES,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    WINDOW_FUNCTIONS,
)


def parse_yml_config(filepath: str):
    with open(filepath) as file:
        configs = yaml.safe_load(file)

    # convert to list for easier argument passing later on
    if not isinstance(configs['data_args']['root'], list):
        configs['data_args']['root'] = [configs['data_args']['root']]

    validate_data_args(configs['data_args'])
    validate_feat_args(configs['feature_args'])
    validate_training_args(configs['training_args'])
    validate_inout_args(configs['inout'])

    return configs


def validate_data_args(data_args: dict):
    if data_args['type'] not in DATASETS:
        raise ValueError(f"Dataset type should be one of: {list(DATASETS.keys())}")
    elif data_args['type'] == 'fma' and len(data_args['root']) != 2:
        raise ValueError(
            "Invalid 'root' argument for FMA dataset!"
            "Specify as [metadata_root, audio_root]."
        )
    elif not all(os.path.exists(p) for p in data_args['root']):
        raise FileNotFoundError(
            "Cannot find the given root path! "
            "Please check if the given path is correct."
        )

    if not isinstance(data_args['train_ratio'], float):
        raise TypeError('train_ratio should be a float between 0 and 11')
    elif not 0 <= data_args['train_ratio'] <= 1:
        raise ValueError(
            'Found invalid value for train_ratio! '
            'Please specify as a number between 0 and 1.'
        )

    if not isinstance(data_args['first_n_secs'], (int, float)):
        raise TypeError('first_n_secs should be a positive number!')
    elif data_args['first_n_secs'] <= 0:
        data_args['first_n_secs'] = -1

    if not isinstance(data_args['random_crops'], int):
        raise TypeError('random_crops must be a non-negative integer!')
    elif data_args['random_crops'] < 0:
        raise ValueError(
            'Found invalid value for random_crops!'
            'Please specify as a non-negative integer.'
        )

def validate_feat_args(feat_args):
    if feat_args['feature_type'] not in FEATURE_TYPES:
        raise ValueError(f'feature_type should be one of: {list(FEATURE_TYPES.keys())}')

    if feat_args['feature_type'] in ('mel', 'mfcc'):
        if not isinstance(feat_args['n_mels'], int):
            raise TypeError('n_mels should be a positive integer!')
        elif feat_args['n_mels'] <= 0:
            raise ValueError('n_mels should be a positive integer!')
    if feat_args['feature_type'] == 'mfcc':
        if not isinstance(feat_args['n_mfcc'], int):
            raise TypeError('n_mfcc should be a positive integer!')
        elif feat_args['n_mfcc'] <= 0:
            raise ValueError('n_mfcc should be a positive integer!')

    if not isinstance(feat_args['n_fft'], int):
        raise TypeError('n_fft should be a positive integer!')
    elif feat_args['n_fft'] <= 0:
        raise ValueError(
            'Found invalid value for n_fft! '
            'Please specify as a positive integer.'
        )

    if feat_args['window_type'] not in WINDOW_FUNCTIONS:
        raise ValueError(f'window_type should be one of : {list(WINDOW_FUNCTIONS.keys())}')

def validate_training_args(training_args: dict):
    if not isinstance(training_args['epochs'], int):
        raise TypeError('epochs should be a positive integer!')
    elif training_args['epochs'] <= 0:
        raise ValueError(
            'Found invalid value for epochs! '
            'Please specify as a positive integer.'
        )

    if not isinstance(training_args['batch_size'], int):
        raise TypeError('batch_size should be a positive integer!')
    elif training_args['batch_size'] <= 0:
        raise ValueError(
            'Found invalid value for batch_size! '
            'Please specify as a positive integer.'
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

    opt_args = training_args['optimizer']
    if opt_args['use_8bit_optimizer']:
        if OPTIMIZERS_8BIT is None:
            raise ModuleNotFoundError(
                'Cannot find bitsandbytes! Make sure bitsandbytes is '
                'installed in order to use 8bit-optimization.'
            )
        avail_opts = OPTIMIZERS_8BIT
    else:
        avail_opts = OPTIMIZERS
    if opt_args['type'] not in avail_opts:
        raise ValueError(f'optimizer should be one of: {list(avail_opts.keys())}')

    if opt_args['kwargs'].get('lr') is None:
        raise KeyError(
            'Missing required argument "lr" for optimizer!'
        )
    elif not isinstance(opt_args['kwargs']['lr'], (int, float)):
        raise TypeError('learning_rate should be a positive number!')
    elif opt_args['kwargs']['lr'] <= 0:
        raise ValueError(
            'Found invalid value for learning_rate! '
            'Please specify as a positive number.'
        )

def validate_inout_args(inout: dict):
    if not os.path.exists(inout['ckpt_dir']):
        os.makedirs(inout['ckpt_dir'])

    if not os.path.exists(inout['model_path']):
        raise FileNotFoundError(
            "Model configuration file does not exist!"
            "Please check if the given path is correct."
        )

    if inout['checkpoint'] == 'latest':
        ckpts = list(filter(
            lambda path: path[-4:] == '.pth',
            os.listdir(inout['ckpt_dir'])
        ))
        inout['checkpoint'] =  f"{inout['ckpt_dir']}/{ckpts[-1]}" if ckpts else ''
    elif (
        inout['checkpoint'] and
        not os.path.exists(inout['checkpoint'])
    ):
        raise FileNotFoundError(
            'Checkpoint does not exist! '
            'Please check if the given path is correct'
        )
