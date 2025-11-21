import os
import yaml

from .structs import (
    FEATURE_TYPES,
    MODELS,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    WINDOW_FUNCTIONS,
)


def parse_yml_config(filepath: str):
    with open(filepath) as file:
        configs = file.read()
    configs = yaml.safe_load(configs)

    validate_data_args(configs['data_args'])
    validate_training_args(configs['training_args'])
    validate_inout_args(configs['inout'])
    validate_model_architecure(configs['model'])

    configs['model'] = {
        "backbone": configs['model']['backbone'],
        "kwargs": {
            key: value
            for key, value in configs['model'].items()
            if key != 'backbone'
        }
    }

    return configs


def validate_data_args(data_args: dict):
    if not os.path.exists(data_args['root']):
        raise FileNotFoundError(
            'Data root folder does not exist! '
            'Please check if the given path is correct.'
        )

    if not isinstance(data_args['first_n_secs'], (int, float)):
        raise TypeError('first_n_secs should be a positive number!')
    elif data_args['first_n_secs'] <= 0:
        data_args['first_n_secs'] = -1

    if data_args['feature_type'] not in FEATURE_TYPES:
        raise ValueError(f'feature_type should be one of: {list(FEATURE_TYPES.keys())}')

    if data_args['feature_type'] == 'mfcc':
        if isinstance(data_args['n_mels'], int):
            raise TypeError('n_mels should be a positive integer!')
        elif data_args['n_mels'] <= 0:
            raise ValueError('n_mels should be a positive integer!')

    if not isinstance(data_args['n_fft'], int):
        raise TypeError('n_fft should be a positive integer!')
    elif data_args['n_fft'] <= 0:
        raise ValueError(
            'Found invalid value for n_fft! '
            'Please specify as a positive integer.'
        )

    if data_args['window_type'] not in WINDOW_FUNCTIONS:
        raise ValueError(f'window_type should be one of : {list(WINDOW_FUNCTIONS.keys())}')
    
    if not isinstance(data_args['train_ratio'], float):
        raise TypeError('train_ratio should be a float between 0 and 11')
    elif not 0 <= data_args['train_ratio'] <= 1:
        raise ValueError(
            'Found invalid value for train_ratio! '
            'Please specify as a number between 0 and 1.'
        )

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

    if not isinstance(training_args['learning_rate'], (int, float)):
        raise TypeError('learning_rate should be a positive number!')
    elif training_args['learning_rate'] <= 0:
        raise ValueError(
            'Found invalid value for learning_rate! '
            'Please specify as a positive number.'
        )

    if not isinstance(training_args['regularization_lambda'], (int, float)):
        raise TypeError('Regularization parameter should be a positive number')
    elif training_args['regularization_lambda'] < 0:
        raise ValueError(
            'Found invalid value for regularization parameter! '
            'Please specify as a positive number'
        )

    if training_args['use_8bit_optimizer']:
        if OPTIMIZERS_8BIT is None:
            raise ModuleNotFoundError(
                'Cannot find bitsandbytes! Make sure bitsandbytes is '
                'installed in order to use 8bit-optimization.'
            )
        avail_opts = OPTIMIZERS_8BIT
    else:
        avail_opts = OPTIMIZERS
    if training_args['optimizer'] not in avail_opts:
        raise ValueError(f'optimizer should be one of: {list(avail_opts.keys())}')

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

def validate_inout_args(inout: dict):
    if not os.path.exists(inout['ckpt_dir']):
        os.makedirs(inout['ckpt_dir'])

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
            'checkpoint does not exist! '
            'Please check if the given path is correct'
        )

def validate_model_architecure(model_args: dict):
    if model_args['backbone'] not in MODELS:
        raise ValueError(
            "Backbone of model must be 'cnn' or 'resnet'. "
            f"Found {model_args['backbone']}."
        )

    if not all(isinstance(i, int) for i in model_args['inner_channels']):
        raise TypeError(
            "Found incorrect type for inner_channels! "
            "Inner channels must be a list of positive integers."
        )
    elif not all(i > 0 for i in model_args['inner_channels']):
        raise ValueError(
            "Found negative number in inner_channels! "
            "Inner channels must be a list of positive integers."
        )

    if not all(isinstance(i, int) for i in model_args['downsampling_rates']):
        raise TypeError(
            "Found incorrect type for downsampling_rates. "
            "Inner channels must be a list of positive integers."
        )
    elif not all(i > 0 for i in model_args['downsampling_rates']):
        raise ValueError(
            "Found negative number in downsampling_rates. "
            "Inner channels must be a list of positive integers."
        )

    if not isinstance(model_args['num_linear_layers'], int):
        raise TypeError(
            "Found incorrect type for num_linear_layers! "
            "Please specify as a non-negative integer."
        )
    elif model_args['num_linear_layers'] < 0:
        raise ValueError(
            "Found negative number for num_linear_layers! "
            "Please specify as a non-negative integer."
        )
