import os
import yaml


_ALLOWED_VALUES = {
    'feature_type': ['chroma', 'midi'],
    'window_type': ['hann'],
    'optimizer': ['adam', 'adamw', 'sgd']
}

def parse_yml_config(filepath: str):
    with open(filepath) as file:
        configs = file.read()
    configs = yaml.safe_load(configs)

    validate_data_args(configs['data_args'])
    validate_training_args(configs['training_args'])
    validate_inout_args(configs['inout'])

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

    if data_args['feature_type'] not in _ALLOWED_VALUES['feature_type']:
        raise ValueError(f'feature_type should be one of: {_ALLOWED_VALUES["feature_type"]}')

    if not isinstance(data_args['n_fft'], int):
        raise TypeError('n_fft should be a positive integer!')
    elif data_args['n_fft'] <= 0:
        raise ValueError(
            'Found invalid value for n_fft! '
            'Please specify as a positive integer.'
        )

    if data_args['window_type'] not in _ALLOWED_VALUES['window_type']:
        raise ValueError(f'window_type should be one of : {_ALLOWED_VALUES["window_type"]}')
    
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

    if training_args['optimizer'] not in _ALLOWED_VALUES['optimizer']:
        raise ValueError(f'optimizer should be one of: {_ALLOWED_VALUES["optimizer"]}')

def validate_inout_args(inout: dict):
    if not os.path.exists(inout['ckpt_dir']):
        os.makedirs(inout['ckpt_dir'])

    if inout['checkpoint'] and not os.path.exists(inout['checkpoint']):
        raise FileNotFoundError(
            'checkpoint does not exist! '
            'Please check if the given path is correct'
        )