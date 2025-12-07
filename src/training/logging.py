import os, sys, logging

from logging.handlers import RotatingFileHandler

# root logger has logging.WARNING by default so we need to
# set to notset first in order to config any new loggers
logging.root.setLevel(logging.NOTSET)


def setup_logger(
    name: str,
    log_file: str,
    to_stdout: bool = True
) -> logging.Logger:
    FORMATTER = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s: %(message)s'
    )
    logger = logging.getLogger(name)

    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(FORMATTER)
        logger.addHandler(stream_handler)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    handler = RotatingFileHandler(
        log_file,
        maxBytes=2000000,
        backupCount=3,
        encoding='utf-8'
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(FORMATTER)
    logger.addHandler(handler)
    return logger
