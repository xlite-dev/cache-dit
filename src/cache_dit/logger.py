import logging
import os
import sys
import torch.distributed as dist

_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LOG_LEVEL = os.environ.get("CACHE_DIT_LOG_LEVEL", "info")
_LOG_LEVEL = getattr(logging, _LOG_LEVEL.upper(), 0)
_LOG_DIR = os.environ.get("CACHE_DIT_LOG_DIR", None)


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("CACHE_DIT")
_default_handler = None
_default_file_handler = None
_inference_log_file_handler = {}


def _setup_logger():
    _root_logger.setLevel(_LOG_LEVEL)
    global _default_handler
    global _default_file_handler
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)

    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(_LOG_LEVEL)
        _root_logger.addHandler(_default_handler)

    if _default_file_handler is None and _LOG_DIR is not None:
        if not os.path.exists(_LOG_DIR):
            try:
                os.makedirs(_LOG_DIR)
            except OSError as e:
                _root_logger.warn(f"Error creating directory {_LOG_DIR} : {e}")
        _default_file_handler = logging.FileHandler(_LOG_DIR + "/default.log")
        _default_file_handler.setLevel(_LOG_LEVEL)
        _default_file_handler.setFormatter(fmt)
        _root_logger.addHandler(_default_file_handler)

    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    pid = os.getpid()
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.addHandler(_default_handler)
    if _LOG_DIR is not None and pid is None:
        logger.addHandler(_default_file_handler)
    elif _LOG_DIR is not None:
        if _inference_log_file_handler.get(pid, None) is not None:
            logger.addHandler(_inference_log_file_handler[pid])
        else:
            if not os.path.exists(_LOG_DIR):
                try:
                    os.makedirs(_LOG_DIR)
                except OSError as e:
                    _root_logger.warn(
                        f"Error creating directory {_LOG_DIR} : {e}"
                    )
            _inference_log_file_handler[pid] = logging.FileHandler(
                _LOG_DIR + f"/process.{pid}.log"
            )
            _inference_log_file_handler[pid].setLevel(_LOG_LEVEL)
            _inference_log_file_handler[pid].setFormatter(
                NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
            )
            _root_logger.addHandler(_inference_log_file_handler[pid])
            logger.addHandler(_inference_log_file_handler[pid])
    logger.propagate = False
    return logger


def logging_rank_0(
    logger: logging.Logger, message: str, level: int = logging.INFO
):
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be an instance of logging.Logger")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(level, int):
        raise TypeError("level must be an integer representing a logging level")

    def _logging_msg():
        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        else:
            logger.info(message)

    if dist.is_initialized():
        if dist.get_rank() == 0:
            _logging_msg()
    else:
        _logging_msg()
