import sys

from loguru import logger

from ..utils.utils import is_main_process


def set_log_level(
    min_log_level: str = "INFO",
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
):
    def log_level_filter(record):
        return (
            record["level"].no >= logger.level(min_log_level).no and is_main_process()
        )

    logger.remove()
    logger.add(sys.stderr, filter=log_level_filter, format=log_format)
