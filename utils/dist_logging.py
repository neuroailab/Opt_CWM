import logging
import os
import sys


def get_logger(name):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if not logging.getLogger().hasHandlers():
        if rank == 0:
            logging.basicConfig(
                stream=sys.stdout,
                level=logging.INFO,
            )
        else:
            logging.basicConfig(level=logging.CRITICAL)

    return logging.getLogger(name)
