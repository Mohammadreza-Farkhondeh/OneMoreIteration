import logging
import time


class Logger:
    def __init__(self):
        logging.basicConfig(
            filename="logs/log_{}.log".format(time.strftime("%Y%m%d-%H%M%S")),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def get_logger():
        return logging.getLogger()
