import logging
import os
from config import LOG_FILE

def get_logger(name="project_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        # Console
        console_handler = logging.StreamHandler()

        # File
        file_handler = logging.FileHandler(LOG_FILE)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger