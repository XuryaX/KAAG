import logging
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, file_path: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the given name and level.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).
        file_path (Optional[str]): Path to save log file (default: None, logs to console only).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if file_path:
        file_handler = logging.FileHandler(file_path)