from loguru import logger
from retinaradar.paths import PATHS

def initialize_log():

    # disable automatic logging to stderr
    logger.remove()

    # set up log
    logger.add(
        PATHS["loguru"],
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    
