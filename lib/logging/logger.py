import datetime
import logging
import os
import sys
import colorlog
from typing import TextIO, Dict


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """Function setup as many loggers as you want"""

    numeric_level: int = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        # Default to INFO if level name is invalid
        logging.warning(f"Invalid log level '{level}'. Defaulting to INFO.")
        numeric_level = logging.INFO
        
    # Generate a timestamp
    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the directory for log files
    log_dir: str = "./logs"
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    # Use the timestamp to create a unique log file for each session
    log_file: str = f"{log_dir}/log_{timestamp}.log"
    # Create a handler for writing to the log file
    file_handler: logging.FileHandler = logging.FileHandler(log_file, encoding='utf-8')    
    file_formatter: logging.Formatter = logging.Formatter('%(asctime)s|(%(filename)s)[%(levelname)s]:%(message)s')
    file_handler.setFormatter(file_formatter)

    # Create a handler for writing to the console
    # Ensure that sys.stdout is treated as TextIO for the purpose of type checking `open`
    stdout_buffer: TextIO = sys.stdout # type: ignore 
    console_handler: logging.StreamHandler = logging.StreamHandler()
    # The fileno() approach can be problematic on some systems or if stdout is redirected in certain ways.
    # A more robust way is to directly use sys.stdout if it's suitable, 
    # or handle potential errors if a specific encoding/buffering is strictly needed via open(sys.stdout.fileno()).
    # For simplicity and broad compatibility, directly using console_handler.setStream(sys.stdout) is often safer
    # unless the specific utf-8 encoding and buffering=1 are critical and sys.stdout default behavior is insufficient.
    # Assuming the original intent for utf-8 and buffering=1 is critical:
    try:
        # Attempt to open sys.stdout with specific encoding and buffering
        # This is more of a TextIOWrapper around the OS-level file descriptor
        console_stream: TextIO = open(stdout_buffer.fileno(), mode='w', encoding='utf-8', buffering=1)
        console_handler.setStream(console_stream)
    except Exception as e_console_stream: # Catching generic Exception as various issues can arise
        # Fallback to default sys.stdout if the fileno approach fails
        logging.warning(f"Could not open sys.stdout with specific encoding/buffering: {e_console_stream}. Falling back to default sys.stdout.")
        console_handler.setStream(sys.stdout)
        
    # Define a color scheme for the log levels
    log_colors_config: Dict[str, str] = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    console_formatter: colorlog.ColoredFormatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s|[%(levelname)s]%(reset)s:%(message)s',
        datefmt='%H:%M:%S',
        log_colors=log_colors_config)
    console_handler.setFormatter(console_formatter)

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger