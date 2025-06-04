"""
Logging configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


class TimingLogger:
    """Context manager for timing operations"""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"{self.operation} failed after {duration:.2f}s: {exc_val}")
        else:
            self.logger.info(f"{self.operation} completed in {duration:.2f}s")
