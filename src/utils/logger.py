"""Logging utilities for ML pipeline."""

import logging
import os
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Logger for ML pipeline execution."""
    
    def __init__(self, name: str = 'ml_pipeline', log_dir: str = 'logs'):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with file and console handlers."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with UTF-8 encoding for unicode/emoji support
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f'pipeline_{timestamp}.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler with error handling for non-UTF8 terminals
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # Set error handler to replace problematic characters instead of raising errors
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


def get_logger(name: str = 'ml_pipeline', log_dir: str = 'logs') -> PipelineLogger:
    """Get or create a pipeline logger."""
    return PipelineLogger(name, log_dir)
