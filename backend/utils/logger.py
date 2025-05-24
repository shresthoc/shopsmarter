"""
Logger utility for ShopSmarter backend
Provides consistent logging across all backend services
"""

import logging
import json
from datetime import datetime
import os
import sys
from typing import Any, Dict, Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels"""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "[%(asctime)s] [%(levelname)s] %(message)s" + reset,
        logging.INFO: blue + "[%(asctime)s] [%(levelname)s] %(message)s" + reset,
        logging.WARNING: yellow + "[%(asctime)s] [%(levelname)s] %(message)s" + reset,
        logging.ERROR: red + "[%(asctime)s] [%(levelname)s] %(message)s" + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] [%(levelname)s] %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class Logger:
    """Custom logger for ShopSmarter backend"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        
        # Create file handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"),
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _format_message(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Format log message with optional data"""
        if data:
            return f"{message}\nData: {json.dumps(data, indent=2)}"
        return message
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(self._format_message(message, data))
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(self._format_message(message, data))
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(self._format_message(message, data))
    
    def error(self, message: str, error: Optional[Exception] = None, data: Optional[Dict[str, Any]] = None):
        """Log error message with optional exception details"""
        error_data = data or {}
        if error:
            error_data.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_traceback': getattr(error, '__traceback__', None)
            })
        self.logger.error(self._format_message(message, error_data))
    
    def critical(self, message: str, error: Optional[Exception] = None, data: Optional[Dict[str, Any]] = None):
        """Log critical error message"""
        error_data = data or {}
        if error:
            error_data.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_traceback': getattr(error, '__traceback__', None)
            })
        self.logger.critical(self._format_message(message, error_data))

def get_logger(name: str) -> Logger:
    """Get a logger instance for the given name"""
    return Logger(name) 