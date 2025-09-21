"""
Professional logging system for FinGAT Vietnam Market project.

This module provides centralized logging configuration with multiple handlers,
formatters, and support for both file and console logging.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml


class FinGATLogger:
    """Professional logger class with multiple handlers and formatters."""

    def __init__(self, name: str = "FinGAT", config_path: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            name: Logger name
            config_path: Path to logging configuration file
        """
        self.name = name
        self.config_path = config_path
        self._logger = None

    def setup_logger(
        self,
        log_level: str = "INFO",
        log_dir: str = "logs",
        console_logging: bool = True,
        file_logging: bool = True,
        experiment_name: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up logger with specified configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            console_logging: Enable console logging
            file_logging: Enable file logging
            experiment_name: Name for experiment-specific log file

        Returns:
            Configured logger instance
        """
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        simple_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )

        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)

        # File handlers
        if file_logging:
            # Main log file
            main_log_file = log_path / "fingat_main.log"
            file_handler = logging.FileHandler(main_log_file, mode="a")
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

            # Error log file
            error_log_file = log_path / "fingat_errors.log"
            error_handler = logging.FileHandler(error_log_file, mode="a")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)

            # Experiment-specific log file
            if experiment_name:
                exp_log_file = log_path / f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                exp_handler = logging.FileHandler(exp_log_file, mode="w")
                exp_handler.setLevel(logging.INFO)
                exp_handler.setFormatter(detailed_formatter)
                logger.addHandler(exp_handler)

        self._logger = logger
        return logger

    @classmethod
    def from_config(cls, config_file: str) -> logging.Logger:
        """
        Create logger from configuration file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Configured logger instance
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logging_config = config.get('logging', {})

        instance = cls()
        return instance.setup_logger(
            log_level=logging_config.get('level', 'INFO'),
            console_logging=logging_config.get('console_logging', True),
            file_logging=logging_config.get('file_logging', True)
        )

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        if self._logger is None:
            self._logger = self.setup_logger()
        return self._logger


class MLFlowLoggingHandler(logging.Handler):
    """Custom logging handler for MLFlow experiment tracking."""

    def __init__(self, experiment_name: str):
        super().__init__()
        self.experiment_name = experiment_name

    def emit(self, record):
        """Emit log record to MLFlow."""
        try:
            import mlflow
            log_entry = self.format(record)
            mlflow.log_text(log_entry, f"logs/{record.levelname.lower()}.txt")
        except Exception:
            pass  # Silently fail if MLFlow is not available


class WandBLoggingHandler(logging.Handler):
    """Custom logging handler for Weights & Biases experiment tracking."""

    def __init__(self):
        super().__init__()

    def emit(self, record):
        """Emit log record to WandB."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"log_message": self.format(record)})
        except Exception:
            pass  # Silently fail if WandB is not available


def setup_experiment_logging(
    experiment_name: str,
    log_level: str = "INFO",
    use_mlflow: bool = False,
    use_wandb: bool = False
) -> logging.Logger:
    """
    Set up logging for a specific experiment.

    Args:
        experiment_name: Name of the experiment
        log_level: Logging level
        use_mlflow: Enable MLFlow logging
        use_wandb: Enable WandB logging

    Returns:
        Configured logger for the experiment
    """
    logger_instance = FinGATLogger(name=f"FinGAT.{experiment_name}")
    logger = logger_instance.setup_logger(
        log_level=log_level,
        experiment_name=experiment_name
    )

    # Add experiment tracking handlers
    if use_mlflow:
        mlflow_handler = MLFlowLoggingHandler(experiment_name)
        mlflow_handler.setLevel(logging.INFO)
        logger.addHandler(mlflow_handler)

    if use_wandb:
        wandb_handler = WandBLoggingHandler()
        wandb_handler.setLevel(logging.INFO)
        logger.addHandler(wandb_handler)

    return logger


# Convenience functions
def get_logger(name: str = "FinGAT") -> logging.Logger:
    """Get a logger instance with default configuration."""
    return FinGATLogger(name).get_logger()


def log_system_info(logger: logging.Logger):
    """Log system information for reproducibility."""
    import platform
    import torch

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")

    logger.info("=" * 50)


def log_config(logger: logging.Logger, config: Dict):
    """Log configuration parameters."""
    logger.info("=== Configuration ===")

    def _log_dict(d: Dict, prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                _log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")

    _log_dict(config)
    logger.info("=" * 50)


if __name__ == "__main__":
    # Example usage
    logger = get_logger()
    logger.info("FinGAT logging system initialized successfully!")
    log_system_info(logger)