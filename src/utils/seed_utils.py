"""
Seed Control and Reproducibility Utilities for FinGAT Vietnam Project

This module provides comprehensive reproducibility control for all random operations
across different libraries (PyTorch, NumPy, Python random, CUDA, etc.)
"""

import os
import random
import numpy as np
import torch
from typing import Optional
import logging

# Get logger
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = True):
    """
    Set seed for all random number generators to ensure reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (may reduce performance)
        warn_only: If True, only warn about deterministic algorithms instead of error
    """
    logger.info(f"Setting random seed to {seed}")

    # Python random
    random.seed(seed)
    logger.debug("Python random seed set")

    # NumPy
    np.random.seed(seed)
    logger.debug("NumPy random seed set")

    # PyTorch
    torch.manual_seed(seed)
    logger.debug("PyTorch CPU random seed set")

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        logger.debug("PyTorch CUDA random seed set")

        # CUDA deterministic operations
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("CUDA deterministic mode enabled")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.debug("CUDA benchmark mode enabled (non-deterministic)")

    # PyTorch deterministic algorithms
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            logger.debug("PyTorch deterministic algorithms enabled")
        except Exception as e:
            logger.warning(f"Could not enable deterministic algorithms: {e}")

    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations

    logger.info(f"Random seed {seed} set for all generators")


def get_random_state():
    """
    Get current random state from all generators.

    Returns:
        Dict containing current random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: dict):
    """
    Set random state for all generators.

    Args:
        state: Dict containing random states from get_random_state()
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])

    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])

    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])

    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])

        if 'torch_cuda_random_all' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])

    logger.debug("Random state restored from checkpoint")


class SeedManager:
    """
    Context manager for temporary seed changes.

    Usage:
        with SeedManager(12345):
            # Operations with seed 12345
            pass
        # Original seed restored
    """

    def __init__(self, seed: int, deterministic: bool = None):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
        self.original_deterministic = None

    def __enter__(self):
        # Save current state
        self.original_state = get_random_state()

        # Save current deterministic settings
        if torch.cuda.is_available():
            self.original_deterministic = {
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark
            }

        # Set new seed
        set_seed(self.seed, deterministic=self.deterministic or False)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        set_random_state(self.original_state)

        # Restore deterministic settings
        if self.original_deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.original_deterministic['cudnn_deterministic']
            torch.backends.cudnn.benchmark = self.original_deterministic['cudnn_benchmark']


def configure_torch_deterministic(enabled: bool = True, warn_only: bool = True):
    """
    Configure PyTorch deterministic behavior.

    Args:
        enabled: Whether to enable deterministic algorithms
        warn_only: If True, only warn instead of error for unsupported operations
    """
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set environment variable for deterministic CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        logger.info("PyTorch deterministic mode enabled")
    else:
        torch.use_deterministic_algorithms(False)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        logger.info("PyTorch deterministic mode disabled")


def verify_deterministic_setup():
    """
    Verify that deterministic setup is working correctly.

    Returns:
        Dict with verification results
    """
    results = {}

    # Test NumPy reproducibility
    np.random.seed(42)
    np_test1 = np.random.random(5)
    np.random.seed(42)
    np_test2 = np.random.random(5)
    results['numpy_deterministic'] = np.allclose(np_test1, np_test2)

    # Test PyTorch CPU reproducibility
    torch.manual_seed(42)
    torch_cpu_test1 = torch.rand(5)
    torch.manual_seed(42)
    torch_cpu_test2 = torch.rand(5)
    results['torch_cpu_deterministic'] = torch.allclose(torch_cpu_test1, torch_cpu_test2)

    # Test PyTorch CUDA reproducibility (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch_cuda_test1 = torch.rand(5, device='cuda')
        torch.cuda.manual_seed(42)
        torch_cuda_test2 = torch.rand(5, device='cuda')
        results['torch_cuda_deterministic'] = torch.allclose(torch_cuda_test1, torch_cuda_test2)

        # Check CUDA settings
        results['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        results['cudnn_benchmark'] = not torch.backends.cudnn.benchmark

    # Check environment variables
    results['pythonhashseed_set'] = 'PYTHONHASHSEED' in os.environ
    results['cublas_workspace_set'] = 'CUBLAS_WORKSPACE_CONFIG' in os.environ

    return results


def print_deterministic_status():
    """Print current deterministic configuration status."""
    print("=== Deterministic Configuration Status ===")

    # PyTorch settings
    print(f"PyTorch version: {torch.__version__}")

    try:
        deterministic_algos = torch.are_deterministic_algorithms_enabled()
        print(f"Deterministic algorithms: {deterministic_algos}")
    except AttributeError:
        print("Deterministic algorithms: Not available (old PyTorch version)")

    # CUDA settings
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("CUDA available: False")

    # Environment variables
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}")

    # Verification
    verification = verify_deterministic_setup()
    print("\n=== Verification Results ===")
    for key, value in verification.items():
        status = "âœ“" if value else "âœ—"
        print(f"{status} {key}: {value}")


# Initialize with project default seed on import
DEFAULT_SEED = 42

def init_reproducibility(seed: int = DEFAULT_SEED, deterministic: bool = True):
    """
    Initialize reproducibility settings for the entire project.

    This function should be called at the start of any training script.

    Args:
        seed: Random seed to use
        deterministic: Whether to enable deterministic operations
    """
    logger.info("Initializing reproducibility settings")
    set_seed(seed, deterministic=deterministic)

    # Print status for verification
    print_deterministic_status()

    logger.info("Reproducibility initialization completed")


# Example usage and testing
if __name__ == "__main__":
    # Test the reproducibility system
    print("Testing reproducibility system...")

    # Initialize with default settings
    init_reproducibility(seed=42, deterministic=True)

    # Test with context manager
    print("\nTesting SeedManager context manager...")
    with SeedManager(12345):
        test_tensor = torch.rand(3)
        print(f"Random tensor with seed 12345: {test_tensor}")

    # Should be different after context exit
    test_tensor2 = torch.rand(3)
    print(f"Random tensor after context: {test_tensor2}")

    print("\nReproducibility system test completed!")