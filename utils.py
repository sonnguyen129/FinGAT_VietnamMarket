"""Utility functions for FinGAT experiments."""

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def Timer(name: str = ""):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed:.2f}s")


def save_results(results: dict, path: str):
    """Save experiment results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def init_weights(model: torch.nn.Module):
    """Xavier uniform initialization for all parameters with dim > 1."""
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
