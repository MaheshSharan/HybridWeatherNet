"""
Training module for weather forecast bias correction.

This package contains implementations for training and evaluating
deep learning models for weather forecast bias correction.
"""

from .train import main as train_model

__all__ = ['train_model'] 