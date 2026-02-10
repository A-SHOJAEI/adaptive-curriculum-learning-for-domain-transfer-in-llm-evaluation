"""
Adaptive Curriculum Learning for Domain Transfer in LLM Evaluation.

This package provides a novel curriculum learning framework that automatically
orders MMLU questions by difficulty and domain similarity to enable efficient
domain transfer learning.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .utils.config import Config
from .data.loader import MMluDataLoader
from .models.model import AdaptiveCurriculumModel
from .training.trainer import CurriculumTrainer
from .evaluation.metrics import CurriculumEvaluator

__all__ = [
    "Config",
    "MMluDataLoader",
    "AdaptiveCurriculumModel",
    "CurriculumTrainer",
    "CurriculumEvaluator",
]