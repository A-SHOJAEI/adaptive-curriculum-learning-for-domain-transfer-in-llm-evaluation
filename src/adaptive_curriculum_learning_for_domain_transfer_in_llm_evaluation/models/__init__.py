"""Model architecture for adaptive curriculum learning."""

from .components import (
    AttentionPooling,
    DifficultyPredictor,
    DomainClassifier,
    GradientReversalLayer,
    LowRankAdapter,
    gradient_reversal,
)
from .model import AdaptiveCurriculumModel, CurriculumScheduler

__all__ = [
    "AdaptiveCurriculumModel",
    "CurriculumScheduler",
    "LowRankAdapter",
    "DomainClassifier",
    "DifficultyPredictor",
    "AttentionPooling",
    "GradientReversalLayer",
    "gradient_reversal",
]