"""Custom model components for adaptive curriculum learning.

This module contains specialized neural network components used in the
adaptive curriculum learning framework, including low-rank adapters,
domain classifiers, and difficulty predictors.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAdapter(nn.Module):
    """Low-rank adaptation layer for efficient domain-specific fine-tuning.

    Implements LoRA-style adapters that add learnable low-rank matrices
    to existing weight matrices, enabling parameter-efficient domain adaptation.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for adapter output.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank decomposition: W = B @ A
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize adapter parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-rank adapter to input.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Adapted tensor of shape (..., out_features).
        """
        # Apply dropout to input
        x = self.dropout(x)

        # Low-rank transformation: x @ A @ B
        adapted = x @ self.lora_A @ self.lora_B

        # Apply scaling
        return adapted * self.scaling


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation.

    A multi-layer neural network that predicts the source domain of input
    representations. Used in adversarial training to learn domain-invariant
    features.

    Args:
        input_dim: Dimension of input features.
        num_domains: Number of domains to classify.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize classifier parameters."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify domain from input features.

        Args:
            features: Input features of shape (batch_size, input_dim).

        Returns:
            Domain logits of shape (batch_size, num_domains).
        """
        return self.classifier(features)


class DifficultyPredictor(nn.Module):
    """Neural network for predicting question difficulty.

    Predicts a difficulty score in [0, 1] from input representations,
    used for curriculum scheduling.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize predictor parameters."""
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict difficulty from input features.

        Args:
            features: Input features of shape (batch_size, input_dim).

        Returns:
            Difficulty scores of shape (batch_size, 1) in range [0, 1].
        """
        return self.predictor(features)


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence representations.

    Computes weighted average of sequence embeddings using learned attention,
    providing better sequence-level representations than mean pooling.

    Args:
        input_dim: Dimension of input features.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize attention parameters."""
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention pooling to sequence.

        Args:
            sequence: Input sequence of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Tuple of:
                - Pooled representation of shape (batch_size, input_dim).
                - Attention weights of shape (batch_size, seq_len).
        """
        # Compute attention scores
        scores = self.attention(sequence).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)

        # Apply attention to sequence
        pooled = torch.bmm(
            weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            sequence,  # (batch_size, seq_len, input_dim)
        ).squeeze(1)  # (batch_size, input_dim)

        return pooled, weights


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training.

    Implements the gradient reversal operation used in domain-adversarial
    neural networks. During forward pass, acts as identity. During backward
    pass, reverses and scales gradients.

    This enables learning domain-invariant features by maximizing domain
    classification loss (which appears as minimization in the main network).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass (identity).

        Args:
            ctx: Context for backward pass.
            x: Input tensor.
            alpha: Gradient reversal scaling factor.

        Returns:
            Input tensor unchanged.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass (reverse and scale gradient).

        Args:
            ctx: Context from forward pass.
            grad_output: Gradient from downstream layers.

        Returns:
            Tuple of (reversed gradient, None for alpha).
        """
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to input tensor.

    Args:
        x: Input tensor.
        alpha: Gradient reversal scaling factor.

    Returns:
        Tensor with gradient reversal applied.
    """
    return GradientReversalLayer.apply(x, alpha)
