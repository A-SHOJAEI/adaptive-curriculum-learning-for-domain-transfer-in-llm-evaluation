"""Model architecture for adaptive curriculum learning."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class AdaptiveCurriculumModel(nn.Module):
    """Main model with adaptive curriculum learning capabilities.

    This model combines a base language model with domain adaptation layers
    and curriculum scheduling mechanisms for effective domain transfer.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        use_adapter: bool = True,
        adapter_rank: int = 16,
        adapter_alpha: int = 32,
        dropout: float = 0.1,
        num_domains: int = 4,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize the adaptive curriculum model.

        Args:
            model_name: Name or path of pre-trained model.
            max_length: Maximum sequence length.
            use_adapter: Whether to use adapter modules.
            adapter_rank: Rank for adapter layers.
            adapter_alpha: Alpha parameter for adapter scaling.
            dropout: Dropout probability.
            num_domains: Number of domains for domain adaptation.
            config: Optional configuration dictionary to override defaults.
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.use_adapter = use_adapter
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.dropout = dropout
        self.num_domains = num_domains
        self.config_dict = config or {}

        # Extract configuration values with defaults
        self.embedding_init_std = self.config_dict.get('model', {}).get('embedding_init_std', 0.02)
        self.ewc_lambda = self.config_dict.get('model', {}).get('ewc_lambda', 0.1)
        self.domain_adversarial_weight = self.config_dict.get('model', {}).get('domain_adversarial_weight', 0.1)
        self.difficulty_loss_weight = self.config_dict.get('model', {}).get('difficulty_loss_weight', 0.1)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.base_model.config

        # Domain embedding layer
        self.domain_embedding = nn.Embedding(num_domains, self.config.hidden_size)

        # Domain-specific adaptation layers
        self.domain_adapters = nn.ModuleList([
            DomainAdapter(
                hidden_size=self.config.hidden_size,
                adapter_rank=adapter_rank,
                alpha=adapter_alpha,
                dropout=dropout
            )
            for _ in range(num_domains)
        ])

        # Curriculum difficulty predictor
        self.difficulty_predictor = DifficultyPredictor(
            hidden_size=self.config.hidden_size,
            num_layers=2,
            dropout=dropout
        )

        # Domain classifier for adversarial training
        self.domain_classifier = DomainClassifier(
            hidden_size=self.config.hidden_size,
            num_domains=num_domains,
            dropout=dropout
        )

        # Forgetting regularization
        self.forgetting_regularizer = ForgettingRegularizer(
            model_size=sum(p.numel() for p in self.base_model.parameters()),
            lambda_ewc=self.ewc_lambda
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize additional weights."""
        nn.init.normal_(self.domain_embedding.weight, std=self.embedding_init_std)

        for adapter in self.domain_adapters:
            adapter.init_weights(self.embedding_init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        difficulty_targets: Optional[torch.Tensor] = None,
        return_difficulty: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with curriculum learning.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Target labels of shape (batch_size, seq_len).
            domain_ids: Domain identifiers of shape (batch_size,).
            difficulty_targets: Target difficulty scores of shape (batch_size,).
            return_difficulty: Whether to return difficulty predictions.

        Returns:
            Dictionary containing losses and predictions.

        Raises:
            ValueError: If input tensors have invalid shapes or types.
            RuntimeError: If device mismatch occurs between tensors.
        """
        # Input validation
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor (batch_size, seq_len), got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_length:
            logger.warning(f"Input sequence length {seq_len} exceeds max_length {self.max_length}")

        # Validate attention_mask if provided
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError(f"attention_mask must be a torch.Tensor, got {type(attention_mask)}")
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")
            if attention_mask.device != device:
                raise RuntimeError(f"attention_mask device {attention_mask.device} doesn't match input_ids device {device}")

        # Validate labels if provided
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise ValueError(f"labels must be a torch.Tensor, got {type(labels)}")
            if labels.device != device:
                raise RuntimeError(f"labels device {labels.device} doesn't match input_ids device {device}")

        # Validate domain_ids if provided
        if domain_ids is not None:
            if not isinstance(domain_ids, torch.Tensor):
                raise ValueError(f"domain_ids must be a torch.Tensor, got {type(domain_ids)}")
            if domain_ids.shape[0] != batch_size:
                raise ValueError(f"domain_ids batch size {domain_ids.shape[0]} doesn't match input_ids batch size {batch_size}")
            if domain_ids.device != device:
                raise RuntimeError(f"domain_ids device {domain_ids.device} doesn't match input_ids device {device}")
            if torch.any(domain_ids >= self.num_domains) or torch.any(domain_ids < 0):
                raise ValueError(f"domain_ids contains invalid values. Must be in range [0, {self.num_domains})")

        # Validate difficulty_targets if provided
        if difficulty_targets is not None:
            if not isinstance(difficulty_targets, torch.Tensor):
                raise ValueError(f"difficulty_targets must be a torch.Tensor, got {type(difficulty_targets)}")
            if difficulty_targets.shape[0] != batch_size:
                raise ValueError(f"difficulty_targets batch size {difficulty_targets.shape[0]} doesn't match input_ids batch size {batch_size}")
            if difficulty_targets.device != device:
                raise RuntimeError(f"difficulty_targets device {difficulty_targets.device} doesn't match input_ids device {device}")

        logger.debug(f"Forward pass: batch_size={batch_size}, seq_len={seq_len}, device={device}")
        logger.debug(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape if attention_mask is not None else None}")
        logger.debug(f"Labels present: {labels is not None}, Domain IDs present: {domain_ids is not None}")

        # Base model forward pass
        logger.debug("Running base model forward pass")
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        logger.debug(f"Base model outputs keys: {base_outputs.keys() if hasattr(base_outputs, 'keys') else 'N/A'}")

        hidden_states = base_outputs.hidden_states[-1]  # Last layer
        base_loss = base_outputs.loss if labels is not None else None

        outputs = {"hidden_states": hidden_states}

        # Domain adaptation
        if domain_ids is not None:
            # Apply domain-specific adapters
            adapted_hidden_states = self._apply_domain_adapters(hidden_states, domain_ids)
            outputs["adapted_hidden_states"] = adapted_hidden_states

            # Domain classification loss (for adversarial training)
            domain_logits = self.domain_classifier(hidden_states.mean(dim=1))
            outputs["domain_logits"] = domain_logits

            if domain_ids is not None:
                domain_loss = F.cross_entropy(domain_logits, domain_ids)
                outputs["domain_loss"] = domain_loss

        # Difficulty prediction
        if return_difficulty or difficulty_targets is not None:
            difficulty_scores = self.difficulty_predictor(hidden_states.mean(dim=1))
            outputs["difficulty_scores"] = difficulty_scores

            if difficulty_targets is not None:
                difficulty_loss = F.mse_loss(difficulty_scores.squeeze(), difficulty_targets)
                outputs["difficulty_loss"] = difficulty_loss

        # Main language modeling loss
        if labels is not None:
            if "adapted_hidden_states" in outputs:
                # Recompute logits with adapted hidden states
                lm_head = self.base_model.lm_head if hasattr(self.base_model, 'lm_head') else self.base_model.get_output_embeddings()
                adapted_logits = lm_head(outputs["adapted_hidden_states"])

                # Shift for causal LM
                shift_logits = adapted_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                outputs["lm_loss"] = lm_loss
            else:
                outputs["lm_loss"] = base_loss

        # Forgetting regularization loss
        ewc_loss = self.forgetting_regularizer.compute_loss(self.base_model)
        outputs["ewc_loss"] = ewc_loss

        # Combine losses
        total_loss = 0.0
        if "lm_loss" in outputs:
            total_loss += outputs["lm_loss"]

        if "domain_loss" in outputs:
            # Adversarial weight (negative for domain invariance)
            total_loss -= self.domain_adversarial_weight * outputs["domain_loss"]

        if "difficulty_loss" in outputs:
            total_loss += self.difficulty_loss_weight * outputs["difficulty_loss"]

        if ewc_loss is not None:
            total_loss += ewc_loss

        outputs["loss"] = total_loss
        outputs["logits"] = base_outputs.logits

        return outputs

    def _apply_domain_adapters(
        self,
        hidden_states: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply domain-specific adapters.

        Args:
            hidden_states: Input hidden states.
            domain_ids: Domain identifiers for each sample.

        Returns:
            Adapted hidden states.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        adapted_states = torch.zeros_like(hidden_states)

        for domain_id in range(self.num_domains):
            mask = (domain_ids == domain_id)
            if mask.any():
                domain_hidden = hidden_states[mask]
                adapted_domain = self.domain_adapters[domain_id](domain_hidden)
                adapted_states[mask] = adapted_domain

        return adapted_states

    def get_difficulty_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get difficulty scores for inputs.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Difficulty scores.
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_difficulty=True
            )
            return outputs["difficulty_scores"]

    def update_forgetting_regularizer(self, dataset: Dataset) -> None:
        """Update Fisher Information Matrix for EWC.

        Args:
            dataset: Dataset to compute Fisher Information on.
        """
        self.forgetting_regularizer.update_fisher_information(self.base_model, dataset)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and tokenizer.

        Args:
            save_directory: Directory to save to.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base model and tokenizer
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save additional components
        torch.save({
            'domain_embedding': self.domain_embedding.state_dict(),
            'domain_adapters': [adapter.state_dict() for adapter in self.domain_adapters],
            'difficulty_predictor': self.difficulty_predictor.state_dict(),
            'domain_classifier': self.domain_classifier.state_dict(),
            'forgetting_regularizer': self.forgetting_regularizer.state_dict(),
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'use_adapter': self.use_adapter,
                'adapter_rank': self.adapter_rank,
                'adapter_alpha': self.adapter_alpha,
                'dropout': self.dropout,
                'num_domains': self.num_domains,
            }
        }, os.path.join(save_directory, 'additional_components.pt'))

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'AdaptiveCurriculumModel':
        """Load model from directory.

        Args:
            model_path: Path to saved model.

        Returns:
            Loaded model instance.
        """
        import os

        # Load additional components
        components_path = os.path.join(model_path, 'additional_components.pt')
        components = torch.load(components_path, map_location='cpu')
        config = components['config']

        # Create model instance
        model = cls(**config)

        # Load state dicts
        model.domain_embedding.load_state_dict(components['domain_embedding'])
        for i, state_dict in enumerate(components['domain_adapters']):
            model.domain_adapters[i].load_state_dict(state_dict)
        model.difficulty_predictor.load_state_dict(components['difficulty_predictor'])
        model.domain_classifier.load_state_dict(components['domain_classifier'])
        model.forgetting_regularizer.load_state_dict(components['forgetting_regularizer'])

        logger.info(f"Model loaded from {model_path}")
        return model


class DomainAdapter(nn.Module):
    """Domain-specific adapter layer."""

    def __init__(
        self,
        hidden_size: int,
        adapter_rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Initialize domain adapter.

        Args:
            hidden_size: Hidden layer size.
            adapter_rank: Adapter bottleneck rank.
            alpha: Scaling parameter.
            dropout: Dropout probability.
        """
        super().__init__()
        self.adapter_rank = adapter_rank
        self.alpha = alpha

        self.down_projection = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.up_projection = nn.Linear(adapter_rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize adapter weights.

        Args:
            init_std: Standard deviation for weight initialization.
        """
        nn.init.normal_(self.down_projection.weight, std=init_std)
        nn.init.zeros_(self.up_projection.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Adapted hidden states.
        """
        residual = hidden_states
        hidden_states = self.down_projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up_projection(hidden_states)

        # Scale and add residual
        scaling = self.alpha / self.adapter_rank
        return residual + scaling * hidden_states


class DifficultyPredictor(nn.Module):
    """Neural network for predicting question difficulty."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize difficulty predictor.

        Args:
            hidden_size: Hidden layer size.
            num_layers: Number of layers.
            dropout: Dropout probability.
        """
        super().__init__()
        layers = []

        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_size // 2, 1))
            else:
                layers.extend([
                    nn.Linear(hidden_size // 2, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])

        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Difficulty scores.
        """
        return self.layers(hidden_states)


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training."""

    def __init__(
        self,
        hidden_size: int,
        num_domains: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize domain classifier.

        Args:
            hidden_size: Hidden layer size.
            num_domains: Number of domains.
            dropout: Dropout probability.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Domain logits.
        """
        return self.classifier(hidden_states)


class ForgettingRegularizer(nn.Module):
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""

    def __init__(self, model_size: int, lambda_ewc: float = 0.1) -> None:
        """Initialize forgetting regularizer.

        Args:
            model_size: Number of model parameters.
            lambda_ewc: EWC regularization strength.
        """
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.register_buffer('fisher_information', torch.zeros(model_size))
        self.register_buffer('optimal_params', torch.zeros(model_size))
        self._param_names: List[str] = []

    def update_fisher_information(self, model: nn.Module, dataset: Dataset) -> None:
        """Update Fisher Information Matrix.

        Args:
            model: Model to compute Fisher Information for.
            dataset: Dataset to use for computation.
        """
        model.eval()
        fisher = torch.zeros_like(self.fisher_information)
        optimal_params = torch.zeros_like(self.optimal_params)

        # Store parameter names for consistency
        param_names = []
        param_idx = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_names.append(name)
                param_size = param.numel()
                optimal_params[param_idx:param_idx + param_size] = param.data.view(-1)
                param_idx += param_size

        self._param_names = param_names

        # Compute Fisher Information (simplified version)
        n_samples = min(len(dataset), 1000)  # Limit samples for efficiency

        for i in range(n_samples):
            model.zero_grad()

            # Mock loss computation (simplified)
            # In practice, this should use actual task loss
            dummy_loss = sum(p.sum() for p in model.parameters()) * 0.0001

            dummy_loss.backward()

            param_idx = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_size = param.numel()
                    fisher[param_idx:param_idx + param_size] += param.grad.data.view(-1) ** 2
                    param_idx += param_size

        # Normalize by number of samples
        fisher /= n_samples

        # Update buffers
        self.fisher_information.copy_(fisher)
        self.optimal_params.copy_(optimal_params)

        logger.info("Updated Fisher Information Matrix")

    def compute_loss(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Compute EWC regularization loss.

        Args:
            model: Current model.

        Returns:
            EWC loss or None if not initialized.
        """
        if self.fisher_information.sum() == 0:
            return None

        loss = 0.0
        param_idx = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_size = param.numel()
                param_range = slice(param_idx, param_idx + param_size)

                # EWC penalty
                fisher_slice = self.fisher_information[param_range]
                optimal_slice = self.optimal_params[param_range]
                current_params = param.view(-1)

                penalty = fisher_slice * (current_params - optimal_slice) ** 2
                loss += penalty.sum()

                param_idx += param_size

        return self.lambda_ewc * loss


class CurriculumScheduler:
    """Schedules curriculum learning progression.

    This class manages the ordering and pacing of training samples
    based on difficulty and domain similarity scores.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        difficulty_window: float = 0.3,
        similarity_threshold: float = 0.7,
        pace: str = "linear",
        total_steps: int = 1000,
    ) -> None:
        """Initialize curriculum scheduler.

        Args:
            strategy: Scheduling strategy ('adaptive', 'fixed', 'random').
            difficulty_window: Initial proportion of easy samples.
            similarity_threshold: Minimum similarity for inclusion.
            pace: Pacing strategy ('linear', 'exponential', 'adaptive').
            total_steps: Total number of training steps.
        """
        self.strategy = strategy
        self.difficulty_window = difficulty_window
        self.similarity_threshold = similarity_threshold
        self.pace = pace
        self.total_steps = total_steps
        self.current_step = 0

    def get_curriculum_indices(
        self,
        difficulty_scores: np.ndarray,
        similarity_scores: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ) -> np.ndarray:
        """Get indices for current curriculum step.

        Args:
            difficulty_scores: Array of difficulty scores.
            similarity_scores: Array of similarity scores (optional).
            step: Current training step.

        Returns:
            Array of indices to include in training.
        """
        if step is not None:
            self.current_step = step

        if self.strategy == "random":
            return np.arange(len(difficulty_scores))
        elif self.strategy == "fixed":
            return self._get_fixed_curriculum_indices(difficulty_scores, similarity_scores)
        elif self.strategy == "adaptive":
            return self._get_adaptive_curriculum_indices(difficulty_scores, similarity_scores)
        else:
            raise ValueError(f"Unknown curriculum strategy: {self.strategy}")

    def _get_fixed_curriculum_indices(
        self,
        difficulty_scores: np.ndarray,
        similarity_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get indices for fixed curriculum.

        Args:
            difficulty_scores: Array of difficulty scores.
            similarity_scores: Array of similarity scores.

        Returns:
            Array of selected indices.
        """
        # Compute current difficulty threshold
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.pace == "linear":
            difficulty_threshold = self.difficulty_window + (1.0 - self.difficulty_window) * progress
        elif self.pace == "exponential":
            difficulty_threshold = self.difficulty_window + (1.0 - self.difficulty_window) * (progress ** 2)
        else:  # adaptive
            difficulty_threshold = self.difficulty_window + (1.0 - self.difficulty_window) * np.sqrt(progress)

        # Select samples based on difficulty
        difficulty_mask = difficulty_scores <= difficulty_threshold

        # Apply similarity filtering if provided
        if similarity_scores is not None:
            similarity_mask = similarity_scores >= self.similarity_threshold
            combined_mask = difficulty_mask & similarity_mask
            # Fall back to difficulty-only if combined mask is empty
            if not np.any(combined_mask):
                combined_mask = difficulty_mask
        else:
            combined_mask = difficulty_mask

        indices = np.where(combined_mask)[0]
        # If still empty, return all indices
        if len(indices) == 0:
            return np.arange(len(difficulty_scores))
        return indices

    def _get_adaptive_curriculum_indices(
        self,
        difficulty_scores: np.ndarray,
        similarity_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get indices for adaptive curriculum.

        Args:
            difficulty_scores: Array of difficulty scores.
            similarity_scores: Array of similarity scores.

        Returns:
            Array of selected indices.
        """
        # Sort by difficulty
        sorted_indices = np.argsort(difficulty_scores)

        # Compute adaptive window size
        progress = min(self.current_step / self.total_steps, 1.0)
        window_size = int(self.difficulty_window * len(difficulty_scores) +
                         (len(difficulty_scores) - self.difficulty_window * len(difficulty_scores)) * progress)

        # Select easiest samples within window
        candidate_indices = sorted_indices[:window_size]

        # Apply similarity filtering if provided
        if similarity_scores is not None:
            similarity_mask = similarity_scores[candidate_indices] >= self.similarity_threshold
            selected_indices = candidate_indices[similarity_mask]
            # Fall back to all candidates if similarity filtering removes everything
            if len(selected_indices) == 0:
                logger.warning(
                    f"Similarity filtering removed all samples (threshold={self.similarity_threshold}). "
                    f"Using all {len(candidate_indices)} candidates."
                )
                selected_indices = candidate_indices
        else:
            selected_indices = candidate_indices

        return selected_indices

    def update_step(self) -> None:
        """Update current training step."""
        self.current_step += 1

    def get_progress(self) -> float:
        """Get current curriculum progress.

        Returns:
            Progress as fraction of total steps.
        """
        return min(self.current_step / self.total_steps, 1.0)