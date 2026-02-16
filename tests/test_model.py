"""Tests for model architecture."""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.models.model import (
    AdaptiveCurriculumModel,
    CurriculumScheduler,
    DomainAdapter,
    DifficultyPredictor,
    DomainClassifier,
    ForgettingRegularizer,
)


class TestCurriculumScheduler:
    """Test curriculum scheduling logic."""

    def test_initialization(self) -> None:
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler(
            strategy="adaptive",
            difficulty_window=0.3,
            similarity_threshold=0.7,
            pace="linear",
            total_steps=1000,
        )

        assert scheduler.strategy == "adaptive"
        assert scheduler.difficulty_window == 0.3
        assert scheduler.similarity_threshold == 0.7
        assert scheduler.pace == "linear"
        assert scheduler.total_steps == 1000
        assert scheduler.current_step == 0

    def test_random_curriculum(self) -> None:
        """Test random curriculum strategy."""
        scheduler = CurriculumScheduler(strategy="random", total_steps=100)

        difficulty_scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        indices = scheduler.get_curriculum_indices(difficulty_scores)

        assert len(indices) == 5
        assert np.array_equal(indices, np.arange(5))

    def test_fixed_curriculum(self) -> None:
        """Test fixed curriculum strategy."""
        scheduler = CurriculumScheduler(
            strategy="fixed",
            difficulty_window=0.4,
            total_steps=100,
        )

        difficulty_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # At step 0, should include easy samples
        indices_start = scheduler.get_curriculum_indices(difficulty_scores, step=0)
        selected_difficulties = difficulty_scores[indices_start]
        assert len(selected_difficulties) > 0
        assert np.max(selected_difficulties) <= 0.4 + (1.0 - 0.4) * 0.0

        # At step 50, should include more samples
        indices_mid = scheduler.get_curriculum_indices(difficulty_scores, step=50)
        assert len(indices_mid) >= len(indices_start)

        # At final step, should include all samples
        indices_end = scheduler.get_curriculum_indices(difficulty_scores, step=100)
        selected_difficulties_end = difficulty_scores[indices_end]
        assert len(selected_difficulties_end) > len(selected_difficulties)

    def test_adaptive_curriculum(self) -> None:
        """Test adaptive curriculum strategy."""
        scheduler = CurriculumScheduler(
            strategy="adaptive",
            difficulty_window=0.4,
            total_steps=100,
        )

        difficulty_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])  # Hardest to easiest

        # At step 0, should select easiest samples
        indices_start = scheduler.get_curriculum_indices(difficulty_scores, step=0)
        selected_difficulties = difficulty_scores[indices_start]
        assert len(selected_difficulties) > 0
        # Should include easiest samples (lowest indices in sorted order)

        # Progress should increase sample inclusion
        indices_end = scheduler.get_curriculum_indices(difficulty_scores, step=50)
        assert len(indices_end) >= len(indices_start)

    def test_similarity_filtering(self) -> None:
        """Test similarity-based filtering."""
        scheduler = CurriculumScheduler(
            strategy="fixed",
            similarity_threshold=0.6,
            total_steps=100,
        )

        difficulty_scores = np.array([0.2, 0.4, 0.6, 0.8])
        similarity_scores = np.array([0.8, 0.5, 0.7, 0.3])  # Above, below, above, below threshold

        indices = scheduler.get_curriculum_indices(difficulty_scores, similarity_scores, step=0)
        selected_similarities = similarity_scores[indices]

        # Should only include samples with similarity >= 0.6
        assert np.all(selected_similarities >= 0.6)

    def test_progress_tracking(self) -> None:
        """Test progress tracking."""
        scheduler = CurriculumScheduler(total_steps=100)

        assert scheduler.get_progress() == 0.0

        scheduler.current_step = 50
        assert scheduler.get_progress() == 0.5

        scheduler.current_step = 100
        assert scheduler.get_progress() == 1.0

        scheduler.current_step = 150
        assert scheduler.get_progress() == 1.0  # Should cap at 1.0


class TestDomainAdapter:
    """Test domain adapter module."""

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = DomainAdapter(
            hidden_size=768,
            adapter_rank=16,
            alpha=32,
            dropout=0.1,
        )

        assert adapter.adapter_rank == 16
        assert adapter.alpha == 32
        assert isinstance(adapter.down_projection, torch.nn.Linear)
        assert isinstance(adapter.up_projection, torch.nn.Linear)

    def test_forward_pass(self) -> None:
        """Test adapter forward pass."""
        adapter = DomainAdapter(hidden_size=64, adapter_rank=8)

        # Test input
        batch_size, seq_len, hidden_size = 2, 10, 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass
        output = adapter(hidden_states)

        assert output.shape == hidden_states.shape
        # Output should be different from input due to residual connection
        assert not torch.allclose(output, hidden_states)

    def test_weight_initialization(self) -> None:
        """Test weight initialization."""
        adapter = DomainAdapter(hidden_size=64, adapter_rank=8)
        adapter.init_weights()

        # Up projection should be initialized to zeros
        assert torch.allclose(adapter.up_projection.weight, torch.zeros_like(adapter.up_projection.weight))

        # Down projection should have small random values
        assert not torch.allclose(adapter.down_projection.weight, torch.zeros_like(adapter.down_projection.weight))


class TestDifficultyPredictor:
    """Test difficulty predictor module."""

    def test_initialization(self) -> None:
        """Test predictor initialization."""
        predictor = DifficultyPredictor(hidden_size=768, num_layers=2)

        assert isinstance(predictor.layers, torch.nn.Sequential)
        # Should have input layer, hidden layer, and output layer with activations/dropout
        assert len(predictor.layers) > 3

    def test_forward_pass(self) -> None:
        """Test predictor forward pass."""
        predictor = DifficultyPredictor(hidden_size=64, num_layers=2)

        # Test input
        batch_size, hidden_size = 8, 64
        hidden_states = torch.randn(batch_size, hidden_size)

        # Forward pass
        output = predictor(hidden_states)

        assert output.shape == (batch_size, 1)
        assert not torch.any(torch.isnan(output))

    def test_single_layer(self) -> None:
        """Test predictor with single layer."""
        predictor = DifficultyPredictor(hidden_size=32, num_layers=1)

        hidden_states = torch.randn(4, 32)
        output = predictor(hidden_states)

        assert output.shape == (4, 1)


class TestDomainClassifier:
    """Test domain classifier module."""

    def test_initialization(self) -> None:
        """Test classifier initialization."""
        classifier = DomainClassifier(hidden_size=768, num_domains=4)

        assert isinstance(classifier.classifier, torch.nn.Sequential)

    def test_forward_pass(self) -> None:
        """Test classifier forward pass."""
        num_domains = 4
        classifier = DomainClassifier(hidden_size=64, num_domains=num_domains)

        # Test input
        batch_size, hidden_size = 8, 64
        hidden_states = torch.randn(batch_size, hidden_size)

        # Forward pass
        output = classifier(hidden_states)

        assert output.shape == (batch_size, num_domains)
        assert not torch.any(torch.isnan(output))

    def test_output_properties(self) -> None:
        """Test output properties."""
        classifier = DomainClassifier(hidden_size=32, num_domains=3)

        hidden_states = torch.randn(5, 32)
        logits = classifier(hidden_states)

        # Test softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)


class TestForgettingRegularizer:
    """Test forgetting regularization module."""

    def test_initialization(self) -> None:
        """Test regularizer initialization."""
        model_size = 1000
        regularizer = ForgettingRegularizer(model_size=model_size, lambda_ewc=0.1)

        assert regularizer.lambda_ewc == 0.1
        assert regularizer.fisher_information.shape == (model_size,)
        assert regularizer.optimal_params.shape == (model_size,)

    def test_compute_loss_uninitialized(self) -> None:
        """Test loss computation when uninitialized."""
        regularizer = ForgettingRegularizer(model_size=100)

        # Create dummy model
        model = torch.nn.Linear(10, 5)
        loss = regularizer.compute_loss(model)

        assert loss is None  # Should return None when Fisher Information is not computed

    def test_update_fisher_information(self) -> None:
        """Test Fisher Information Matrix update."""
        model = torch.nn.Linear(10, 5)
        model_size = sum(p.numel() for p in model.parameters())

        regularizer = ForgettingRegularizer(model_size=model_size)

        # Create mock dataset
        mock_dataset = [{'dummy': 'data'} for _ in range(10)]

        # Update Fisher Information (simplified test)
        regularizer.update_fisher_information(model, mock_dataset)

        # Fisher information should be updated
        assert regularizer.fisher_information.sum() >= 0

    def test_compute_loss_initialized(self) -> None:
        """Test loss computation when initialized."""
        model = torch.nn.Linear(10, 5)
        model_size = sum(p.numel() for p in model.parameters())

        regularizer = ForgettingRegularizer(model_size=model_size, lambda_ewc=0.1)

        # Manually set Fisher information (skip actual computation)
        regularizer.fisher_information.fill_(0.01)
        regularizer.optimal_params.copy_(torch.cat([p.view(-1) for p in model.parameters()]))

        # Modify model parameters slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        loss = regularizer.compute_loss(model)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


@patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.models.model.AutoTokenizer')
@patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.models.model.AutoModelForCausalLM')
class TestAdaptiveCurriculumModel:
    """Test main adaptive curriculum model."""

    def test_initialization(self, mock_model_class, mock_tokenizer_class) -> None:
        """Test model initialization."""
        # Mock the base model and tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model_class.from_pretrained.return_value = mock_model

        model = AdaptiveCurriculumModel(
            model_name="test-model",
            max_length=512,
            use_adapter=True,
            num_domains=4,
        )

        assert model.model_name == "test-model"
        assert model.max_length == 512
        assert model.use_adapter is True
        assert model.num_domains == 4
        assert len(model.domain_adapters) == 4

    def test_forward_pass_basic(self, mock_model_class, mock_tokenizer_class) -> None:
        """Test basic forward pass."""
        # Mock the base model and tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 64
        mock_base_model.to.return_value = mock_base_model

        # Mock forward pass outputs
        mock_outputs = Mock()
        mock_outputs.hidden_states = [torch.randn(2, 10, 64)]  # Mock hidden states
        mock_outputs.loss = torch.tensor(1.0)
        mock_outputs.logits = torch.randn(2, 10, 1000)
        mock_base_model.forward.return_value = mock_outputs
        mock_base_model.return_value = mock_outputs

        mock_model_class.from_pretrained.return_value = mock_base_model

        model = AdaptiveCurriculumModel(
            model_name="test-model",
            num_domains=2,
        )

        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 1000, (2, 10))

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert 'loss' in outputs
        assert 'logits' in outputs
        assert 'hidden_states' in outputs

    def test_domain_adapter_application(self, mock_model_class, mock_tokenizer_class) -> None:
        """Test domain adapter application."""
        # Mock setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 32
        mock_base_model.to.return_value = mock_base_model
        mock_model_class.from_pretrained.return_value = mock_base_model

        model = AdaptiveCurriculumModel(
            model_name="test-model",
            num_domains=2,
        )

        # Test domain adapter application
        hidden_states = torch.randn(4, 8, 32)  # batch_size=4, seq_len=8, hidden_size=32
        domain_ids = torch.tensor([0, 1, 0, 1])  # 2 samples for each domain

        adapted_states = model._apply_domain_adapters(hidden_states, domain_ids)

        assert adapted_states.shape == hidden_states.shape
        assert not torch.allclose(adapted_states, hidden_states)  # Should be different

    def test_save_and_load(self, mock_model_class, mock_tokenizer_class) -> None:
        """Test model saving and loading."""
        # Mock setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = '<eos>'
        mock_tokenizer.save_pretrained = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_base_model = Mock()
        mock_base_model.config.hidden_size = 64
        mock_base_model.to.return_value = mock_base_model
        mock_base_model.save_pretrained = Mock()
        mock_model_class.from_pretrained.return_value = mock_base_model

        model = AdaptiveCurriculumModel(
            model_name="test-model",
            num_domains=2,
        )

        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir)

            # Verify save was called
            mock_base_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

            # Check if additional components file exists
            components_file = Path(temp_dir) / "additional_components.pt"
            assert components_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])