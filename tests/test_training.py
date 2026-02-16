"""Tests for training modules."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer import CurriculumTrainer
from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.evaluation.metrics import CurriculumEvaluator


class TestCurriculumTrainer:
    """Test curriculum trainer functionality."""

    def create_mock_model(self) -> Mock:
        """Create mock model for testing."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.train.return_value = None
        mock_model.eval.return_value = None

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = '<pad>'
        mock_model.tokenizer = mock_tokenizer

        # Mock base model
        mock_model.base_model = Mock()

        # Mock forward pass
        mock_outputs = {
            'loss': torch.tensor(1.0),
            'logits': torch.randn(2, 10, 1000),
            'hidden_states': torch.randn(2, 10, 768)
        }
        mock_model.return_value = mock_outputs
        mock_model.forward.return_value = mock_outputs

        return mock_model

    def create_mock_dataset(self, size: int = 10) -> Mock:
        """Create mock dataset for testing."""
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = size

        # Mock pandas DataFrame
        import pandas as pd
        df_data = {
            'domain': ['STEM'] * (size // 2) + ['Social Sciences'] * (size - size // 2),
            'subject': ['math'] * (size // 2) + ['sociology'] * (size - size // 2),
            'formatted_question': [f'Question {i}' for i in range(size)],
            'choices': [['A', 'B', 'C', 'D']] * size,
            'answer': [i % 4 for i in range(size)],
            'correct_answer_text': [['A', 'B', 'C', 'D'][i % 4] for i in range(size)],
        }
        df = pd.DataFrame(df_data)
        mock_dataset.to_pandas.return_value = df

        # Mock dataset indexing
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: {
            'input_ids': torch.randint(0, 1000, (20,)),
            'attention_mask': torch.ones(20),
            'labels': torch.randint(0, 1000, (20,)),
            'domain': df.iloc[i]['domain'] if i < len(df) else 'Other'
        })

        return mock_dataset

    def create_mock_config(self) -> dict:
        """Create mock configuration for testing."""
        return {
            'curriculum': {
                'difficulty_metric': 'entropy',
                'similarity_metric': 'sentence_embeddings',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'curriculum_strategy': 'adaptive',
                'difficulty_window': 0.3,
                'similarity_threshold': 0.7,
                'curriculum_pace': 'linear',
            },
            'training': {
                'num_epochs': 2,
                'batch_size': 4,
                'gradient_accumulation_steps': 2,
                'learning_rate': 2e-5,
                'warmup_steps': 10,
                'weight_decay': 0.01,
                'max_grad_norm': 1.0,
                'eval_steps': 5,
                'save_steps': 10,
                'logging_steps': 2,
                'early_stopping_patience': 3,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
            },
            'device': {
                'use_cuda': False,  # Use CPU for testing
                'mixed_precision': False,
            },
            'mlflow': {
                'tracking_uri': 'mlruns',
                'experiment_name': 'test_experiment',
            },
            'data': {
                'random_seed': 42,
            }
        }

    def test_trainer_initialization(self) -> None:
        """Test trainer initialization."""
        model = self.create_mock_model()
        config = self.create_mock_config()

        with patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow'):
            trainer = CurriculumTrainer(model=model, config=config)

        assert trainer.model == model
        assert trainer.config == config
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.device.type == 'cpu'  # Since we disabled CUDA

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_difficulty_computation(self, mock_mlflow) -> None:
        """Test difficulty score computation."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        dataset = self.create_mock_dataset(20)
        difficulties = trainer._compute_difficulty_scores(dataset)

        assert isinstance(difficulties, np.ndarray)
        assert len(difficulties) == 20
        assert np.all(difficulties >= 0) and np.all(difficulties <= 1)

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.SentenceTransformer')
    def test_similarity_computation(self, mock_st, mock_mlflow) -> None:
        """Test domain similarity computation."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 128)
        mock_st.return_value = mock_model

        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        dataset = self.create_mock_dataset(20)
        source_domains = ['STEM']
        target_domains = ['Social Sciences']

        similarities = trainer._compute_similarity_scores(dataset, source_domains, target_domains)

        assert similarities is not None
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == 20

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_domain_mapping_creation(self, mock_mlflow) -> None:
        """Test domain mapping creation."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        dataset = self.create_mock_dataset(10)
        domain_mapping = trainer._create_domain_mapping(dataset)

        assert isinstance(domain_mapping, dict)
        assert 'STEM' in domain_mapping
        assert 'Social Sciences' in domain_mapping
        assert isinstance(list(domain_mapping.values())[0], int)

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_optimizer_creation(self, mock_mlflow) -> None:
        """Test optimizer creation."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        optimizer = trainer._create_optimizer()

        assert optimizer is not None
        # Should be AdamW optimizer
        assert 'AdamW' in str(type(optimizer))

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_lr_scheduler_creation(self, mock_mlflow) -> None:
        """Test learning rate scheduler creation."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        optimizer = trainer._create_optimizer()
        lr_scheduler = trainer._create_lr_scheduler(optimizer, dataset_size=100)

        assert lr_scheduler is not None
        # Should have warmup
        assert hasattr(lr_scheduler, 'get_last_lr')

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_collate_function(self, mock_mlflow) -> None:
        """Test collate function for data loading."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        # Create mock batch
        batch = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [4, 5, 6],
                'domain': 'STEM'
            },
            {
                'input_ids': [7, 8],
                'attention_mask': [1, 1],
                'labels': [9, 10],
                'domain': 'Social Sciences'
            }
        ]

        collated = trainer._collate_fn(batch)

        assert 'input_ids' in collated
        assert 'attention_mask' in collated
        assert 'labels' in collated
        assert 'domain_ids' in collated

        assert collated['input_ids'].shape[0] == 2  # batch size
        assert collated['input_ids'].shape[1] == 3  # max length

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_early_stopping_logic(self, mock_mlflow) -> None:
        """Test early stopping logic."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        # Test improvement
        eval_metrics = {'eval_loss': 0.5}
        trainer.best_metric = 1.0  # Higher is worse for loss
        should_stop = trainer._should_stop_early(eval_metrics)
        assert not should_stop  # Should continue training
        assert trainer.patience_counter == 0

        # Test no improvement
        eval_metrics = {'eval_loss': 1.5}  # Worse than best
        should_stop = trainer._should_stop_early(eval_metrics)
        assert not should_stop  # First time, should continue
        assert trainer.patience_counter == 1

        # Test exceeding patience
        trainer.patience_counter = 3
        should_stop = trainer._should_stop_early(eval_metrics)
        assert should_stop  # Should stop now

    @patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.training.trainer.mlflow')
    def test_evaluation(self, mock_mlflow) -> None:
        """Test evaluation during training."""
        model = self.create_mock_model()
        config = self.create_mock_config()
        trainer = CurriculumTrainer(model=model, config=config)

        dataset = self.create_mock_dataset(10)
        domain_mapping = {'STEM': 0, 'Social Sciences': 1}

        eval_metrics = trainer.evaluate(dataset, domain_mapping)

        assert isinstance(eval_metrics, dict)
        assert 'eval_loss' in eval_metrics
        assert 'eval_accuracy' in eval_metrics


class TestCurriculumEvaluator:
    """Test curriculum evaluator functionality."""

    def create_mock_model(self) -> Mock:
        """Create mock model for testing."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Mock forward pass
        mock_outputs = {
            'loss': torch.tensor(0.5),
            'logits': torch.randn(2, 10, 1000),
            'hidden_states': torch.randn(2, 10, 768)
        }
        mock_model.return_value = mock_outputs

        return mock_model

    def create_mock_dataset(self, size: int = 10) -> Mock:
        """Create mock dataset for testing."""
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = size

        # Mock pandas DataFrame
        import pandas as pd
        df_data = {
            'domain': ['STEM'] * (size // 2) + ['Social Sciences'] * (size - size // 2),
            'subject': ['math'] * (size // 2) + ['sociology'] * (size - size // 2),
            'formatted_question': [f'Question {i}' for i in range(size)],
        }
        df = pd.DataFrame(df_data)
        mock_dataset.to_pandas.return_value = df

        return mock_dataset

    def test_evaluator_initialization(self) -> None:
        """Test evaluator initialization."""
        model = self.create_mock_model()
        tokenizer = Mock()
        config = {'evaluation': {'batch_size': 8}}

        evaluator = CurriculumEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config
        )

        assert evaluator.model == model
        assert evaluator.tokenizer == tokenizer
        assert evaluator.config == config

    def test_accuracy_computation(self) -> None:
        """Test accuracy computation."""
        model = self.create_mock_model()
        tokenizer = Mock()
        config = {'evaluation': {'batch_size': 4}}

        # Mock DataLoader and model outputs
        with patch('adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation.evaluation.metrics.DataLoader') as mock_dataloader:
            # Mock batch data
            mock_batch = {
                'input_ids': torch.randint(0, 1000, (2, 10)),
                'attention_mask': torch.ones(2, 10),
                'labels': torch.randint(0, 1000, (2, 10))
            }

            # Mock model outputs with correct predictions
            mock_outputs = {
                'loss': torch.tensor(0.5),
                'logits': torch.zeros(2, 10, 1000)  # Will give predictions of 0
            }

            # Set up the mock to return our batch
            mock_dataloader.return_value.__iter__.return_value = [mock_batch]

            model.return_value = mock_outputs
            model.side_effect = None

            evaluator = CurriculumEvaluator(model=model, tokenizer=tokenizer, config=config)
            dataset = self.create_mock_dataset(10)

            accuracy = evaluator._compute_accuracy(dataset)

            assert isinstance(accuracy, float)
            assert 0 <= accuracy <= 1

    def test_bootstrap_accuracy(self) -> None:
        """Test bootstrap confidence intervals."""
        model = self.create_mock_model()
        tokenizer = Mock()
        config = {}

        evaluator = CurriculumEvaluator(model=model, tokenizer=tokenizer, config=config)

        # Mock the _compute_accuracy method to return consistent values
        evaluator._compute_accuracy = Mock(return_value=0.7)

        dataset = self.create_mock_dataset(20)
        bootstrap_accuracies = evaluator._bootstrap_accuracy(dataset, n_bootstrap=10)

        assert isinstance(bootstrap_accuracies, np.ndarray)
        assert len(bootstrap_accuracies) == 10
        # Since we mocked to return 0.7 always, all values should be 0.7
        assert np.all(bootstrap_accuracies == 0.7)

    def test_baseline_performance(self) -> None:
        """Test baseline performance handling."""
        model = self.create_mock_model()
        tokenizer = Mock()
        config = {}

        evaluator = CurriculumEvaluator(model=model, tokenizer=tokenizer, config=config)

        # Set baseline performance
        baseline_results = {'source': 0.8, 'target': 0.6}
        evaluator.set_baseline_performance(baseline_results)

        assert evaluator.baseline_performance['source'] == 0.8
        assert evaluator.baseline_performance['target'] == 0.6

        # Test getting baseline performance
        dataset = self.create_mock_dataset()
        baseline = evaluator._get_baseline_performance(dataset, 'source')
        assert baseline == 0.8

    def test_report_generation(self) -> None:
        """Test evaluation report generation."""
        model = self.create_mock_model()
        tokenizer = Mock()
        config = {'target_metrics': {'accuracy': 0.8, 'transfer_gain': 0.1}}

        evaluator = CurriculumEvaluator(model=model, tokenizer=tokenizer, config=config)

        # Mock results
        results = {
            'average_mmlu_accuracy': 0.75,
            'cross_domain_transfer_gain': 0.12,
            'forgetting_rate': 0.05,
            'source_test_accuracy': 0.82,
            'target_test_accuracy': 0.68,
            'domain_analysis': {
                'STEM': {'test_accuracy': 0.85},
                'Social Sciences': {'test_accuracy': 0.65}
            }
        }

        report = evaluator.generate_evaluation_report(results)

        assert isinstance(report, str)
        assert 'EVALUATION REPORT' in report
        assert '0.75' in report  # Should contain accuracy value
        assert 'STEM' in report  # Should contain domain analysis


if __name__ == "__main__":
    pytest.main([__file__])