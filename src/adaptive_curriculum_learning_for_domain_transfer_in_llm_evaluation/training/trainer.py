"""Training utilities for curriculum learning."""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from ..data.preprocessing import DifficultyEstimator, DomainSimilarityComputer
from ..models.model import AdaptiveCurriculumModel, CurriculumScheduler

logger = logging.getLogger(__name__)


def log_gpu_memory_usage() -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # Convert to GB
        logger.debug(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class CurriculumTrainer:
    """Trainer for curriculum learning with domain transfer.

    This trainer manages the curriculum learning process, including difficulty
    estimation, domain similarity computation, and adaptive scheduling.
    """

    def __init__(
        self,
        model: AdaptiveCurriculumModel,
        config: Dict,
        experiment_name: str = "adaptive_curriculum_learning",
    ) -> None:
        """Initialize curriculum trainer.

        Args:
            model: Adaptive curriculum model.
            config: Training configuration.
            experiment_name: MLflow experiment name.
        """
        self.model = model
        self.config = config
        self.experiment_name = experiment_name

        # Initialize components
        self.difficulty_estimator = DifficultyEstimator(
            method=config.get('curriculum', {}).get('difficulty_metric', 'entropy')
        )

        self.similarity_computer = DomainSimilarityComputer(
            method=config.get('curriculum', {}).get('similarity_metric', 'sentence_embeddings'),
            embedding_model=config.get('curriculum', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )

        self.curriculum_scheduler = CurriculumScheduler(
            strategy=config.get('curriculum', {}).get('curriculum_strategy', 'adaptive'),
            difficulty_window=config.get('curriculum', {}).get('difficulty_window', 0.3),
            similarity_threshold=config.get('curriculum', {}).get('similarity_threshold', 0.7),
            pace=config.get('curriculum', {}).get('curriculum_pace', 'linear'),
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        greater_is_better = config.get('training', {}).get('greater_is_better', False)
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.patience_counter = 0
        self.training_history: List[Dict] = []

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', {}).get('use_cuda', True) else 'cpu')
        self.model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.get('device', {}).get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # MLflow setup
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        mlflow_config = self.config.get('mlflow', {})

        try:
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()

            # Log configuration
            for key, value in self.config.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}.{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)

            logger.info("MLflow tracking initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        source_domains: Optional[List[str]] = None,
        target_domains: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train model with curriculum learning.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            source_domains: Source domain names.
            target_domains: Target domain names.

        Returns:
            Training metrics.
        """
        logger.info("Starting curriculum learning training")

        # Prepare datasets
        train_difficulty_scores = self._compute_difficulty_scores(train_dataset)
        train_similarity_scores = self._compute_similarity_scores(
            train_dataset, source_domains, target_domains
        )

        # Create domain mapping
        domain_mapping = self._create_domain_mapping(train_dataset)

        # Set up optimizer and scheduler
        optimizer = self._create_optimizer()
        lr_scheduler = self._create_lr_scheduler(optimizer, len(train_dataset))

        # Training configuration
        logger.info("Starting curriculum training")
        log_gpu_memory_usage()
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 5)
        batch_size = training_config.get('batch_size', 8)
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 4)
        eval_steps = training_config.get('eval_steps', 100)
        save_steps = training_config.get('save_steps', 500)
        logging_steps = training_config.get('logging_steps', 50)

        # Update scheduler total steps
        total_steps = (len(train_dataset) // batch_size) * num_epochs
        self.curriculum_scheduler.total_steps = total_steps

        # Training loop
        self.model.train()
        epoch_losses = []

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Get curriculum indices for this epoch
            logger.debug(f"Computing curriculum indices for epoch {epoch + 1} (global step: {self.global_step})")
            curriculum_indices = self.curriculum_scheduler.get_curriculum_indices(
                train_difficulty_scores,
                train_similarity_scores,
                self.global_step
            )
            logger.info(f"Curriculum selected {len(curriculum_indices)}/{len(train_dataset)} samples for epoch {epoch + 1}")

            # Create curriculum subset
            curriculum_subset = Subset(train_dataset, curriculum_indices)
            logger.debug(f"Creating data loader with batch_size={batch_size}, num_workers={training_config.get('dataloader_num_workers', 0)}")
            train_dataloader = DataLoader(
                curriculum_subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
                num_workers=training_config.get('dataloader_num_workers', 0),
            )

            epoch_loss = 0.0
            optimizer.zero_grad()

            logger.debug(f"Training on {len(train_dataloader)} batches for epoch {epoch + 1}")

            # Epoch training loop
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / gradient_accumulation_steps

                epoch_loss += loss.item()

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation and update
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            training_config.get('max_grad_norm', 1.0)
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            training_config.get('max_grad_norm', 1.0)
                        )
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    self.curriculum_scheduler.update_step()

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                        'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        'step': self.global_step
                    })

                    # Logging
                    if self.global_step % logging_steps == 0:
                        log_gpu_memory_usage()
                        self._log_metrics({
                            'train_loss': loss.item() * gradient_accumulation_steps,
                            'learning_rate': lr_scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'curriculum_progress': self.curriculum_scheduler.get_progress(),
                        })

                    # Evaluation
                    if eval_dataset is not None and self.global_step % eval_steps == 0:
                        eval_metrics = self.evaluate(eval_dataset, domain_mapping)
                        self._log_metrics(eval_metrics)

                        # Early stopping check
                        if self._should_stop_early(eval_metrics):
                            logger.info("Early stopping triggered")
                            return self._finalize_training(epoch_losses)

                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(epoch, self.global_step)

            epoch_losses.append(epoch_loss / len(train_dataloader))
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {epoch_losses[-1]:.4f}")
            logger.debug(f"Throughput: {len(train_dataloader) / epoch_time:.2f} batches/second")

            # Update Fisher Information for EWC after each epoch
            if hasattr(self.model, 'update_forgetting_regularizer'):
                logger.info("Updating forgetting regularizer")
                self.model.update_forgetting_regularizer(train_dataset)

        return self._finalize_training(epoch_losses)

    def evaluate(
        self,
        eval_dataset: Dataset,
        domain_mapping: Dict[str, int],
    ) -> Dict[str, float]:
        """Evaluate model on dataset.

        Args:
            eval_dataset: Evaluation dataset.
            domain_mapping: Mapping from domain names to IDs.

        Returns:
            Evaluation metrics.
        """
        logger.info("Running evaluation")
        self.model.eval()

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 8),
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs["loss"]
                logits = outputs["logits"]

                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                # Compute accuracy (simplified)
                if "labels" in batch:
                    predictions = torch.argmax(logits, dim=-1)
                    labels = batch["labels"]

                    # Only consider non-padding tokens
                    label_pad_token_id = self.config.get('training', {}).get('label_pad_token_id', -100)
                    mask = labels != label_pad_token_id
                    correct = (predictions == labels) & mask
                    correct_predictions += correct.sum().item()

        # Compute metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
        }

        self.model.train()
        logger.info(f"Evaluation completed. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return metrics

    def _compute_difficulty_scores(self, dataset: Dataset) -> np.ndarray:
        """Compute difficulty scores for dataset.

        Args:
            dataset: Dataset to score.

        Returns:
            Array of difficulty scores.
        """
        logger.info("Computing difficulty scores")

        if self.difficulty_estimator.method in ['confidence', 'loss']:
            # Use model for difficulty estimation
            self.difficulty_estimator.model = self.model.base_model
            self.difficulty_estimator.tokenizer = self.model.tokenizer

        return self.difficulty_estimator.estimate_difficulty(dataset)

    def _compute_similarity_scores(
        self,
        dataset: Dataset,
        source_domains: Optional[List[str]],
        target_domains: Optional[List[str]],
    ) -> Optional[np.ndarray]:
        """Compute domain similarity scores.

        Args:
            dataset: Dataset to analyze.
            source_domains: Source domain names.
            target_domains: Target domain names.

        Returns:
            Array of similarity scores or None.
        """
        if source_domains is None or target_domains is None:
            return None

        logger.info("Computing domain similarity scores")

        # Get all domains
        all_domains = list(set(source_domains + target_domains))

        # Compute domain similarity matrix
        similarity_matrix = self.similarity_computer.compute_domain_similarity(dataset, all_domains)

        # For each sample, compute similarity to target domains
        df = dataset.to_pandas()
        similarity_scores = np.zeros(len(dataset))

        for i, row in df.iterrows():
            sample_domain = row['domain']

            if sample_domain in all_domains:
                domain_idx = all_domains.index(sample_domain)

                # Compute average similarity to target domains
                target_similarities = []
                for target_domain in target_domains:
                    if target_domain in all_domains:
                        target_idx = all_domains.index(target_domain)
                        target_similarities.append(similarity_matrix[domain_idx, target_idx])

                if target_similarities:
                    similarity_scores[i] = np.mean(target_similarities)

        return similarity_scores

    def _create_domain_mapping(self, dataset: Dataset) -> Dict[str, int]:
        """Create mapping from domain names to integer IDs.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Domain mapping dictionary.
        """
        df = dataset.to_pandas()
        unique_domains = sorted(df['domain'].unique())
        return {domain: idx for idx, domain in enumerate(unique_domains)}

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer.

        Returns:
            Configured optimizer.
        """
        training_config = self.config.get('training', {})

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        lr = float(training_config.get('learning_rate', 2e-5))
        betas_raw = training_config.get('optimizer_betas', [0.9, 0.999])
        betas = tuple(float(b) for b in betas_raw)
        eps = float(training_config.get('optimizer_eps', 1e-8))

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    def _create_lr_scheduler(self, optimizer: torch.optim.Optimizer, dataset_size: int):
        """Create learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule.
            dataset_size: Size of training dataset.

        Returns:
            Learning rate scheduler.
        """
        training_config = self.config.get('training', {})

        total_steps = (
            dataset_size //
            (training_config.get('batch_size', 8) * training_config.get('gradient_accumulation_steps', 4))
        ) * training_config.get('num_epochs', 5)

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config.get('warmup_steps', 500),
            num_training_steps=total_steps,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of samples.

        Returns:
            Collated batch.
        """
        # Basic collation - in practice, you'd use DataCollatorWithPadding
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Pad sequences
                max_len = max(len(sample[key]) for sample in batch)
                padded = []

                for sample in batch:
                    sequence = sample[key]
                    if len(sequence) < max_len:
                        if key == 'labels':
                            label_pad_token_id = self.config.get('training', {}).get('label_pad_token_id', -100)
                            padding = [label_pad_token_id] * (max_len - len(sequence))
                        else:
                            attention_pad_token_id = self.config.get('training', {}).get('attention_pad_token_id', 0)
                            padding = [attention_pad_token_id] * (max_len - len(sequence))
                        sequence = sequence + padding

                    padded.append(sequence)

                collated[key] = torch.tensor(padded)

            elif key == 'domain':
                # Convert domain names to IDs (simplified)
                domain_mapping = {'STEM': 0, 'Social Sciences': 1, 'Humanities': 2, 'Other': 3}
                domain_ids = [domain_mapping.get(sample[key], 3) for sample in batch]
                collated['domain_ids'] = torch.tensor(domain_ids)

        return collated

    def _log_metrics(self, metrics: Dict[str, Union[float, int]]) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log.
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=self.global_step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def _should_stop_early(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered.

        Args:
            eval_metrics: Current evaluation metrics.

        Returns:
            True if training should stop early.
        """
        training_config = self.config.get('training', {})
        patience = training_config.get('early_stopping_patience', 3)
        threshold = training_config.get('early_stopping_threshold', 0.001)
        metric_name = training_config.get('metric_for_best_model', 'eval_loss')
        greater_is_better = training_config.get('greater_is_better', False)

        if metric_name not in eval_metrics:
            return False

        current_metric = eval_metrics[metric_name]

        if greater_is_better:
            improved = current_metric > self.best_metric + threshold
        else:
            improved = current_metric < self.best_metric - threshold

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"New best {metric_name}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement in {metric_name}. Patience: {self.patience_counter}/{patience}")

        return self.patience_counter >= patience

    def _save_checkpoint(self, epoch: int, step: int) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            step: Current step.
        """
        checkpoint_dir = f"checkpoint-epoch-{epoch}-step-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': step,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
        }, os.path.join(checkpoint_dir, 'training_state.pt'))

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _finalize_training(self, epoch_losses: List[float]) -> Dict[str, float]:
        """Finalize training and return summary metrics.

        Args:
            epoch_losses: List of epoch losses.

        Returns:
            Final training metrics.
        """
        final_metrics = {
            'final_train_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'best_metric': self.best_metric,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step,
        }

        # Log final metrics
        try:
            for key, value in final_metrics.items():
                mlflow.log_metric(key, value)
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to log final metrics: {e}")

        logger.info("Training completed successfully")
        return final_metrics