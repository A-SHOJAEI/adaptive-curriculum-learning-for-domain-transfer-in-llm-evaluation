#!/usr/bin/env python3
"""Training script for adaptive curriculum learning."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation import (
    Config,
    MMluDataLoader,
    AdaptiveCurriculumModel,
    CurriculumTrainer,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train adaptive curriculum learning model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per domain (for debugging)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="adaptive_curriculum_learning",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """Set up output directory.

    Args:
        output_dir: Output directory path.

    Returns:
        Output directory path as Path object.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)

    return output_path


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 256,
) -> Dataset:
    """Tokenize a dataset for causal LM training.

    Args:
        dataset: HuggingFace Dataset with 'formatted_question' and 'correct_answer_text' columns.
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset with input_ids, attention_mask, labels columns.
    """
    def tokenize_fn(examples):
        inputs = []
        targets = []

        for i in range(len(examples['formatted_question'])):
            question = examples['formatted_question'][i]
            answer = examples['correct_answer_text'][i]
            input_text = f"Please answer the following multiple choice question:\n\n{question}\n\nAnswer: {answer}"
            inputs.append(input_text)

        # Tokenize
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels = input_ids (shifted internally by model)
        model_inputs['labels'] = model_inputs['input_ids'].copy()

        # Preserve domain info if present
        if 'domain' in examples:
            model_inputs['domain'] = examples['domain']

        return model_inputs

    # Keep columns needed for difficulty estimation and domain similarity
    keep_cols = {'domain', 'subject', 'answer', 'formatted_question', 'correct_answer_text'}
    columns_to_remove = [
        col for col in dataset.column_names
        if col not in keep_cols
    ]

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )

    return tokenized


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = Config(args.config)

    # Override config with command line arguments
    if args.model_name:
        config.set('model.name', args.model_name)
    if args.num_epochs:
        config.set('training.num_epochs', args.num_epochs)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)

    # Set up output directory
    output_path = setup_output_directory(args.output_dir)
    logger.info(f"Output directory: {output_path}")

    # Save configuration
    config.save(output_path / "config.yaml")

    try:
        # ====== Step 1: Load MMLU dataset ======
        logger.info("=" * 60)
        logger.info("Step 1: Loading MMLU dataset")
        logger.info("=" * 60)

        dataset_name = config.get('data.dataset_name')
        cache_dir = config.get('data.cache_dir')
        max_samples = args.max_samples or config.get('data.max_samples_per_domain')

        data_loader = MMluDataLoader(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            max_samples_per_domain=max_samples,
            random_seed=config.get('data.random_seed', 42),
        )

        dataset = data_loader.load_dataset()
        logger.info(f"Loaded dataset with splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")

        # ====== Step 2: Create domain splits ======
        logger.info("=" * 60)
        logger.info("Step 2: Creating domain splits")
        logger.info("=" * 60)

        source_domains = config.get('evaluation.source_domains', ['STEM'])
        target_domains = config.get('evaluation.target_domains', ['Humanities'])

        logger.info(f"Source domains: {source_domains}")
        logger.info(f"Target domains: {target_domains}")

        domain_splits = data_loader.create_domain_splits(
            source_domains=source_domains,
            target_domains=target_domains,
            validation_split=config.get('data.validation_split', 0.1),
            test_split=config.get('data.test_split', 0.2),
        )

        for split_name, split_data in domain_splits.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")

        # ====== Step 3: Create model ======
        logger.info("=" * 60)
        logger.info("Step 3: Creating model")
        logger.info("=" * 60)

        all_domains = set(source_domains + target_domains)
        num_domains = len(all_domains)

        model_name = config.get('model.name')
        logger.info(f"Model: {model_name}, Domains: {num_domains}")

        model = AdaptiveCurriculumModel(
            model_name=model_name,
            max_length=config.get('model.max_length', 256),
            use_adapter=config.get('model.use_adapter', True),
            adapter_rank=config.get('model.adapter_rank', 8),
            adapter_alpha=config.get('model.adapter_alpha', 16),
            dropout=config.get('model.dropout', 0.1),
            num_domains=num_domains,
            config=config.to_dict(),
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # ====== Step 4: Tokenize datasets ======
        logger.info("=" * 60)
        logger.info("Step 4: Tokenizing datasets")
        logger.info("=" * 60)

        max_length = config.get('model.max_length', 256)

        train_dataset = domain_splits.get('source_train')
        eval_dataset = domain_splits.get('source_val')

        if train_dataset is None:
            raise ValueError("No source_train split found in domain splits")

        logger.info(f"Tokenizing train set ({len(train_dataset)} samples)...")
        tokenized_train = tokenize_dataset(train_dataset, model.tokenizer, max_length)

        tokenized_eval = None
        if eval_dataset is not None:
            logger.info(f"Tokenizing eval set ({len(eval_dataset)} samples)...")
            tokenized_eval = tokenize_dataset(eval_dataset, model.tokenizer, max_length)

        logger.info(f"Tokenized train columns: {tokenized_train.column_names}")
        logger.info(f"Tokenized train size: {len(tokenized_train)}")

        # ====== Step 5: Train ======
        logger.info("=" * 60)
        logger.info("Step 5: Starting curriculum training")
        logger.info("=" * 60)

        trainer = CurriculumTrainer(
            model=model,
            config=config.to_dict(),
            experiment_name=args.experiment_name,
        )

        start_time = time.time()

        training_results = trainer.train(
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            source_domains=source_domains,
            target_domains=target_domains,
        )

        training_time = time.time() - start_time

        # ====== Step 6: Save results ======
        logger.info("=" * 60)
        logger.info("Step 6: Saving results")
        logger.info("=" * 60)

        logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} min)")
        logger.info(f"Training results: {training_results}")

        # Save final model
        final_model_path = output_path / "models" / "final_model"
        model.save_pretrained(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        # Save training results
        results_path = output_path / "training_results.json"
        serializable_results = {}
        for k, v in training_results.items():
            if isinstance(v, (float, int)):
                serializable_results[k] = v
            elif isinstance(v, np.floating):
                serializable_results[k] = float(v)
            else:
                serializable_results[k] = str(v)
        serializable_results['training_time_seconds'] = training_time

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Training results saved to {results_path}")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        # Clean up MLflow
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
