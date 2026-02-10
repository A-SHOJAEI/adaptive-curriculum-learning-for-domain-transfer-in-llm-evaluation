#!/usr/bin/env python3
"""Evaluation script for adaptive curriculum learning."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation import (
    Config,
    MMluDataLoader,
    AdaptiveCurriculumModel,
    CurriculumEvaluator,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate adaptive curriculum learning model")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        help="Path to baseline results JSON file for comparison",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate detailed evaluation report",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per domain (for debugging)",
    )
    parser.add_argument(
        "--compute-significance",
        action="store_true",
        help="Compute statistical significance tests",
    )

    return parser.parse_args()


def load_model_and_config(model_path: str, config_path: str) -> tuple:
    """Load trained model and configuration.

    Args:
        model_path: Path to trained model.
        config_path: Path to configuration file.

    Returns:
        Tuple of (model, tokenizer, config).

    Raises:
        FileNotFoundError: If model or config file doesn't exist.
        RuntimeError: If model loading fails.
        ValueError: If loaded model is invalid.
    """
    logger.info(f"Loading model from {model_path}")

    try:
        # Validate paths exist
        from pathlib import Path
        model_path_obj = Path(model_path)
        config_path_obj = Path(config_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config path does not exist: {config_path}")

        logger.debug(f"Loading model from verified path: {model_path}")

        # Load model
        model = AdaptiveCurriculumModel.from_pretrained(model_path)
        if model is None:
            raise RuntimeError("Model loading returned None")

        # Validate model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            raise ValueError("Loaded model has no parameters")

        # Load configuration
        config = Config(config_path)
        if config is None:
            raise RuntimeError("Config loading returned None")

        # Validate tokenizer is available
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            raise ValueError("Model does not have a valid tokenizer")

        logger.info("Model and configuration loaded successfully")
        logger.debug(f"Model has {total_params:,} parameters")

        return model, model.tokenizer, config

    except FileNotFoundError as e:
        logger.error(f"File not found during model loading: {e}")
        raise
    except (RuntimeError, ValueError) as e:
        logger.error(f"Model loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed due to unexpected error: {e}") from e


def load_test_datasets(config: Config, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Load test datasets for evaluation.

    Args:
        config: Configuration object.
        max_samples: Maximum samples per domain (optional).

    Returns:
        Dictionary containing test datasets.

    Raises:
        RuntimeError: If dataset loading fails.
        ValueError: If configuration is invalid or no test data available.
    """
    logger.info("Loading test datasets")

    try:
        # Validate configuration
        dataset_name = config.get('data.dataset_name')
        if not dataset_name:
            raise ValueError("Missing required configuration: data.dataset_name")

        # Initialize data loader
        data_loader = MMluDataLoader(
            dataset_name=dataset_name,
            cache_dir=config.get('data.cache_dir'),
            max_samples_per_domain=max_samples or config.get('data.max_samples_per_domain'),
            random_seed=config.get('data.random_seed'),
        )

        # Load dataset
        dataset = data_loader.load_dataset()
        if not dataset:
            raise RuntimeError(f"Failed to load dataset: {dataset_name}")

        # Get source and target domains from config
        source_domains = config.get('evaluation.source_domains', [])
        target_domains = config.get('evaluation.target_domains', [])

        if not source_domains and not target_domains:
            raise ValueError("At least one of source_domains or target_domains must be specified")

        logger.debug(f"Source domains: {source_domains}")
        logger.debug(f"Target domains: {target_domains}")

        # Create domain-specific splits
        domain_splits = data_loader.create_domain_splits(
            source_domains=source_domains,
            target_domains=target_domains,
            validation_split=config.get('data.validation_split'),
            test_split=config.get('data.test_split'),
        )

        if not domain_splits:
            raise RuntimeError("Failed to create domain splits - no valid data found")

        # Check if test data exists
        test_splits = {name: dataset for name, dataset in domain_splits.items() if 'test' in name}
        if not test_splits:
            logger.warning("No test splits found in domain_splits")

        # Create domain mapping
        subject_to_domain = {}
        try:
            if dataset and len(dataset) > 0:
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()

                # Validate required columns exist
                if 'subject' not in df.columns or 'domain' not in df.columns:
                    logger.warning("Dataset missing 'subject' or 'domain' columns")
                else:
                    subject_to_domain = dict(zip(df['subject'], df['domain']))
                    logger.debug(f"Created domain mapping for {len(subject_to_domain)} subjects")
        except Exception as e:
            logger.warning(f"Failed to create domain mapping: {e}")

        return {
            'datasets': domain_splits,
            'domain_mapping': subject_to_domain,
            'source_domains': source_domains,
            'target_domains': target_domains,
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"Test dataset loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during test dataset loading: {e}", exc_info=True)
        raise RuntimeError(f"Test dataset loading failed due to unexpected error: {e}") from e


def run_evaluation(
    model,
    tokenizer,
    config: Config,
    test_data: Dict,
    compute_significance: bool = False,
) -> Dict[str, Any]:
    """Run comprehensive evaluation.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        config: Configuration.
        test_data: Test datasets and metadata.
        compute_significance: Whether to compute statistical significance.

    Returns:
        Evaluation results.
    """
    logger.info("Starting comprehensive evaluation")

    # Create evaluator
    evaluator = CurriculumEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config.to_dict(),
    )

    # Load baseline results if available
    baseline_results = {}
    # This would typically load pre-computed baseline results
    # For now, we'll estimate them during evaluation

    evaluator.set_baseline_performance(baseline_results)

    # Run comprehensive evaluation
    test_datasets = test_data['datasets']
    domain_mapping = test_data['domain_mapping']

    # Filter for test splits only
    test_only_datasets = {
        name: dataset for name, dataset in test_datasets.items()
        if 'test' in name
    }

    if not test_only_datasets:
        logger.warning("No test datasets found, using all available datasets")
        test_only_datasets = test_datasets

    logger.info(f"Evaluating on {len(test_only_datasets)} datasets: {list(test_only_datasets.keys())}")

    # Run evaluation
    results = evaluator.evaluate_comprehensive(
        test_datasets=test_only_datasets,
        domain_mapping=domain_mapping,
    )

    logger.info("Evaluation completed")
    return results


def compare_with_targets(results: Dict, target_metrics: Dict[str, float]) -> Dict[str, bool]:
    """Compare results with target metrics.

    Args:
        results: Evaluation results.
        target_metrics: Target metric values.

    Returns:
        Dictionary indicating whether each target was met.
    """
    comparison = {}

    for metric_name, target_value in target_metrics.items():
        actual_value = results.get(metric_name, 0.0)
        met_target = actual_value >= target_value
        comparison[metric_name] = {
            'target': target_value,
            'actual': actual_value,
            'met': met_target,
            'difference': actual_value - target_value,
        }

    return comparison


def save_results(results: Dict, output_dir: Path, generate_report: bool = False) -> None:
    """Save evaluation results.

    Args:
        results: Evaluation results.
        output_dir: Output directory.
        generate_report: Whether to generate detailed report.
    """
    # Save raw results as JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    if generate_report:
        # Generate detailed report (this would use the evaluator's report generation)
        report_path = output_dir / "evaluation_report.txt"
        # Simplified report generation
        with open(report_path, 'w') as f:
            f.write("ADAPTIVE CURRICULUM LEARNING EVALUATION REPORT\\n")
            f.write("=" * 50 + "\\n\\n")

            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\\n")
                elif isinstance(value, dict) and len(value) < 10:  # Avoid huge dictionaries
                    f.write(f"{key}:\\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            f.write(f"  {sub_key}: {sub_value:.4f}\\n")
                    f.write("\\n")

        logger.info(f"Detailed report saved to {report_path}")


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and configuration
        model, tokenizer, config = load_model_and_config(args.model_path, args.config)

        # Load test datasets
        test_data = load_test_datasets(config, args.max_samples)

        logger.info(f"Loaded {len(test_data['datasets'])} test datasets")

        # Run evaluation
        results = run_evaluation(
            model=model,
            tokenizer=tokenizer,
            config=config,
            test_data=test_data,
            compute_significance=args.compute_significance,
        )

        # Compare with target metrics
        target_metrics = config.get('evaluation.target_metrics', {
            'cross_domain_transfer_gain': 0.15,
            'forgetting_rate_reduction': 0.4,
            'curriculum_efficiency_ratio': 2.5,
            'average_mmlu_accuracy': 0.72,
        })

        target_comparison = compare_with_targets(results, target_metrics)

        # Add target comparison to results
        results['target_comparison'] = target_comparison

        # Log key results
        logger.info("=" * 50)
        logger.info("KEY EVALUATION RESULTS:")
        logger.info("=" * 50)

        for metric, comparison in target_comparison.items():
            status = "✓" if comparison['met'] else "✗"
            logger.info(f"{metric}: {comparison['actual']:.4f} / {comparison['target']:.4f} {status}")

        logger.info("=" * 50)

        # Save results
        save_results(results, output_dir, args.generate_report)

        # Print summary
        met_targets = sum(1 for comp in target_comparison.values() if comp['met'])
        total_targets = len(target_comparison)

        print(f"\\nEVALUATION SUMMARY:")
        print(f"Met {met_targets}/{total_targets} target metrics")
        print(f"Results saved to: {output_dir}")

        if met_targets == total_targets:
            print("🎉 All target metrics achieved!")
        elif met_targets >= total_targets * 0.75:
            print("👍 Most target metrics achieved")
        else:
            print("⚠️  Several target metrics not met")

    except FileNotFoundError as e:
        logger.error(f"Evaluation failed - file not found: {e}")
        logger.error("Please check that the model path and config file exist")
        raise
    except RuntimeError as e:
        logger.error(f"Evaluation failed - runtime error: {e}")
        logger.error("This may be due to model loading issues, invalid data, or configuration problems")
        raise
    except ValueError as e:
        logger.error(f"Evaluation failed - invalid configuration or data: {e}")
        logger.error("Please check your configuration parameters and model compatibility")
        raise
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed with unexpected error: {e}", exc_info=True)
        logger.error("Please check the full traceback above for details")
        raise RuntimeError(f"Evaluation failed due to unexpected error: {e}") from e


if __name__ == "__main__":
    main()