#!/usr/bin/env python3
"""Inference script for trained adaptive curriculum model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_curriculum_learning_for_domain_transfer_in_llm_evaluation import (
    AdaptiveCurriculumModel,
    Config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, config_path: str) -> tuple:
    """Load trained model and tokenizer.

    Args:
        model_path: Path to saved model directory.
        config_path: Path to config file.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    logger.info(f"Loading configuration from {config_path}")
    config = Config(config_path)

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info(f"Loading model from {model_path}")
    model = AdaptiveCurriculumModel(
        model_name=model_path,
        num_domains=4,
        config=config.to_dict()
    )

    # Load saved state if available
    state_dict_path = Path(model_path) / "pytorch_model.bin"
    if state_dict_path.exists():
        logger.info("Loading saved model weights")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model, tokenizer, config


def format_question(question: str, choices: List[str], subject: str = "") -> str:
    """Format question with multiple choice options.

    Args:
        question: Question text.
        choices: List of answer choices.
        subject: Subject/domain of the question.

    Returns:
        Formatted question string.
    """
    formatted = f"Question: {question}\n"
    if subject:
        formatted = f"Subject: {subject}\n{formatted}"

    choice_labels = ["A", "B", "C", "D"]
    for label, choice in zip(choice_labels, choices):
        formatted += f"{label}. {choice}\n"

    formatted += "Answer:"
    return formatted


def predict(
    model: AdaptiveCurriculumModel,
    tokenizer: AutoTokenizer,
    questions: List[Dict],
    config: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict]:
    """Run inference on a list of questions.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        questions: List of question dictionaries.
        config: Configuration object.
        device: Device to run inference on.

    Returns:
        List of prediction results.
    """
    model = model.to(device)
    results = []

    logger.info(f"Running inference on {len(questions)} questions")

    with torch.no_grad():
        for i, question_data in enumerate(questions):
            # Format the question
            formatted_q = format_question(
                question_data.get("question", ""),
                question_data.get("choices", []),
                question_data.get("subject", "")
            )

            # Tokenize
            inputs = tokenizer(
                formatted_q,
                return_tensors="pt",
                max_length=config.get("model.max_length", 256),
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model outputs
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # Get logits for answer tokens
            logits = outputs["logits"][:, -1, :]  # Last token logits

            # Get token IDs for A, B, C, D
            choice_tokens = tokenizer(["A", "B", "C", "D"], add_special_tokens=False)
            choice_token_ids = [tokens[0] for tokens in choice_tokens["input_ids"]]

            # Get probabilities for each choice
            choice_logits = logits[:, choice_token_ids]
            probabilities = torch.softmax(choice_logits, dim=-1).cpu().numpy()[0]

            # Get prediction
            predicted_idx = int(probabilities.argmax())
            predicted_choice = ["A", "B", "C", "D"][predicted_idx]

            result = {
                "question": question_data.get("question", ""),
                "subject": question_data.get("subject", ""),
                "predicted_answer": predicted_choice,
                "confidence": float(probabilities[predicted_idx]),
                "probabilities": {
                    choice: float(prob)
                    for choice, prob in zip(["A", "B", "C", "D"], probabilities)
                },
            }

            # Include correct answer if available
            if "answer" in question_data:
                result["correct_answer"] = question_data["answer"]
                result["is_correct"] = predicted_choice == question_data["answer"]

            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")

    return results


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained adaptive curriculum model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs/models/final_model",
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input JSON file with questions (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/predictions.json",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer, config = load_model(args.model_path, args.config)

    # Load or create sample questions
    if args.input:
        logger.info(f"Loading questions from {args.input}")
        with open(args.input, "r") as f:
            questions = json.load(f)
    else:
        # Create sample questions for demonstration
        logger.info("Using sample questions (no input file provided)")
        questions = [
            {
                "question": "What is the primary function of mitochondria in eukaryotic cells?",
                "choices": [
                    "Protein synthesis",
                    "Energy production through ATP synthesis",
                    "DNA replication",
                    "Lipid storage"
                ],
                "subject": "biology",
                "answer": "B"
            },
            {
                "question": "Which of the following best describes the concept of opportunity cost in economics?",
                "choices": [
                    "The monetary cost of a purchase",
                    "The value of the next best alternative foregone",
                    "The total cost including taxes",
                    "The difference between price and value"
                ],
                "subject": "economics",
                "answer": "B"
            },
            {
                "question": "What philosophical movement emphasized individual freedom and authenticity in the mid-20th century?",
                "choices": [
                    "Rationalism",
                    "Empiricism",
                    "Existentialism",
                    "Pragmatism"
                ],
                "subject": "philosophy",
                "answer": "C"
            }
        ]

    # Run predictions
    results = predict(model, tokenizer, questions, config, device=args.device)

    # Calculate accuracy if answers provided
    correct = sum(1 for r in results if r.get("is_correct", False))
    total = len(results)
    if any("is_correct" in r for r in results):
        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Predictions saved to {output_path}")

    # Print sample results
    logger.info("\nSample predictions:")
    for i, result in enumerate(results[:3]):
        logger.info(f"\nQuestion {i+1}: {result['question'][:80]}...")
        logger.info(f"  Predicted: {result['predicted_answer']} (confidence: {result['confidence']:.3f})")
        if "correct_answer" in result:
            logger.info(f"  Correct: {result['correct_answer']} ({'✓' if result['is_correct'] else '✗'})")


if __name__ == "__main__":
    main()
