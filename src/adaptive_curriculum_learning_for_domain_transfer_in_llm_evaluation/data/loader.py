"""Data loading utilities for MMLU dataset."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class MMluDataLoader:
    """Data loader for the MMLU dataset with curriculum learning support.

    This class handles loading, preprocessing, and organizing MMLU data for
    curriculum learning experiments. It provides functionality to split data
    by domains, create curriculum orderings, and prepare data for training.
    """

    def __init__(
        self,
        dataset_name: str = "cais/mmlu",
        cache_dir: Optional[str] = None,
        max_samples_per_domain: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize the MMLU data loader.

        Args:
            dataset_name: Name of the MMLU dataset on Hugging Face.
            cache_dir: Directory to cache downloaded datasets.
            max_samples_per_domain: Maximum samples to load per domain. None for all.
            random_seed: Random seed for reproducibility.
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.max_samples_per_domain = max_samples_per_domain
        self.random_seed = random_seed
        self._dataset: Optional[DatasetDict] = None
        self._domain_mapping = self._get_domain_mapping()

    def _get_domain_mapping(self) -> Dict[str, str]:
        """Get mapping from MMLU subjects to broader domains.

        Returns:
            Dictionary mapping subject names to domain categories.
        """
        return {
            # STEM domains
            "abstract_algebra": "STEM",
            "anatomy": "STEM",
            "astronomy": "STEM",
            "college_biology": "STEM",
            "college_chemistry": "STEM",
            "college_computer_science": "STEM",
            "college_mathematics": "STEM",
            "college_physics": "STEM",
            "computer_security": "STEM",
            "conceptual_physics": "STEM",
            "elementary_mathematics": "STEM",
            "high_school_biology": "STEM",
            "high_school_chemistry": "STEM",
            "high_school_computer_science": "STEM",
            "high_school_mathematics": "STEM",
            "high_school_physics": "STEM",
            "high_school_statistics": "STEM",
            "machine_learning": "STEM",

            # Social Sciences
            "econometrics": "Social Sciences",
            "high_school_geography": "Social Sciences",
            "high_school_government_and_politics": "Social Sciences",
            "high_school_macroeconomics": "Social Sciences",
            "high_school_microeconomics": "Social Sciences",
            "high_school_psychology": "Social Sciences",
            "human_sexuality": "Social Sciences",
            "professional_psychology": "Social Sciences",
            "public_relations": "Social Sciences",
            "security_studies": "Social Sciences",
            "sociology": "Social Sciences",
            "us_foreign_policy": "Social Sciences",

            # Humanities
            "formal_logic": "Humanities",
            "high_school_european_history": "Humanities",
            "high_school_us_history": "Humanities",
            "high_school_world_history": "Humanities",
            "logical_fallacies": "Humanities",
            "moral_disputes": "Humanities",
            "moral_scenarios": "Humanities",
            "philosophy": "Humanities",
            "prehistory": "Humanities",
            "professional_law": "Humanities",
            "world_religions": "Humanities",

            # Other
            "business_ethics": "Other",
            "clinical_knowledge": "Other",
            "college_medicine": "Other",
            "global_facts": "Other",
            "human_aging": "Other",
            "management": "Other",
            "marketing": "Other",
            "medical_genetics": "Other",
            "miscellaneous": "Other",
            "nutrition": "Other",
            "professional_accounting": "Other",
            "professional_medicine": "Other",
            "virology": "Other",
        }

    def load_dataset(self) -> DatasetDict:
        """Load the MMLU dataset.

        Returns:
            Loaded and preprocessed dataset.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            logger.info(f"Loading dataset {self.dataset_name}")
            import time
            start_time = time.time()
            self._dataset = load_dataset(
                self.dataset_name,
                "all",
                cache_dir=self.cache_dir,
            )
            load_time = time.time() - start_time
            logger.debug(f"Dataset loaded in {load_time:.2f} seconds")

            # Process each split
            processed_dataset = {}
            for split_name, split_data in self._dataset.items():
                logger.debug(f"Processing split '{split_name}' with {len(split_data)} samples")
                processed_split = self._process_split(split_data, split_name)
                processed_dataset[split_name] = processed_split

            self._dataset = DatasetDict(processed_dataset)
            logger.info(f"Successfully loaded dataset with {len(self._dataset)} splits")
            return self._dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}")

    def _process_split(self, split_data: Dataset, split_name: str) -> Dataset:
        """Process a single split of the dataset.

        Args:
            split_data: Raw dataset split.
            split_name: Name of the split.

        Returns:
            Processed dataset split.
        """
        logger.info(f"Processing {split_name} split with {len(split_data)} samples")

        # Convert to pandas for easier processing
        df = split_data.to_pandas()

        # Add domain information
        df['domain'] = df['subject'].map(self._domain_mapping)
        df['domain'] = df['domain'].fillna('Other')

        # Limit samples per domain if specified
        if self.max_samples_per_domain is not None:
            df = df.groupby('subject').apply(
                lambda x: x.sample(
                    min(len(x), self.max_samples_per_domain),
                    random_state=self.random_seed
                )
            ).reset_index(drop=True)

        # Format questions and choices
        df['formatted_question'] = df.apply(self._format_question, axis=1)
        df['correct_answer_text'] = df.apply(self._get_correct_answer, axis=1)

        # Convert back to dataset
        processed_dataset = Dataset.from_pandas(df, preserve_index=False)

        logger.info(f"Processed {split_name}: {len(processed_dataset)} samples")
        return processed_dataset

    def _format_question(self, row: pd.Series) -> str:
        """Format a question with its choices.

        Args:
            row: DataFrame row containing question data.

        Returns:
            Formatted question string.
        """
        question = row['question']
        choices = row['choices']

        formatted = f"Question: {question}\\n\\nChoices:\\n"
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            formatted += f"{letter}. {choice}\\n"

        return formatted.strip()

    def _get_correct_answer(self, row: pd.Series) -> str:
        """Get the correct answer text for a question.

        Args:
            row: DataFrame row containing question data.

        Returns:
            Correct answer text.
        """
        answer_idx = row['answer']
        choices = row['choices']

        if 0 <= answer_idx < len(choices):
            return choices[answer_idx]
        else:
            logger.warning(f"Invalid answer index {answer_idx} for question: {row['question'][:50]}...")
            return choices[0] if choices else ""

    def get_domain_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about domains in the dataset.

        Returns:
            Dictionary with domain statistics for each split.

        Raises:
            RuntimeError: If dataset is not loaded.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        statistics = {}
        for split_name, split_data in self._dataset.items():
            df = split_data.to_pandas()
            domain_counts = df.groupby(['domain', 'subject']).size().reset_index(name='count')

            split_stats = {}
            for domain in df['domain'].unique():
                domain_data = domain_counts[domain_counts['domain'] == domain]
                split_stats[domain] = {
                    'total_samples': domain_data['count'].sum(),
                    'subjects': domain_data['subject'].tolist(),
                    'samples_per_subject': dict(zip(domain_data['subject'], domain_data['count']))
                }

            statistics[split_name] = split_stats

        return statistics

    def create_domain_splits(
        self,
        source_domains: List[str],
        target_domains: List[str],
        validation_split: float = 0.1,
        test_split: float = 0.2,
    ) -> Dict[str, Dataset]:
        """Create domain-specific splits for transfer learning experiments.

        Args:
            source_domains: List of source domain names.
            target_domains: List of target domain names.
            validation_split: Proportion of data for validation.
            test_split: Proportion of data for testing.

        Returns:
            Dictionary with domain-specific dataset splits.

        Raises:
            RuntimeError: If dataset is not loaded.
            ValueError: If domain names are invalid.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        # Use the 'all' split if available, otherwise combine train/dev/test
        if 'all' in self._dataset:
            full_data = self._dataset['all']
        else:
            # Combine available splits
            splits_to_combine = []
            for split_name in ['train', 'dev', 'test', 'validation']:
                if split_name in self._dataset:
                    splits_to_combine.append(self._dataset[split_name])

            if not splits_to_combine:
                raise RuntimeError("No valid splits found in dataset")

            # Combine datasets
            from datasets import concatenate_datasets
            full_data = concatenate_datasets(splits_to_combine)

        df = full_data.to_pandas()

        # Validate domain names
        available_domains = set(df['domain'].unique())
        invalid_source = set(source_domains) - available_domains
        invalid_target = set(target_domains) - available_domains

        if invalid_source:
            raise ValueError(f"Invalid source domains: {invalid_source}")
        if invalid_target:
            raise ValueError(f"Invalid target domains: {invalid_target}")

        # Create splits
        domain_splits = {}

        # Source domain data
        source_data = df[df['domain'].isin(source_domains)]
        if len(source_data) > 0:
            source_train, source_temp = train_test_split(
                source_data,
                test_size=validation_split + test_split,
                random_state=self.random_seed,
                stratify=source_data['subject']
            )
            source_val, source_test = train_test_split(
                source_temp,
                test_size=test_split / (validation_split + test_split),
                random_state=self.random_seed,
                stratify=source_temp['subject']
            )

            domain_splits['source_train'] = Dataset.from_pandas(source_train, preserve_index=False)
            domain_splits['source_val'] = Dataset.from_pandas(source_val, preserve_index=False)
            domain_splits['source_test'] = Dataset.from_pandas(source_test, preserve_index=False)

        # Target domain data
        target_data = df[df['domain'].isin(target_domains)]
        if len(target_data) > 0:
            target_train, target_temp = train_test_split(
                target_data,
                test_size=validation_split + test_split,
                random_state=self.random_seed,
                stratify=target_data['subject']
            )
            target_val, target_test = train_test_split(
                target_temp,
                test_size=test_split / (validation_split + test_split),
                random_state=self.random_seed,
                stratify=target_temp['subject']
            )

            domain_splits['target_train'] = Dataset.from_pandas(target_train, preserve_index=False)
            domain_splits['target_val'] = Dataset.from_pandas(target_val, preserve_index=False)
            domain_splits['target_test'] = Dataset.from_pandas(target_test, preserve_index=False)

        logger.info(f"Created domain splits with {len(domain_splits)} datasets")
        return domain_splits

    def prepare_for_training(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> DatasetDict:
        """Prepare dataset for training by tokenizing and formatting.

        Args:
            tokenizer: Tokenizer to use for encoding.
            max_length: Maximum sequence length.

        Returns:
            Tokenized dataset ready for training.

        Raises:
            RuntimeError: If dataset is not loaded.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        def tokenize_function(examples):
            """Tokenize examples for training."""
            # Prepare input-output pairs
            inputs = []
            targets = []

            for i in range(len(examples['formatted_question'])):
                question = examples['formatted_question'][i]
                answer = examples['correct_answer_text'][i]

                # Format as instruction-following task
                input_text = f"Please answer the following multiple choice question:\\n\\n{question}\\n\\nAnswer:"
                target_text = f" {answer}"

                inputs.append(input_text)
                targets.append(target_text)

            # Tokenize inputs and targets
            model_inputs = tokenizer(
                inputs,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            # Tokenize targets
            targets_tokenized = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            model_inputs['labels'] = targets_tokenized['input_ids']
            return model_inputs

        # Apply tokenization
        tokenized_dataset = self._dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self._dataset[list(self._dataset.keys())[0]].column_names,
        )

        logger.info("Successfully tokenized dataset for training")
        return tokenized_dataset

    def get_subjects_by_domain(self, domain: str) -> List[str]:
        """Get list of subjects belonging to a domain.

        Args:
            domain: Domain name.

        Returns:
            List of subject names in the domain.
        """
        return [
            subject for subject, mapped_domain in self._domain_mapping.items()
            if mapped_domain == domain
        ]

    def get_dataset(self) -> Optional[DatasetDict]:
        """Get the loaded dataset.

        Returns:
            The loaded dataset or None if not loaded.
        """
        return self._dataset