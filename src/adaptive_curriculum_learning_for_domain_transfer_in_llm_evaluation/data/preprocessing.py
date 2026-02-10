"""Data preprocessing utilities for curriculum learning."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DifficultyEstimator:
    """Estimates difficulty of questions for curriculum learning.

    This class provides multiple methods for estimating question difficulty,
    including entropy-based measures, model confidence, and loss-based approaches.
    """

    def __init__(
        self,
        method: str = "entropy",
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        """Initialize difficulty estimator.

        Args:
            method: Difficulty estimation method ('entropy', 'confidence', 'loss').
            model: Pre-trained model for confidence/loss-based estimation.
            tokenizer: Tokenizer for the model.
        """
        self.method = method
        self.model = model
        self.tokenizer = tokenizer

        if method in ['confidence', 'loss'] and (model is None or tokenizer is None):
            raise ValueError(f"Method '{method}' requires both model and tokenizer")

    def estimate_difficulty(self, dataset: Dataset) -> np.ndarray:
        """Estimate difficulty scores for all questions in dataset.

        Args:
            dataset: Dataset containing questions to score.

        Returns:
            Array of difficulty scores (higher = more difficult).

        Raises:
            ValueError: If estimation method is invalid.
        """
        logger.info(f"Estimating difficulty using {self.method} method")

        if self.method == "entropy":
            return self._estimate_entropy_difficulty(dataset)
        elif self.method == "confidence":
            return self._estimate_confidence_difficulty(dataset)
        elif self.method == "loss":
            return self._estimate_loss_difficulty(dataset)
        else:
            raise ValueError(f"Unknown difficulty estimation method: {self.method}")

    def _estimate_entropy_difficulty(self, dataset: Dataset) -> np.ndarray:
        """Estimate difficulty based on answer choice distribution entropy.

        Questions with more uniform answer distributions are considered harder.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Array of entropy-based difficulty scores.
        """
        difficulties = []

        # Group by subject to get answer distributions
        df = dataset.to_pandas()
        subjects = df['subject'].unique()

        for subject in subjects:
            subject_data = df[df['subject'] == subject]
            answer_counts = subject_data['answer'].value_counts(normalize=True)

            # Calculate entropy of answer distribution
            entropy = -np.sum(answer_counts * np.log2(answer_counts + 1e-8))

            # Assign same entropy to all questions in subject
            subject_difficulty = entropy
            difficulties.extend([subject_difficulty] * len(subject_data))

        # Normalize to [0, 1] range
        difficulties = np.array(difficulties)
        if len(np.unique(difficulties)) > 1:
            difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min())

        logger.info(f"Computed entropy-based difficulties: mean={difficulties.mean():.3f}, std={difficulties.std():.3f}")
        return difficulties

    def _estimate_confidence_difficulty(self, dataset: Dataset) -> np.ndarray:
        """Estimate difficulty based on model confidence.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Array of confidence-based difficulty scores.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        difficulties = []

        with torch.no_grad():
            for i in range(len(dataset)):
                example = dataset[i]
                question = example['formatted_question']
                choices = example['choices']

                # Get model predictions for each choice
                choice_probs = []
                for j, choice in enumerate(choices):
                    # Format input
                    input_text = f"{question}\\n\\nAnswer: {choice}"
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(device)

                    # Get model output
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state

                    # Compute probability (simplified)
                    prob = F.softmax(logits, dim=-1).max().item()
                    choice_probs.append(prob)

                # Difficulty is inverse of confidence (1 - max_prob)
                max_confidence = max(choice_probs)
                difficulty = 1.0 - max_confidence
                difficulties.append(difficulty)

        difficulties = np.array(difficulties)
        logger.info(f"Computed confidence-based difficulties: mean={difficulties.mean():.3f}, std={difficulties.std():.3f}")
        return difficulties

    def _estimate_loss_difficulty(self, dataset: Dataset) -> np.ndarray:
        """Estimate difficulty based on model loss.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Array of loss-based difficulty scores.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        difficulties = []

        with torch.no_grad():
            for i in range(len(dataset)):
                example = dataset[i]
                question = example['formatted_question']
                correct_answer = example['correct_answer_text']

                # Format input
                input_text = f"{question}\\n\\nAnswer:"
                target_text = f" {correct_answer}"

                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)

                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)

                # Compute loss
                outputs = self.model(**inputs, labels=targets['input_ids'])
                loss = outputs.loss.item()
                difficulties.append(loss)

        # Normalize losses
        difficulties = np.array(difficulties)
        if difficulties.std() > 0:
            difficulties = (difficulties - difficulties.mean()) / difficulties.std()
            difficulties = torch.sigmoid(torch.tensor(difficulties)).numpy()

        logger.info(f"Computed loss-based difficulties: mean={difficulties.mean():.3f}, std={difficulties.std():.3f}")
        return difficulties


class DomainSimilarityComputer:
    """Computes domain similarity for curriculum learning.

    This class provides methods to compute similarity between different domains
    using various approaches including sentence embeddings and keyword analysis.
    """

    def __init__(
        self,
        method: str = "sentence_embeddings",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize domain similarity computer.

        Args:
            method: Similarity computation method ('sentence_embeddings', 'domain_keywords').
            embedding_model: Name or path of sentence embedding model.
            config: Optional configuration dictionary.
        """
        self.method = method
        self.embedding_model_name = embedding_model
        self.config = config or {}

        if method == "sentence_embeddings":
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

        self._domain_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._question_embeddings: Optional[np.ndarray] = None

    def compute_domain_similarity(
        self,
        dataset: Dataset,
        domains: List[str],
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between domains.

        Args:
            dataset: Dataset containing questions from multiple domains.
            domains: List of domain names to compare.

        Returns:
            Symmetric similarity matrix of shape (n_domains, n_domains).
        """
        logger.info(f"Computing domain similarity using {self.method} method")

        if self.method == "sentence_embeddings":
            return self._compute_embedding_similarity(dataset, domains)
        elif self.method == "domain_keywords":
            return self._compute_keyword_similarity(dataset, domains)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")

    def _compute_embedding_similarity(
        self,
        dataset: Dataset,
        domains: List[str],
    ) -> np.ndarray:
        """Compute similarity using sentence embeddings.

        Args:
            dataset: Dataset to analyze.
            domains: List of domain names.

        Returns:
            Embedding-based similarity matrix.
        """
        # Create domain representations by averaging question embeddings
        df = dataset.to_pandas()
        domain_embeddings = {}

        for domain in domains:
            domain_questions = df[df['domain'] == domain]['formatted_question'].tolist()

            if not domain_questions:
                logger.warning(f"No questions found for domain: {domain}")
                continue

            # Sample questions if too many (for efficiency)
            max_questions = self.config.get('curriculum', {}).get('max_domain_questions', 100)
            if len(domain_questions) > max_questions:
                import random
                domain_questions = random.sample(domain_questions, max_questions)

            # Compute embeddings
            embeddings = self.embedding_model.encode(domain_questions, show_progress_bar=False)
            domain_embedding = np.mean(embeddings, axis=0)
            domain_embeddings[domain] = domain_embedding

        self._domain_embeddings = domain_embeddings

        # Compute pairwise similarities
        n_domains = len(domains)
        similarity_matrix = np.zeros((n_domains, n_domains))

        for i, domain_i in enumerate(domains):
            for j, domain_j in enumerate(domains):
                if domain_i in domain_embeddings and domain_j in domain_embeddings:
                    emb_i = domain_embeddings[domain_i].reshape(1, -1)
                    emb_j = domain_embeddings[domain_j].reshape(1, -1)
                    similarity = cosine_similarity(emb_i, emb_j)[0, 0]
                    similarity_matrix[i, j] = similarity
                else:
                    similarity_matrix[i, j] = 0.0

        logger.info(f"Computed embedding similarity matrix for {n_domains} domains")
        return similarity_matrix

    def _compute_keyword_similarity(
        self,
        dataset: Dataset,
        domains: List[str],
    ) -> np.ndarray:
        """Compute similarity based on domain keywords.

        Args:
            dataset: Dataset to analyze.
            domains: List of domain names.

        Returns:
            Keyword-based similarity matrix.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Extract domain-specific text
        df = dataset.to_pandas()
        domain_texts = {}

        for domain in domains:
            domain_questions = df[df['domain'] == domain]['formatted_question'].tolist()
            domain_text = " ".join(domain_questions)
            domain_texts[domain] = domain_text

        if not domain_texts:
            logger.warning("No domain texts found")
            return np.eye(len(domains))

        # Compute TF-IDF features
        max_features = self.config.get('curriculum', {}).get('max_tfidf_features', 1000)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

        domain_names_ordered = list(domain_texts.keys())
        domain_texts_list = [domain_texts[domain] for domain in domain_names_ordered]

        tfidf_matrix = vectorizer.fit_transform(domain_texts_list)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Reorder to match input domain order
        reorder_indices = [domain_names_ordered.index(domain) for domain in domains if domain in domain_names_ordered]
        similarity_matrix = similarity_matrix[np.ix_(reorder_indices, reorder_indices)]

        logger.info(f"Computed keyword similarity matrix for {len(domains)} domains")
        return similarity_matrix

    def compute_question_similarity(
        self,
        questions: List[str],
        reference_questions: List[str],
    ) -> np.ndarray:
        """Compute similarity between questions and reference set.

        Args:
            questions: List of questions to score.
            reference_questions: Reference set of questions.

        Returns:
            Array of similarity scores for each question.
        """
        if self.method != "sentence_embeddings":
            raise ValueError("Question similarity only supported with sentence embeddings")

        # Compute embeddings
        question_embeddings = self.embedding_model.encode(questions, show_progress_bar=False)
        reference_embeddings = self.embedding_model.encode(reference_questions, show_progress_bar=False)

        # Compute mean reference embedding
        mean_reference = np.mean(reference_embeddings, axis=0)

        # Compute similarities
        similarities = cosine_similarity(
            question_embeddings,
            mean_reference.reshape(1, -1)
        ).flatten()

        return similarities

    def get_domain_embedding(self, domain: str) -> Optional[np.ndarray]:
        """Get embedding for a specific domain.

        Args:
            domain: Domain name.

        Returns:
            Domain embedding or None if not available.
        """
        if self._domain_embeddings is None:
            return None
        return self._domain_embeddings.get(domain)

    def cluster_questions(
        self,
        questions: List[str],
        n_clusters: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster questions by similarity.

        Args:
            questions: List of questions to cluster.
            n_clusters: Number of clusters.

        Returns:
            Tuple of (cluster_labels, cluster_centers).
        """
        if self.method != "sentence_embeddings":
            raise ValueError("Question clustering only supported with sentence embeddings")

        # Compute embeddings
        embeddings = self.embedding_model.encode(questions, show_progress_bar=False)

        # Reduce dimensionality for visualization
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        logger.info(f"Clustered {len(questions)} questions into {n_clusters} clusters")
        return cluster_labels, kmeans.cluster_centers_