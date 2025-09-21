"""
Evaluation Metrics for FinGAT Vietnam Project

This module implements evaluation metrics for ranking and classification tasks
as described in the FinGAT paper.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import logging

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at K (MRR@K).

    MRR@K measures the average reciprocal rank of the first relevant item
    within the top-k predictions.

    Args:
        y_true: True binary labels (1 for positive, 0 for negative)
        y_scores: Predicted scores for ranking
        k: Number of top predictions to consider

    Returns:
        MRR@K score
    """
    if len(y_true) == 0:
        return 0.0

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]

    # Get top-k predictions
    top_k_indices = sorted_indices[:k]

    # Check if any of the top-k predictions are positive
    top_k_labels = y_true[top_k_indices]

    # Find the rank of the first positive prediction (1-indexed)
    positive_ranks = np.where(top_k_labels == 1)[0]

    if len(positive_ranks) > 0:
        # Reciprocal of the rank of the first positive prediction
        first_positive_rank = positive_ranks[0] + 1  # Convert to 1-indexed
        return 1.0 / first_positive_rank
    else:
        return 0.0


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Precision at K (P@K).

    P@K measures the proportion of relevant items among the top-k predictions.

    Args:
        y_true: True binary labels (1 for positive, 0 for negative)
        y_scores: Predicted scores for ranking
        k: Number of top predictions to consider

    Returns:
        Precision@K score
    """
    if len(y_true) == 0 or k == 0:
        return 0.0

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]

    # Get top-k predictions
    top_k_indices = sorted_indices[:min(k, len(sorted_indices))]

    # Calculate precision
    if len(top_k_indices) == 0:
        return 0.0

    top_k_labels = y_true[top_k_indices]
    precision = np.sum(top_k_labels) / len(top_k_labels)

    return precision


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Recall at K (R@K).

    R@K measures the proportion of relevant items that are retrieved in top-k.

    Args:
        y_true: True binary labels (1 for positive, 0 for negative)
        y_scores: Predicted scores for ranking
        k: Number of top predictions to consider

    Returns:
        Recall@K score
    """
    if len(y_true) == 0 or k == 0:
        return 0.0

    # Total number of relevant items
    total_relevant = np.sum(y_true)

    if total_relevant == 0:
        return 0.0

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]

    # Get top-k predictions
    top_k_indices = sorted_indices[:min(k, len(sorted_indices))]

    # Calculate recall
    if len(top_k_indices) == 0:
        return 0.0

    top_k_labels = y_true[top_k_indices]
    relevant_retrieved = np.sum(top_k_labels)
    recall = relevant_retrieved / total_relevant

    return recall


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

    NDCG@K measures the quality of ranking considering the position of relevant items.

    Args:
        y_true: True relevance scores (can be binary or continuous)
        y_scores: Predicted scores for ranking
        k: Number of top predictions to consider

    Returns:
        NDCG@K score
    """
    if len(y_true) == 0 or k == 0:
        return 0.0

    def dcg(scores: np.ndarray) -> float:
        """Calculate Discounted Cumulative Gain."""
        scores = np.asarray(scores)
        if scores.size == 0:
            return 0.0

        # DCG formula: sum(2^rel_i - 1) / log2(i + 1)
        dcg_val = scores[0]
        for i in range(1, min(len(scores), k)):
            dcg_val += scores[i] / np.log2(i + 2)

        return dcg_val

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]

    # Get top-k predictions
    top_k_indices = sorted_indices[:min(k, len(sorted_indices))]
    top_k_true = y_true[top_k_indices]

    # Calculate DCG for predictions
    predicted_dcg = dcg(top_k_true)

    # Calculate ideal DCG (best possible ranking)
    ideal_sorted_true = np.sort(y_true)[::-1]
    ideal_dcg = dcg(ideal_sorted_true)

    if ideal_dcg == 0:
        return 0.0

    return predicted_dcg / ideal_dcg


def calculate_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                            k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate comprehensive ranking metrics.

    Args:
        y_true: True binary labels or relevance scores
        y_scores: Predicted ranking scores
        k_values: List of k values to evaluate

    Returns:
        Dictionary containing all ranking metrics
    """
    metrics = {}

    for k in k_values:
        metrics[f'mrr@{k}'] = mean_reciprocal_rank(y_true, y_scores, k)
        metrics[f'precision@{k}'] = precision_at_k(y_true, y_scores, k)
        metrics[f'recall@{k}'] = recall_at_k(y_true, y_scores, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(y_true, y_scores, k)

    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_probs: Optional[np.ndarray] = None,
                                   average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_probs: Predicted class probabilities (optional)
        average: Averaging strategy for multi-class metrics

    Returns:
        Dictionary containing all classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    for i, label in enumerate(unique_labels):
        if i < len(precision):
            metrics[f'precision_class_{label}'] = precision[i]
            metrics[f'recall_class_{label}'] = recall[i]
            metrics[f'f1_class_{label}'] = f1[i]
            metrics[f'support_class_{label}'] = support[i]

    return metrics


def stock_return_accuracy(predicted_returns: np.ndarray, actual_returns: np.ndarray, 
                         threshold: float = 0.0) -> float:
    """
    Calculate stock return direction accuracy.

    Args:
        predicted_returns: Predicted stock returns
        actual_returns: Actual stock returns
        threshold: Threshold for determining direction (default: 0.0)

    Returns:
        Direction accuracy (percentage of correct predictions)
    """
    predicted_direction = (predicted_returns > threshold).astype(int)
    actual_direction = (actual_returns > threshold).astype(int)

    return accuracy_score(actual_direction, predicted_direction)


def sharpe_ratio_metric(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for portfolio returns.

    Args:
        returns: Portfolio returns
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def information_ratio_metric(portfolio_returns: np.ndarray, 
                           benchmark_returns: np.ndarray) -> float:
    """
    Calculate Information Ratio.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")

    if len(portfolio_returns) == 0:
        return 0.0

    excess_returns = portfolio_returns - benchmark_returns

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.

    Args:
        returns: Returns series

    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max

    return np.min(drawdown)


class FinGATEvaluator:
    """
    Comprehensive evaluator for FinGAT model performance.
    """

    def __init__(self, k_values: List[int] = [5, 10, 20, 30]):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for ranking metrics
        """
        self.k_values = k_values
        self.metrics_history = []

    def evaluate_batch(self, outputs: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate a single batch of predictions.

        Args:
            outputs: Model outputs containing predictions and scores
            targets: Target labels and values

        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert tensors to numpy
        if isinstance(outputs['movement_probs'], torch.Tensor):
            movement_probs = outputs['movement_probs'].detach().cpu().numpy()
            movement_preds = np.argmax(movement_probs, axis=1)
        else:
            movement_probs = outputs['movement_probs']
            movement_preds = np.argmax(movement_probs, axis=1)

        if isinstance(outputs['ranking_scores'], torch.Tensor):
            ranking_scores = outputs['ranking_scores'].detach().cpu().numpy().flatten()
        else:
            ranking_scores = outputs['ranking_scores'].flatten()

        if isinstance(targets['movement_labels'], torch.Tensor):
            movement_labels = targets['movement_labels'].detach().cpu().numpy()
        else:
            movement_labels = targets['movement_labels']

        if isinstance(targets['returns'], torch.Tensor):
            returns = targets['returns'].detach().cpu().numpy()
        else:
            returns = targets['returns']

        # Classification metrics
        classification_metrics = calculate_classification_metrics(
            movement_labels, movement_preds
        )

        # Ranking metrics (using positive returns as relevant items)
        binary_labels = (returns > 0).astype(int)
        ranking_metrics = calculate_ranking_metrics(
            binary_labels, ranking_scores, self.k_values
        )

        # Return accuracy
        return_accuracy = stock_return_accuracy(ranking_scores, returns)

        # Combine all metrics
        metrics = {
            **classification_metrics,
            **ranking_metrics,
            'return_accuracy': return_accuracy
        }

        return metrics

    def evaluate_epoch(self, all_outputs: List[Dict], all_targets: List[Dict]) -> Dict[str, float]:
        """
        Evaluate performance over an entire epoch.

        Args:
            all_outputs: List of batch outputs
            all_targets: List of batch targets

        Returns:
            Dictionary containing epoch-level metrics
        """
        # Concatenate all predictions
        all_movement_preds = []
        all_movement_labels = []
        all_ranking_scores = []
        all_returns = []

        for outputs, targets in zip(all_outputs, all_targets):
            # Movement predictions
            if isinstance(outputs['movement_probs'], torch.Tensor):
                movement_probs = outputs['movement_probs'].detach().cpu().numpy()
            else:
                movement_probs = outputs['movement_probs']

            movement_preds = np.argmax(movement_probs, axis=1)
            all_movement_preds.extend(movement_preds)

            # Labels and scores
            if isinstance(targets['movement_labels'], torch.Tensor):
                movement_labels = targets['movement_labels'].detach().cpu().numpy()
            else:
                movement_labels = targets['movement_labels']
            all_movement_labels.extend(movement_labels)

            if isinstance(outputs['ranking_scores'], torch.Tensor):
                ranking_scores = outputs['ranking_scores'].detach().cpu().numpy().flatten()
            else:
                ranking_scores = outputs['ranking_scores'].flatten()
            all_ranking_scores.extend(ranking_scores)

            if isinstance(targets['returns'], torch.Tensor):
                returns = targets['returns'].detach().cpu().numpy()
            else:
                returns = targets['returns']
            all_returns.extend(returns)

        # Convert to numpy arrays
        all_movement_preds = np.array(all_movement_preds)
        all_movement_labels = np.array(all_movement_labels)
        all_ranking_scores = np.array(all_ranking_scores)
        all_returns = np.array(all_returns)

        # Calculate comprehensive metrics
        classification_metrics = calculate_classification_metrics(
            all_movement_labels, all_movement_preds
        )

        # Ranking metrics
        binary_labels = (all_returns > 0).astype(int)
        ranking_metrics = calculate_ranking_metrics(
            binary_labels, all_ranking_scores, self.k_values
        )

        # Additional financial metrics
        return_accuracy = stock_return_accuracy(all_ranking_scores, all_returns)

        # Portfolio performance (if applicable)
        if len(all_returns) > 0:
            sharpe = sharpe_ratio_metric(all_returns)
            max_dd = maximum_drawdown(all_returns)
        else:
            sharpe = 0.0
            max_dd = 0.0

        # Combine all metrics
        epoch_metrics = {
            **classification_metrics,
            **ranking_metrics,
            'return_accuracy': return_accuracy,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }

        # Store in history
        self.metrics_history.append(epoch_metrics)

        return epoch_metrics

    def get_best_metrics(self, metric_name: str = 'mrr@10') -> Tuple[Dict, int]:
        """
        Get the best metrics based on a specific metric.

        Args:
            metric_name: Name of metric to optimize for

        Returns:
            Tuple of (best_metrics_dict, best_epoch)
        """
        if not self.metrics_history:
            return {}, -1

        best_value = -1
        best_epoch = 0
        best_metrics = self.metrics_history[0]

        for i, metrics in enumerate(self.metrics_history):
            if metrics.get(metric_name, 0) > best_value:
                best_value = metrics[metric_name]
                best_epoch = i
                best_metrics = metrics

        return best_metrics, best_epoch


# Example usage
if __name__ == "__main__":
    # Example evaluation
    evaluator = FinGATEvaluator(k_values=[5, 10, 20])

    # Dummy data for testing
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    # Test metrics
    ranking_metrics = calculate_ranking_metrics(y_true, y_scores)
    print(f"Ranking metrics: {ranking_metrics}")

    print("FinGAT Evaluator module loaded successfully")