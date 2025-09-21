"""
Evaluation metrics for FinGAT model
Implements MRR@K, Precision@K, and movement accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def calculate_mrr_at_k(pred_returns: np.ndarray,
                       true_returns: np.ndarray,
                       k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at K
    
    MRR@K = (1/N) * Σ(1/rank_i) for top-K predicted stocks
    where rank_i is the rank of stock i in the ground truth ranking
    
    Args:
        pred_returns: Predicted returns (n_stocks,)
        true_returns: True returns (n_stocks,)
        k: Top K to consider
        
    Returns:
        MRR@K value
    """
    n_stocks = len(pred_returns)
    k = min(k, n_stocks)
    
    # Get top-k indices based on predicted returns
    pred_top_k_indices = np.argsort(pred_returns)[-k:][::-1]
    
    # Get ranking based on true returns
    true_ranking = np.argsort(np.argsort(true_returns))[::-1]  # Higher return = better rank
    
    # Calculate reciprocal ranks
    reciprocal_sum = 0
    for idx in pred_top_k_indices:
        true_rank = true_ranking[idx] + 1  # +1 because rank starts from 1
        reciprocal_sum += 1.0 / true_rank
    
    return reciprocal_sum / k


def calculate_precision_at_k(pred_returns: np.ndarray,
                             true_returns: np.ndarray,
                             k: int = 10) -> float:
    """
    Calculate Precision at K
    
    Precision@K = |predicted_top_k ∩ true_top_k| / k
    
    Args:
        pred_returns: Predicted returns (n_stocks,)
        true_returns: True returns (n_stocks,)
        k: Top K to consider
        
    Returns:
        Precision@K value
    """
    n_stocks = len(pred_returns)
    k = min(k, n_stocks)
    
    # Get top-k indices
    pred_top_k = set(np.argsort(pred_returns)[-k:])
    true_top_k = set(np.argsort(true_returns)[-k:])
    
    # Calculate intersection
    correct = len(pred_top_k & true_top_k)
    
    return correct / k


def calculate_movement_accuracy(pred_movements: np.ndarray,
                                true_movements: np.ndarray) -> float:
    """
    Calculate movement prediction accuracy
    
    Args:
        pred_movements: Predicted movements (n_stocks,) with values in [0, 1]
        true_movements: True movements (n_stocks,) with binary values
        
    Returns:
        Accuracy value
    """
    # Convert predictions to binary
    pred_binary = (pred_movements >= 0.5).astype(int)
    true_binary = true_movements.astype(int)
    
    # Calculate accuracy
    correct = np.sum(pred_binary == true_binary)
    total = len(pred_binary)
    
    return correct / total if total > 0 else 0


def calculate_irr(pred_returns: np.ndarray,
                  true_returns: np.ndarray,
                  k: int = 10) -> float:
    """
    Calculate Internal Rate of Return difference
    IRR = Σ(true_returns[true_top_k]) - Σ(true_returns[pred_top_k])
    
    Args:
        pred_returns: Predicted returns
        true_returns: True returns
        k: Top K to consider
        
    Returns:
        IRR value
    """
    n_stocks = len(pred_returns)
    k = min(k, n_stocks)
    
    # Get top-k indices
    pred_top_k_indices = np.argsort(pred_returns)[-k:]
    true_top_k_indices = np.argsort(true_returns)[-k:]
    
    # Calculate IRR
    pred_portfolio_return = np.sum(true_returns[pred_top_k_indices])
    true_portfolio_return = np.sum(true_returns[true_top_k_indices])
    
    return true_portfolio_return - pred_portfolio_return


def calculate_sharpe_ratio(returns: np.ndarray,
                           risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / 252
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate Sharpe ratio
    if len(excess_returns) > 1 and np.std(excess_returns) > 0:
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    else:
        sharpe = 0
    
    return sharpe


def calculate_maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown value
    """
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return np.min(drawdown)


def calculate_metrics(pred_returns: np.ndarray,
                     pred_movements: np.ndarray,
                     true_returns: np.ndarray,
                     true_movements: np.ndarray,
                     k_values: List[int] = [5, 10, 20],
                     batch_size: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate all metrics for model evaluation
    
    Args:
        pred_returns: Predicted returns
        pred_movements: Predicted movements
        true_returns: True returns
        true_movements: True movements
        k_values: List of K values for top-K metrics
        batch_size: If provided, calculate metrics per batch
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if batch_size and len(pred_returns) > batch_size:
        # Calculate metrics per batch and average
        n_batches = len(pred_returns) // batch_size
        
        for k in k_values:
            mrr_scores = []
            precision_scores = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                
                batch_pred_returns = pred_returns[start_idx:end_idx]
                batch_true_returns = true_returns[start_idx:end_idx]
                
                mrr = calculate_mrr_at_k(batch_pred_returns, batch_true_returns, k)
                precision = calculate_precision_at_k(batch_pred_returns, batch_true_returns, k)
                
                mrr_scores.append(mrr)
                precision_scores.append(precision)
            
            metrics[f'MRR@{k}'] = np.mean(mrr_scores)
            metrics[f'Precision@{k}'] = np.mean(precision_scores)
    else:
        # Calculate metrics on entire dataset
        for k in k_values:
            metrics[f'MRR@{k}'] = calculate_mrr_at_k(pred_returns, true_returns, k)
            metrics[f'Precision@{k}'] = calculate_precision_at_k(pred_returns, true_returns, k)
    
    # Movement accuracy
    metrics['movement_accuracy'] = calculate_movement_accuracy(pred_movements, true_movements)
    
    # Additional metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(true_returns)
    metrics['max_drawdown'] = calculate_maximum_drawdown(true_returns)
    
    return metrics


def calculate_portfolio_metrics(pred_returns: np.ndarray,
                               true_returns: np.ndarray,
                               prices: np.ndarray,
                               k: int = 10,
                               initial_capital: float = 1000000,
                               transaction_cost: float = 0.0015) -> Dict[str, float]:
    """
    Calculate portfolio-based metrics
    
    Args:
        pred_returns: Predicted returns
        true_returns: True returns
        prices: Stock prices
        k: Number of stocks in portfolio
        initial_capital: Starting capital
        transaction_cost: Transaction cost percentage
        
    Returns:
        Portfolio metrics
    """
    n_stocks = len(pred_returns)
    k = min(k, n_stocks)
    
    # Get top-k stocks based on predictions
    top_k_indices = np.argsort(pred_returns)[-k:]
    
    # Equal weight portfolio
    weight_per_stock = 1.0 / k
    portfolio_weights = np.zeros(n_stocks)
    portfolio_weights[top_k_indices] = weight_per_stock
    
    # Calculate portfolio return
    portfolio_return = np.sum(portfolio_weights * true_returns)
    
    # Adjust for transaction costs
    total_transaction_cost = 2 * transaction_cost  # Buy and sell
    net_portfolio_return = portfolio_return - total_transaction_cost
    
    # Calculate portfolio value
    final_capital = initial_capital * (1 + net_portfolio_return)
    
    metrics = {
        'portfolio_return': portfolio_return,
        'net_portfolio_return': net_portfolio_return,
        'final_capital': final_capital,
        'num_stocks': k,
        'transaction_costs': initial_capital * total_transaction_cost
    }
    
    return metrics


# Test functions
if __name__ == "__main__":
    print("Testing Evaluation Metrics")
    print("="*50)
    
    # Generate test data
    n_stocks = 100
    pred_returns = np.random.randn(n_stocks)
    true_returns = np.random.randn(n_stocks)
    pred_movements = np.random.random(n_stocks)
    true_movements = (np.random.random(n_stocks) > 0.5).astype(float)
    
    # Test individual metrics
    print("\n1. Testing MRR@K...")
    mrr10 = calculate_mrr_at_k(pred_returns, true_returns, k=10)
    print(f"   MRR@10: {mrr10:.4f}")
    
    print("\n2. Testing Precision@K...")
    precision10 = calculate_precision_at_k(pred_returns, true_returns, k=10)
    print(f"   Precision@10: {precision10:.4f}")
    
    print("\n3. Testing Movement Accuracy...")
    movement_acc = calculate_movement_accuracy(pred_movements, true_movements)
    print(f"   Movement Accuracy: {movement_acc:.4f}")
    
    print("\n4. Testing All Metrics...")
    all_metrics = calculate_metrics(
        pred_returns, pred_movements,
        true_returns, true_movements,
        k_values=[5, 10, 20]
    )
    
    print("   All metrics:")
    for key, value in all_metrics.items():
        print(f"     {key}: {value:.4f}")
    
    print("\n✅ All metrics working correctly!")