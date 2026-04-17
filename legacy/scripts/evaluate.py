"""
Comprehensive evaluation script for FinGAT model
Integrates metrics calculation, backtesting, and visualization
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fingat import FinGAT
from src.data.fingat_datamodule import FinGATDataModule
from src.evaluation.metrics import (
    calculate_mrr_at_k,
    calculate_precision_at_k,
    calculate_movement_accuracy,
    calculate_metrics
)
from src.evaluation.backtesting import FinGATBacktester


class FinGATEvaluator:
    """
    Complete evaluation pipeline for FinGAT model
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 data_dir: str = 'datasets/VN_datasets',
                 experiment_name: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator
        
        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Directory containing Vietnamese stock data
            experiment_name: Name for saving results
            device: Device to use for evaluation
        """
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.device = device
        
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"eval_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create output directories
        self.output_dir = Path(f'experiments/{self.experiment_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'backtest').mkdir(exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize data module
        self.datamodule = FinGATDataModule(
            data_path=data_dir,
            companies_file='datasets/VN_Companies.csv',
            device=device
        )
        
        print(f"✅ Evaluator initialized")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Output: {self.output_dir}")
        print(f"   - Device: {device}")
    
    def _load_model(self) -> FinGAT:
        """Load model from checkpoint"""
        print(f"Loading model from {self.checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get model configuration
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # Default configuration
            model_config = {
                'input_dim': 19,
                'hidden_dim': 16,
                'num_weeks': 3,
                'num_stocks': 100,
                'num_sectors': 10,
                'dropout': 0.2
            }
        
        # Initialize model
        model = FinGAT(**model_config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
    
    def evaluate_model(self, test_loader, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            k_values: K values for top-K metrics
            
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "="*50)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*50)
        
        self.model.eval()
        all_pred_returns = []
        all_true_returns = []
        all_pred_movements = []
        all_true_movements = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get batch data
                features = batch['features']
                target_returns = batch['returns']
                target_movements = batch['movements']
                edge_index_intra = batch['edge_index_intra']
                edge_index_inter = batch['edge_index_inter']
                stock_to_sector = batch['stock_to_sector']
                
                # Forward pass
                pred_returns, pred_movements = self.model(
                    features,
                    edge_index_intra,
                    edge_index_inter,
                    stock_to_sector
                )
                
                # Store predictions
                all_pred_returns.extend(pred_returns.cpu().numpy())
                all_true_returns.extend(target_returns.cpu().numpy())
                all_pred_movements.extend(pred_movements.cpu().numpy())
                all_true_movements.extend(target_movements.cpu().numpy())
        
        # Convert to numpy arrays
        pred_returns = np.array(all_pred_returns)
        true_returns = np.array(all_true_returns)
        pred_movements = np.array(all_pred_movements)
        true_movements = np.array(all_true_movements)
        
        # Calculate metrics
        results = {}
        
        # Ranking metrics
        for k in k_values:
            results[f'MRR@{k}'] = calculate_mrr_at_k(pred_returns, true_returns, k)
            results[f'Precision@{k}'] = calculate_precision_at_k(pred_returns, true_returns, k)
            print(f"  MRR@{k}: {results[f'MRR@{k}']:.4f}")
            print(f"  Precision@{k}: {results[f'Precision@{k}']:.4f}")
        
        # Movement accuracy
        results['movement_accuracy'] = calculate_movement_accuracy(pred_movements, true_movements)
        print(f"  Movement Accuracy: {results['movement_accuracy']:.4f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([results])
        metrics_df.to_csv(self.output_dir / 'metrics' / 'evaluation_metrics.csv', index=False)
        
        print(f"\n✅ Evaluation complete. Results saved to {self.output_dir / 'metrics'}")
        
        return results
    
    def run_complete_evaluation(self,
                               k_values: List[int] = [5, 10, 20]):
        """
        Run complete evaluation pipeline
        
        Args:
            k_values: K values for metrics
        """
        print("\n" + "="*70)
        print("FINGAT COMPLETE EVALUATION PIPELINE")
        print("="*70)
        
        try:
            # Prepare data
            print("\n📊 Preparing data...")
            self.datamodule.prepare_data()
            
            # Get test loader
            _, _, test_loader = self.datamodule.get_dataloaders()
            
            # Evaluate model
            evaluation_metrics = self.evaluate_model(test_loader, k_values)
            
            print("\n" + "="*70)
            print("✅ EVALUATION COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\nAll results saved to: {self.output_dir}")
            
            return evaluation_metrics
            
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate FinGAT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='datasets/VN_datasets',
                       help='Directory containing Vietnamese stock data')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for experiment')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for top-K metrics')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FinGATEvaluator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.run_complete_evaluation(k_values=args.k_values)
    
    if results:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()