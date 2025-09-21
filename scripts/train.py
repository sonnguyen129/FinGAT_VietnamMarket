#!/usr/bin/env python
"""
Main training script for FinGAT on Vietnamese stock market
Supports both default training and hyperparameter tuning modes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

from src.models.fingat import FinGAT
from src.data.datamodule import VNStockDataModule
from src.training.trainer import FinGATTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge override config into base config"""
    import copy
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def train_default(config: dict, resume_from: str = None):
    """
    Train with default configuration
    
    Args:
        config: Configuration dictionary
        resume_from: Path to checkpoint to resume from
    """
    print("\n" + "="*70)
    print("FINGAT TRAINING - DEFAULT MODE")
    print("="*70)
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Initialize data module
    print("\n1. Loading data...")
    data_module = VNStockDataModule(
        data_path=config['data']['processed_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['system']['num_workers']
    )
    
    # Get data statistics
    data_stats = data_module.get_data_stats()
    print(f"   Loaded {data_stats['train_samples']} training samples")
    print(f"   Loaded {data_stats['val_samples']} validation samples")
    print(f"   Loaded {data_stats['test_samples']} test samples")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = FinGAT(
        input_dim=data_stats['num_features'],
        time_steps=config['model']['time_steps'],
        hidden_dim=config['model']['hidden_dim'],
        num_weeks=config['model']['num_weeks'],
        num_stocks=data_stats['num_stocks'],
        num_sectors=data_stats['num_sectors'],
        dropout=config['model']['dropout'],
        device=config['system']['device']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Initialize trainer
    print("\n3. Initializing trainer...")
    trainer = FinGATTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        experiment_name=config['experiment']['name']
    )
    
    # Resume from checkpoint if provided
    if resume_from:
        print(f"\n4. Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Train model
    print("\n5. Starting training...")
    trainer.train()
    
    # Test model
    print("\n6. Testing model...")
    test_metrics = trainer.test()
    
    return test_metrics


def train_hyperparameter_tuning(base_config: dict, tuning_config: dict):
    """
    Train with hyperparameter tuning
    
    Args:
        base_config: Base configuration
        tuning_config: Hyperparameter tuning configuration
    """
    print("\n" + "="*70)
    print("FINGAT TRAINING - HYPERPARAMETER TUNING MODE")
    print("="*70)
    
    # Get search configurations
    if tuning_config['tuning']['method'] == 'grid':
        configurations = tuning_config['grid_search']['configurations']
    else:
        # For random/bayesian search, generate configurations
        configurations = generate_random_configs(
            base_config,
            tuning_config['search_space'],
            tuning_config['random_search']['num_samples']
        )
    
    print(f"\nTesting {len(configurations)} configurations...")
    
    results = []
    
    for idx, config_override in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Configuration {idx+1}/{len(configurations)}")
        
        if isinstance(config_override, dict) and 'name' in config_override:
            print(f"Name: {config_override['name']}")
            config_name = config_override['name']
        else:
            config_name = f"config_{idx+1}"
        
        print(f"{'='*60}")
        
        # Merge configurations
        trial_config = merge_configs(base_config, {'model': {}, 'training': {}})
        
        # Apply overrides
        for key, value in config_override.items():
            if key in ['hidden_dim', 'dropout', 'num_weeks']:
                trial_config['model'][key] = value
            elif key in ['learning_rate', 'batch_size', 'movement_weight', 'l2_lambda', 'weight_decay', 'gradient_clip']:
                trial_config['training'][key] = value
                
                # Update ranking weight based on movement weight
                if key == 'movement_weight':
                    trial_config['training']['ranking_weight'] = 1.0 - value
        
        # Update experiment name
        trial_config['experiment']['name'] = f"tuning_{config_name}"
        
        # Reduce epochs for tuning
        trial_config['training']['epochs'] = tuning_config['tuning_training']['epochs']
        trial_config['training']['patience'] = tuning_config['tuning_training']['patience']
        
        try:
            # Train with this configuration
            metrics = train_default(trial_config)
            
            # Store results
            result = {
                'config_name': config_name,
                'config': config_override,
                'metrics': metrics
            }
            results.append(result)
            
            print(f"\nResults for {config_name}:")
            print(f"  Val MRR@10: {metrics.get('MRR@10', 0):.4f}")
            print(f"  Val Movement Acc: {metrics.get('movement_accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"\n❌ Configuration {config_name} failed: {e}")
            continue
    
    # Sort results by target metric
    target_metric = tuning_config['tuning']['metric']
    results.sort(key=lambda x: x['metrics'].get(target_metric, 0), reverse=True)
    
    # Print summary
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    
    print(f"\nTop 5 configurations by {target_metric}:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. {result['config_name']}")
        print(f"   {target_metric}: {result['metrics'].get(target_metric, 0):.4f}")
        print(f"   Config: {result['config']}")
    
    # Save results
    results_path = Path(tuning_config['output']['results_dir'])
    results_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_path / f"tuning_results_{timestamp}.yaml"
    
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def generate_random_configs(base_config: dict, search_space: dict, num_samples: int):
    """Generate random configurations from search space"""
    configs = []
    
    for i in range(num_samples):
        config = {}
        
        # Sample from each parameter space
        for category, params in search_space.items():
            for param, spec in params.items():
                if spec['type'] == 'choice':
                    config[param] = random.choice(spec['values'])
                elif spec['type'] == 'uniform':
                    config[param] = random.uniform(spec['min'], spec['max'])
                elif spec['type'] == 'loguniform':
                    config[param] = np.exp(random.uniform(np.log(spec['min']), np.log(spec['max'])))
        
        config['name'] = f'random_{i+1}'
        configs.append(config)
    
    return configs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train FinGAT model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='default',
                       choices=['default', 'tuning'],
                       help='Training mode: default or hyperparameter tuning')
    parser.add_argument('--tuning-config', type=str, default='configs/hyperparameter_tuning.yaml',
                       help='Path to hyperparameter tuning configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load base configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['system']['device'] = args.device
    
    # Check CUDA availability
    if config['system']['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        config['system']['device'] = 'cpu'
    
    # Run training based on mode
    if args.mode == 'default':
        train_default(config, args.resume)
    elif args.mode == 'tuning':
        tuning_config = load_config(args.tuning_config)
        train_hyperparameter_tuning(config, tuning_config)
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()