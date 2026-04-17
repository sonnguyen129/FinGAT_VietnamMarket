"""
Configuration Management for FinGAT
Centralizes all configuration in YAML files
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str
    companies_file: str
    processed_path: str
    start_date: str
    end_date: str
    min_history_days: int
    window_size: int
    num_weeks: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_stocks: int  # Number of stocks to load


@dataclass
class ModelConfig:
    """Model configuration"""
    # input_dim: auto-detected from data
    # num_sectors: auto-detected from stock-sector mapping
    time_steps: int
    hidden_dim: int
    num_weeks: int
    dropout: float
    num_heads: int
    input_dim: Optional[int] = None  # Will be set automatically


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    ranking_weight: float
    movement_weight: float
    l2_lambda: float
    optimizer: str
    scheduler: str
    scheduler_step_size: int
    scheduler_gamma: float
    gradient_clip: float
    early_stopping: bool
    patience: int
    min_delta: float
    save_every_epoch: bool
    checkpoint_dir: str
    best_model_metric: str
    log_every_n_steps: int
    val_every_n_epochs: int
    metrics_file: str


@dataclass
class SystemConfig:
    """System configuration"""
    device: str
    num_workers: int
    seed: int
    deterministic: bool


class ConfigManager:
    """
    Centralized configuration management
    """
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """
        Initialize config manager
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # Create structured configs
        self.data = DataConfig(**self.config['data'])
        self.model = ModelConfig(**self.config['model'])
        self.training = TrainingConfig(**self.config['training'])
        self.system = SystemConfig(**self.config['system'])
        
        # Raw config for other sections
        self.evaluation = self.config.get('evaluation', {})
        self.experiment = self.config.get('experiment', {})
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"📋 Loaded config from: {self.config_path}")
        return config
    
    def save_config(self, path: str = None):
        """Save current config to YAML file"""
        save_path = path or self.config_path
        
        # Update config dict with current values
        self.config['data'] = self.data.__dict__
        self.config['model'] = self.model.__dict__
        self.config['training'] = self.training.__dict__
        self.config['system'] = self.system.__dict__
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 Saved config to: {save_path}")
    
    def get_datamodule_config(self) -> Dict[str, Any]:
        """Get configuration for DataModule"""
        return {
            'data_path': self.data.data_path,
            'companies_file': self.data.companies_file,
            'start_date': self.data.start_date,
            'end_date': self.data.end_date,
            'window_size': self.data.window_size,
            'num_weeks': self.data.num_weeks,
            'train_ratio': self.data.train_ratio,
            'val_ratio': self.data.val_ratio,
            'batch_size': self.training.batch_size,
            'device': self.system.device
        }
    
    def get_model_config(self, input_dim: int = None) -> Dict[str, Any]:
        """Get configuration for Model"""
        config = {
            'hidden_dim': self.model.hidden_dim,
            'time_steps': self.model.time_steps,
            'num_weeks': self.model.num_weeks,
            'dropout': self.model.dropout,
            'num_heads': self.model.num_heads,
            'device': self.system.device
        }
        
        # Auto-detected or provided input_dim
        if input_dim is not None:
            config['input_dim'] = input_dim
        elif self.model.input_dim is not None:
            config['input_dim'] = self.model.input_dim
        else:
            # Will be set later when data is loaded
            config['input_dim'] = None
            
        return config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get configuration for Training"""
        return self.training.__dict__
    
    def update_from_args(self, args):
        """Update config from command line arguments"""
        # Update data config
        if hasattr(args, 'data_path') and args.data_path:
            self.data.data_path = args.data_path
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'epochs') and args.epochs:
            self.training.epochs = args.epochs
        if hasattr(args, 'device') and args.device:
            self.system.device = args.device
            
        print("🔄 Updated config from command line arguments")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        
        print("\n📊 DATA:")
        for key, value in self.data.__dict__.items():
            print(f"  {key:20s}: {value}")
        
        print("\n🤖 MODEL:")
        for key, value in self.model.__dict__.items():
            print(f"  {key:20s}: {value}")
        
        print("\n🏋️ TRAINING:")
        for key, value in self.training.__dict__.items():
            print(f"  {key:20s}: {value}")
        
        print("\n⚙️ SYSTEM:")
        for key, value in self.system.__dict__.items():
            print(f"  {key:20s}: {value}")
        
        print("="*60)


# Convenience function to create config manager
def load_config(config_path: str = 'configs/default_config.yaml') -> ConfigManager:
    """Load configuration from YAML file"""
    return ConfigManager(config_path)


# Test config manager
if __name__ == "__main__":
    print("Testing ConfigManager...")
    
    # Load default config
    config_manager = load_config()
    
    # Print config
    config_manager.print_config()
    
    # Test getting specific configs
    print("\n📦 DataModule Config:")
    dm_config = config_manager.get_datamodule_config()
    for key, value in dm_config.items():
        print(f"  {key}: {value}")
    
    print("\n🤖 Model Config:")
    model_config = config_manager.get_model_config(num_stocks=50, num_sectors=10)
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print("\n✅ ConfigManager working correctly!")