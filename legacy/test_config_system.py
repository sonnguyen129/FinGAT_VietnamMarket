"""
Test Config Management System
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.config_manager import load_config
from src.data.fingat_datamodule import FinGATDataModule


def test_config_system():
    """Test the configuration system"""
    print("🧪 Testing Config Management System")
    print("="*60)
    
    # 1. Test ConfigManager
    print("\n1️⃣ Testing ConfigManager...")
    config_manager = load_config('configs/default_config.yaml')
    config_manager.print_config()
    
    # 2. Test DataModule with Config
    print("\n2️⃣ Testing DataModule with Config...")
    datamodule = FinGATDataModule(
        config_manager=config_manager,
        num_stocks_limit=10  # Small test
    )
    
    print(f"✅ DataModule created with config:")
    print(f"  - Data path: {datamodule.data_path}")
    print(f"  - Window size: {datamodule.window_size}")
    print(f"  - Batch size: {datamodule.batch_size}")
    print(f"  - Train ratio: {datamodule.train_ratio}")
    
    # 3. Test model config generation
    print("\n3️⃣ Testing Model Config generation...")
    try:
        # Need to prepare data first to get stocks/sectors count
        datamodule.prepare_data()
        model_config = datamodule.get_model_config()
        
        print(f"✅ Model config generated:")
        for key, value in model_config.items():
            print(f"  - {key}: {value}")
            
    except Exception as e:
        print(f"⚠️ Could not test full pipeline: {e}")
        print("This is expected if datasets are not available")
    
    # 4. Test config saving
    print("\n4️⃣ Testing Config saving...")
    test_config_path = 'test_config.yaml'
    config_manager.save_config(test_config_path)
    
    # Clean up
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
        print(f"🗑️ Cleaned up test file: {test_config_path}")
    
    print("\n✅ Config system test completed!")


if __name__ == "__main__":
    test_config_system()