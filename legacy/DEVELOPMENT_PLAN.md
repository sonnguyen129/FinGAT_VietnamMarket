# Kế Hoạch Phát Triển FinGAT cho Thị Trường Việt Nam

## 1. Phân Tích Hiện Trạng

### Code Hiện Tại
- **clean_data.py**: Xử lý dữ liệu SP500 (cần chuyển đổi cho VN)
- **train.py**: Pipeline huấn luyện với các model CG, CAT, CPool
- **model/graph_pool.py**: Các model FinGAT implementations
- **parse_arg.py**: Xử lý arguments

### Dữ Liệu Việt Nam
- **VN_datasets/**: 400+ file CSV cổ phiếu (OHLCV data)
- **VN_Companies.csv**: Thông tin ngành/sector của các cổ phiếu
- Format: Date, High, Low, Open, Close, Adj Close, Volume

## 2. Kế Hoạch Phát Triển Chi Tiết

### Phase 1: Data Pipeline (Ưu tiên cao)

#### 1.1 Data Loader cho VN Market
```python
# Tạo file: src/data/vn_data_loader.py
- Load tất cả cổ phiếu từ VN_datasets/
- Merge với thông tin sector từ VN_Companies.csv
- Xử lý missing data và alignment
- Filter cổ phiếu có đủ lịch sử
```

#### 1.2 Feature Engineering
```python
# Tạo file: src/data/feature_engineering.py
- Return ratio: (p_t - p_{t-1})/p_{t-1}
- Price ratios: open/close, high/close, low/close
- Moving averages: MA_5, MA_10, MA_15, MA_20, MA_25, MA_30
- Technical indicators (RSI, MACD nếu cần)
- Volume features
```

#### 1.3 Graph Construction
```python
# Tạo file: src/data/graph_builder.py
- Intra-sector edges: Kết nối cổ phiếu cùng ngành
- Inter-sector edges: Kết nối giữa các ngành
- Dynamic graph construction dựa trên correlation
```

#### 1.4 Data Preprocessing Pipeline
```python
# Tạo file: src/data/preprocessor.py
- Sliding window creation (15 days input + 1 day target)
- Week grouping (5 trading days = 1 week)
- Train/Val/Test split (60/20/20)
- Data normalization và scaling
```

### Phase 2: Model Implementation (Ưu tiên cao)

#### 2.1 Refactor Model Architecture
```python
# Cấu trúc mới trong src/models/
├── components/
│   ├── attention.py        # Attention mechanisms
│   ├── encoders.py         # GRU encoders
│   └── gat_layers.py       # GAT implementations
├── fingat.py              # Main FinGAT model
├── fingat_nt.py           # No-sector variant
└── baseline_models.py      # MLP, GRU, etc.
```

#### 2.2 Implement FinGAT Components
- Stock-level modeling:
  - Short-term encoder (GRU + Attention)
  - Intra-sector GAT
  - Long-term encoder (Attentive GRU)
- Sector-level modeling:
  - Sector pooling
  - Inter-sector GAT
- Fusion layer và multi-task heads

### Phase 3: Training Pipeline (Ưu tiên cao)

#### 3.1 Loss Functions
```python
# Tạo file: src/training/losses.py
- Pairwise ranking loss
- Binary cross-entropy for movement
- Multi-task loss combination
- Custom loss với Vietnamese market weights
```

#### 3.2 Training Loop
```python
# Tạo file: src/training/trainer.py
- Support cho multiple models
- Early stopping và checkpointing
- Learning rate scheduling
- Tensorboard/WandB logging
```

#### 3.3 Optimization
```python
# Tạo file: src/training/optimizer.py
- Adam với weight decay
- Gradient clipping
- Mixed precision training
```

### Phase 4: Evaluation Framework (Ưu tiên trung bình)

#### 4.1 Metrics Implementation
```python
# Tạo file: src/evaluation/metrics.py
- MRR@K (K=5,10,20)
- Precision@K
- Movement accuracy
- Sharpe ratio
- Maximum drawdown
```

#### 4.2 Backtesting
```python
# Tạo file: src/evaluation/backtest.py
- Portfolio simulation
- Transaction costs
- Risk metrics
- Performance visualization
```

### Phase 5: Configuration System (Ưu tiên trung bình)

#### 5.1 Config Files
```yaml
# configs/vn_market.yaml
data:
  path: datasets/VN_datasets/
  companies_file: datasets/VN_Companies.csv
  start_date: 2018-01-01
  end_date: 2023-12-31
  
model:
  hidden_dim: 16
  num_weeks: 3
  dropout: 0.2
  
training:
  batch_size: 128
  learning_rate: 0.001
  epochs: 100
  loss_weights:
    ranking: 0.98
    movement: 0.01
    l2_reg: 0.0001
```

### Phase 6: Scripts và Utilities

#### 6.1 Main Scripts
```python
# scripts/prepare_vn_data.py - Chuẩn bị dữ liệu
# scripts/train_vn.py - Training script
# scripts/evaluate_vn.py - Evaluation script
# scripts/predict_vn.py - Inference script
```

#### 6.2 Utilities
```python
# src/utils/
├── logger.py          # Logging utilities
├── visualization.py   # Plot functions
├── data_utils.py     # Data helpers
└── model_utils.py    # Model helpers
```

## 3. Implementation Timeline

### Week 1-2: Data Pipeline
- [x] Analyze existing VN data structure
- [ ] Implement VN data loader
- [ ] Feature engineering pipeline
- [ ] Graph construction
- [ ] Data validation và testing

### Week 3-4: Model Implementation
- [ ] Refactor existing model code
- [ ] Implement FinGAT components
- [ ] Add Vietnamese market adaptations
- [ ] Model testing và debugging

### Week 5-6: Training & Evaluation
- [ ] Implement training pipeline
- [ ] Add evaluation metrics
- [ ] Hyperparameter tuning
- [ ] Results analysis

### Week 7-8: Optimization & Documentation
- [ ] Performance optimization
- [ ] Code documentation
- [ ] Create notebooks for analysis
- [ ] Final testing và validation

## 4. Các Điểm Cần Lưu Ý

### 4.1 Đặc Thù Thị Trường Việt Nam
- **Trading days**: T+2 settlement
- **Price limits**: ±7% daily limit
- **Market hours**: 9:00-15:00 với break 11:30-13:00
- **Liquidity**: Một số cổ phiếu có thanh khoản thấp
- **Sectors**: Phân ngành khác với US market

### 4.2 Technical Considerations
- **Memory optimization**: 400+ stocks cần xử lý hiệu quả
- **Batch processing**: Xử lý theo batch để tránh OOM
- **GPU utilization**: Optimize cho PyTorch Geometric
- **Checkpointing**: Save model regularly

### 4.3 Evaluation Specifics
- **Transaction costs**: 0.15% cho VN market
- **Tax**: 0.1% sell tax
- **Minimum lot**: 100 shares
- **Currency**: VND (000s)

## 5. Expected Deliverables

### 5.1 Code
- Complete FinGAT implementation for VN market
- Modular và reusable components
- Well-documented code
- Unit tests

### 5.2 Models
- Trained FinGAT model
- Baseline models for comparison
- Model checkpoints

### 5.3 Results
- Performance metrics
- Comparison với baselines
- Visualization và analysis
- Trading signals

### 5.4 Documentation
- Technical documentation
- User guide
- API documentation
- Jupyter notebooks

## 6. Next Steps

1. **Immediate**: Start với VN data loader
2. **Priority**: Feature engineering pipeline
3. **Critical**: Graph construction cho VN sectors
4. **Important**: Adapt model cho VN market characteristics

## 7. Risk Mitigation

- **Data quality**: Validate và clean thoroughly
- **Overfitting**: Use proper validation splits
- **Market changes**: Consider regime changes
- **Computational**: Optimize for efficiency

## 8. Success Criteria

- MRR@10 > 0.3
- Movement accuracy > 55%
- Stable training convergence
- Reproducible results
- Clean, maintainable code