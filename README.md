# FinGAT - Vietnamese Stock Market Implementation

## Project Overview
- **Name**: FinGAT-VN (Financial Graph Attention Network for Vietnamese Market)
- **Goal**: Implement and adapt FinGAT model for Vietnamese stock market to recommend top-K profitable stocks
- **Features**: 
  - Graph Attention Network for capturing stock relationships
  - Multi-task learning (ranking + movement prediction)
  - Sector-level and stock-level modeling
  - Support for Vietnamese market data

## Project Structure
```
webapp/
├── src/
│   ├── models/          # FinGAT model implementation
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training pipeline
│   ├── evaluation/     # Metrics and evaluation
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── datasets/           # Data directory
│   ├── VN_datasets/    # Vietnamese stock data
│   ├── VN_companies.csv # Sector information
│   └── processed/      # Processed data
├── experiments/        # Experiment results
├── scripts/           # Training and evaluation scripts
└── notebooks/         # Analysis notebooks
```

## Data Architecture
- **Data Models**: 
  - Stock-level features: OHLCV data, technical indicators
  - Sector relationships: Industry classification
  - Temporal patterns: Weekly and cross-week sequences
- **Storage Services**: Local file system (CSV format)
- **Data Flow**: Raw data → Feature extraction → Graph construction → Model training

## Implementation Plan

### Phase 1: Data Pipeline (Current)
- [ ] Vietnamese market data loader
- [ ] Feature engineering (returns, ratios, moving averages)
- [ ] Sliding window creation
- [ ] Graph construction from sector information

### Phase 2: Model Implementation
- [ ] Stock-level modeling (GRU + Attention)
- [ ] Intra-sector GAT
- [ ] Sector-level modeling
- [ ] Inter-sector GAT
- [ ] Embedding fusion

### Phase 3: Training & Evaluation
- [ ] Multi-task loss implementation
- [ ] Training loop with validation
- [ ] Evaluation metrics (MRR@K, Precision@K, ACC)
- [ ] Visualization and analysis

## Technical Details

### Model Architecture
1. **Stock-level Modeling**:
   - Short-term encoder: GRU over 5-day weeks
   - Intra-sector GAT for relationship modeling
   - Long-term encoder: Attentive GRU across weeks

2. **Sector-level Modeling**:
   - Sector pooling via max-pooling
   - Inter-sector GAT for sector relationships

3. **Multi-task Learning**:
   - Ranking task: Pairwise ranking loss
   - Movement task: Binary classification

### Hyperparameters
- Hidden dimension: 16
- Number of weeks: 3
- Learning rate: 0.001
- Batch size: 128
- Loss weights: δ=0.01 (movement), λ=1e-4 (L2)

## Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Prepare Vietnamese data
python scripts/prepare_vn_data.py

# Train model
python scripts/train.py --config configs/vn_market.yaml
```

### Evaluation
```bash
# Evaluate model
python scripts/evaluate.py --checkpoint experiments/best_model.pth

# Generate predictions
python scripts/predict.py --date 2024-01-01
```

## Results
- **MRR@5**: TBD
- **MRR@10**: TBD
- **Precision@5**: TBD
- **Movement Accuracy**: TBD

## References
- Paper: [FinGAT: Financial Graph Attention Networks](https://arxiv.org/abs/2106.10159)
- Original repo: [GitHub](https://github.com/Roytsai27/Financial-GraphAttention)

## Status
- **Platform**: Local development
- **Status**: 🚧 Under Development
- **Tech Stack**: PyTorch + PyTorch Geometric
- **Last Updated**: 2024-01-21