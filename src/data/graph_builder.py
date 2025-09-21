"""
Graph Builder for Vietnamese Stock Market
Constructs intra-sector and inter-sector graphs for FinGAT
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr


class GraphBuilder:
    """Build graph structures for FinGAT model"""
    
    def __init__(self, 
                 correlation_threshold: float = 0.3,
                 top_k_edges: Optional[int] = None):
        """
        Initialize Graph Builder
        
        Args:
            correlation_threshold: Minimum correlation for edge creation
            top_k_edges: Keep only top-k edges per node
        """
        self.correlation_threshold = correlation_threshold
        self.top_k_edges = top_k_edges
        
        # Graph storage
        self.intra_sector_edges = None
        self.inter_sector_edges = None
        self.stock_to_idx = {}
        self.idx_to_stock = {}
        self.sector_to_idx = {}
        self.idx_to_sector = {}
        
    def build_stock_mapping(self, stocks: List[str]) -> Dict:
        """
        Create mapping between stock symbols and indices
        
        Args:
            stocks: List of stock symbols
        
        Returns:
            Mappings dictionary
        """
        self.stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
        self.idx_to_stock = {idx: stock for stock, idx in self.stock_to_idx.items()}
        
        return {
            'stock_to_idx': self.stock_to_idx,
            'idx_to_stock': self.idx_to_stock
        }
    
    def build_sector_mapping(self, sectors: List[str]) -> Dict:
        """
        Create mapping between sectors and indices
        
        Args:
            sectors: List of unique sectors
        
        Returns:
            Mappings dictionary
        """
        self.sector_to_idx = {sector: idx for idx, sector in enumerate(sectors)}
        self.idx_to_sector = {idx: sector for sector, idx in self.sector_to_idx.items()}
        
        return {
            'sector_to_idx': self.sector_to_idx,
            'idx_to_sector': self.idx_to_sector
        }
    
    def build_intra_sector_edges(self, 
                                 sector_mapping: Dict[str, List[str]],
                                 fully_connected: bool = True,
                                 return_matrix: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Build edges within each sector
        
        Args:
            sector_mapping: Dictionary mapping sectors to stock lists
            fully_connected: If True, create fully connected graph within sector
            return_matrix: Return data for correlation-based edges
        
        Returns:
            Edge index array of shape (2, num_edges)
        """
        edges = []
        
        for sector, stocks in sector_mapping.items():
            stock_indices = [self.stock_to_idx[stock] for stock in stocks 
                           if stock in self.stock_to_idx]
            
            if fully_connected:
                # Create fully connected graph within sector
                for i, idx1 in enumerate(stock_indices):
                    for idx2 in stock_indices[i+1:]:
                        edges.append([idx1, idx2])
                        edges.append([idx2, idx1])  # Bidirectional
            else:
                # Create correlation-based edges
                if return_matrix is not None:
                    edges_in_sector = self._create_correlation_edges(
                        stocks, return_matrix
                    )
                    edges.extend(edges_in_sector)
        
        if len(edges) == 0:
            # Return empty edge tensor if no edges
            return np.array([[], []]).astype(np.int64)
        
        self.intra_sector_edges = np.array(edges).T
        return self.intra_sector_edges
    
    def build_inter_sector_edges(self, 
                                 sectors: List[str],
                                 fully_connected: bool = True) -> np.ndarray:
        """
        Build edges between sectors
        
        Args:
            sectors: List of unique sectors
            fully_connected: If True, create fully connected graph
        
        Returns:
            Edge index array of shape (2, num_edges)
        """
        edges = []
        sector_indices = list(range(len(sectors)))
        
        if fully_connected:
            # Create fully connected graph between sectors
            for i, idx1 in enumerate(sector_indices):
                for idx2 in sector_indices[i+1:]:
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])  # Bidirectional
        
        if len(edges) == 0:
            return np.array([[], []]).astype(np.int64)
        
        self.inter_sector_edges = np.array(edges).T
        return self.inter_sector_edges
    
    def _create_correlation_edges(self, 
                                  stocks: List[str],
                                  return_matrix: pd.DataFrame) -> List:
        """
        Create edges based on return correlation
        
        Args:
            stocks: List of stock symbols
            return_matrix: DataFrame with returns
        
        Returns:
            List of edges
        """
        edges = []
        
        for i, stock1 in enumerate(stocks):
            if stock1 not in return_matrix.columns:
                continue
            
            correlations = []
            for stock2 in stocks[i+1:]:
                if stock2 not in return_matrix.columns:
                    continue
                
                # Calculate correlation
                corr = return_matrix[stock1].corr(return_matrix[stock2])
                
                if not np.isnan(corr) and abs(corr) > self.correlation_threshold:
                    idx1 = self.stock_to_idx[stock1]
                    idx2 = self.stock_to_idx[stock2]
                    correlations.append((idx2, abs(corr)))
            
            # Sort by correlation and keep top-k if specified
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            if self.top_k_edges:
                correlations = correlations[:self.top_k_edges]
            
            # Add edges
            idx1 = self.stock_to_idx[stock1]
            for idx2, _ in correlations:
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])  # Bidirectional
        
        return edges
    
    def build_dynamic_graph(self,
                           return_matrix: pd.DataFrame,
                           window_size: int = 60) -> List[np.ndarray]:
        """
        Build time-varying graphs based on rolling correlation
        
        Args:
            return_matrix: DataFrame with returns
            window_size: Rolling window for correlation
        
        Returns:
            List of edge indices for each time point
        """
        dynamic_edges = []
        stocks = list(return_matrix.columns)
        
        for t in range(window_size, len(return_matrix)):
            # Get returns for window
            window_returns = return_matrix.iloc[t-window_size:t]
            
            # Build correlation matrix
            corr_matrix = window_returns.corr()
            
            # Create edges based on correlation
            edges = []
            for i, stock1 in enumerate(stocks):
                for j, stock2 in enumerate(stocks[i+1:], i+1):
                    corr = corr_matrix.loc[stock1, stock2]
                    
                    if not np.isnan(corr) and abs(corr) > self.correlation_threshold:
                        idx1 = self.stock_to_idx[stock1]
                        idx2 = self.stock_to_idx[stock2]
                        edges.append([idx1, idx2])
                        edges.append([idx2, idx1])
            
            if edges:
                edge_index = np.array(edges).T
            else:
                edge_index = np.array([[], []]).astype(np.int64)
            
            dynamic_edges.append(edge_index)
        
        return dynamic_edges
    
    def create_batch_edges(self,
                          edge_index: np.ndarray,
                          batch_size: int) -> torch.Tensor:
        """
        Create batched edge indices for multiple graphs
        
        Args:
            edge_index: Original edge index
            batch_size: Number of graphs in batch
        
        Returns:
            Batched edge index tensor
        """
        if edge_index.size == 0:
            return torch.tensor([[], []], dtype=torch.long)
        
        num_nodes = len(self.stock_to_idx)
        batch_edges = []
        
        for batch_idx in range(batch_size):
            offset = batch_idx * num_nodes
            batch_edge = edge_index + offset
            batch_edges.append(batch_edge)
        
        batched_edge_index = np.concatenate(batch_edges, axis=1)
        return torch.tensor(batched_edge_index, dtype=torch.long)
    
    def get_edge_statistics(self) -> Dict:
        """Get statistics about constructed graphs"""
        stats = {}
        
        if self.intra_sector_edges is not None:
            stats['intra_sector_edges'] = self.intra_sector_edges.shape[1]
            stats['intra_sector_avg_degree'] = (
                self.intra_sector_edges.shape[1] / len(self.stock_to_idx)
            )
        
        if self.inter_sector_edges is not None:
            stats['inter_sector_edges'] = self.inter_sector_edges.shape[1]
            stats['inter_sector_avg_degree'] = (
                self.inter_sector_edges.shape[1] / len(self.sector_to_idx)
            )
        
        stats['num_stocks'] = len(self.stock_to_idx)
        stats['num_sectors'] = len(self.sector_to_idx)
        
        return stats
    
    def save_edges(self, save_path: str = 'edges/') -> None:
        """Save edge indices to files"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if self.intra_sector_edges is not None:
            np.save(f'{save_path}/intra_sector_edges.npy', self.intra_sector_edges)
        
        if self.inter_sector_edges is not None:
            np.save(f'{save_path}/inter_sector_edges.npy', self.inter_sector_edges)
        
        # Save mappings
        import pickle
        mappings = {
            'stock_to_idx': self.stock_to_idx,
            'idx_to_stock': self.idx_to_stock,
            'sector_to_idx': self.sector_to_idx,
            'idx_to_sector': self.idx_to_sector
        }
        
        with open(f'{save_path}/mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
    
    def load_edges(self, load_path: str = 'edges/') -> None:
        """Load edge indices from files"""
        import os
        import pickle
        
        if os.path.exists(f'{load_path}/intra_sector_edges.npy'):
            self.intra_sector_edges = np.load(f'{load_path}/intra_sector_edges.npy')
        
        if os.path.exists(f'{load_path}/inter_sector_edges.npy'):
            self.inter_sector_edges = np.load(f'{load_path}/inter_sector_edges.npy')
        
        if os.path.exists(f'{load_path}/mappings.pkl'):
            with open(f'{load_path}/mappings.pkl', 'rb') as f:
                mappings = pickle.load(f)
                self.stock_to_idx = mappings['stock_to_idx']
                self.idx_to_stock = mappings['idx_to_stock']
                self.sector_to_idx = mappings['sector_to_idx']
                self.idx_to_sector = mappings['idx_to_sector']


# Example usage
if __name__ == "__main__":
    # Create sample data
    stocks = ['VNM', 'VIC', 'VHM', 'HPG', 'MSN', 'FPT']
    sectors = ['Hàng Tiêu dùng', 'Tài chính', 'Nguyên vật liệu', 'Công nghệ Thông tin']
    
    sector_mapping = {
        'Hàng Tiêu dùng': ['VNM', 'MSN'],
        'Tài chính': ['VIC', 'VHM'],
        'Nguyên vật liệu': ['HPG'],
        'Công nghệ Thông tin': ['FPT']
    }
    
    # Initialize graph builder
    gb = GraphBuilder(correlation_threshold=0.3)
    
    # Build mappings
    gb.build_stock_mapping(stocks)
    gb.build_sector_mapping(sectors)
    
    # Build graphs
    intra_edges = gb.build_intra_sector_edges(sector_mapping, fully_connected=True)
    inter_edges = gb.build_inter_sector_edges(sectors, fully_connected=True)
    
    print("Graph Statistics:")
    print(gb.get_edge_statistics())
    
    # Create batched edges
    batch_size = 32
    batched_edges = gb.create_batch_edges(intra_edges, batch_size)
    print(f"\nBatched edges shape: {batched_edges.shape}")