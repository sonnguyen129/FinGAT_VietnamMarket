"""
Graph Constructor for FinGAT
Creates edge indices and stock-to-sector mappings for GAT layers
Ensures consistency with paper's graph structure
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GraphConstructor:
    """
    Constructs graph structure for FinGAT model
    - Intra-sector edges: connections within same sector
    - Inter-sector edges: connections between different sectors  
    - Stock-to-sector mapping: assigns each stock to its sector
    """
    
    def __init__(self, 
                 stocks: List[str],
                 sector_mapping: Dict[str, str],
                 intra_sector_connectivity: float = 1.0,
                 inter_sector_connectivity: float = 0.3):
        """
        Initialize graph constructor
        
        Args:
            stocks: List of stock symbols
            sector_mapping: Dict mapping stock symbol to sector name
            intra_sector_connectivity: Connectivity ratio within sectors (1.0 = fully connected)
            inter_sector_connectivity: Connectivity ratio between sectors (0.3 = 30% connected)
        """
        self.stocks = stocks
        self.sector_mapping = sector_mapping
        self.intra_sector_connectivity = intra_sector_connectivity
        self.inter_sector_connectivity = inter_sector_connectivity
        
        # Create mappings
        self.stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
        self.idx_to_stock = {idx: stock for idx, stock in enumerate(stocks)}
        
        # Get unique sectors and create sector mappings
        self.sectors = list(set(sector_mapping.values()))
        self.sector_to_idx = {sector: idx for idx, sector in enumerate(self.sectors)}
        self.idx_to_sector = {idx: sector for idx, sector in enumerate(self.sectors)}
        
        # Create stock-to-sector index tensor
        self.stock_to_sector_tensor = self._create_stock_to_sector_tensor()
        
        # Group stocks by sector
        self.stocks_by_sector = self._group_stocks_by_sector()
        
        print(f"📊 Graph Constructor initialized:")
        print(f"  - Stocks: {len(self.stocks)}")
        print(f"  - Sectors: {len(self.sectors)}")
        print(f"  - Intra-sector connectivity: {intra_sector_connectivity:.1%}")
        print(f"  - Inter-sector connectivity: {inter_sector_connectivity:.1%}")
    
    def _create_stock_to_sector_tensor(self) -> torch.LongTensor:
        """
        Create tensor mapping each stock index to its sector index
        
        Returns:
            Tensor of shape (num_stocks,) with sector indices
        """
        stock_to_sector = torch.zeros(len(self.stocks), dtype=torch.long)
        
        for stock_idx, stock in self.idx_to_stock.items():
            if stock in self.sector_mapping:
                sector = self.sector_mapping[stock]
                if sector in self.sector_to_idx:
                    sector_idx = self.sector_to_idx[sector]
                    stock_to_sector[stock_idx] = sector_idx
        
        return stock_to_sector
    
    def _group_stocks_by_sector(self) -> Dict[str, List[int]]:
        """
        Group stock indices by their sectors
        
        Returns:
            Dict mapping sector name to list of stock indices
        """
        stocks_by_sector = {sector: [] for sector in self.sectors}
        
        for stock_idx, stock in self.idx_to_stock.items():
            if stock in self.sector_mapping:
                sector = self.sector_mapping[stock]
                if sector in stocks_by_sector:
                    stocks_by_sector[sector].append(stock_idx)
        
        return stocks_by_sector
    
    def create_intra_sector_edges(self, batch_size: int = 1) -> torch.LongTensor:
        """
        Create intra-sector edges (connections within same sector)
        According to paper: stocks in same sector are connected
        
        Args:
            batch_size: Number of graphs in batch
            
        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        edges = []
        
        for batch_idx in range(batch_size):
            offset = batch_idx * len(self.stocks)
            
            # For each sector, connect stocks within that sector
            for sector, stock_indices in self.stocks_by_sector.items():
                if len(stock_indices) < 2:
                    continue
                
                # Create edges based on connectivity ratio
                n_stocks = len(stock_indices)
                n_possible_edges = n_stocks * (n_stocks - 1)
                n_edges = int(n_possible_edges * self.intra_sector_connectivity)
                
                if self.intra_sector_connectivity >= 1.0:
                    # Fully connected within sector
                    for i, stock_i in enumerate(stock_indices):
                        for j, stock_j in enumerate(stock_indices):
                            if i != j:
                                edges.append([offset + stock_i, offset + stock_j])
                else:
                    # Randomly sample edges based on connectivity
                    possible_edges = []
                    for i, stock_i in enumerate(stock_indices):
                        for j, stock_j in enumerate(stock_indices):
                            if i != j:
                                possible_edges.append([offset + stock_i, offset + stock_j])
                    
                    if possible_edges:
                        sampled_indices = np.random.choice(
                            len(possible_edges), 
                            min(n_edges, len(possible_edges)), 
                            replace=False
                        )
                        for idx in sampled_indices:
                            edges.append(possible_edges[idx])
        
        if edges:
            edge_index = torch.LongTensor(edges).T
        else:
            # Return minimal edge if no edges created
            edge_index = torch.LongTensor([[0], [0]])
        
        return edge_index
    
    def create_inter_sector_edges(self, batch_size: int = 1) -> torch.LongTensor:
        """
        Create inter-sector edges (connections between different sectors)
        According to paper: representative stocks from different sectors are connected
        
        Args:
            batch_size: Number of graphs in batch
            
        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        edges = []
        
        for batch_idx in range(batch_size):
            offset = batch_idx * len(self.stocks)
            
            # Connect representative stocks between sectors
            sector_list = list(self.stocks_by_sector.keys())
            
            for i, sector_i in enumerate(sector_list):
                stocks_i = self.stocks_by_sector[sector_i]
                if not stocks_i:
                    continue
                
                # Select representative stock (e.g., first stock or largest by market cap)
                rep_stock_i = stocks_i[0]  # Simple: use first stock as representative
                
                for j, sector_j in enumerate(sector_list):
                    if i >= j:  # Avoid duplicate connections
                        continue
                    
                    stocks_j = self.stocks_by_sector[sector_j]
                    if not stocks_j:
                        continue
                    
                    rep_stock_j = stocks_j[0]
                    
                    # Add edge based on connectivity ratio
                    if np.random.random() < self.inter_sector_connectivity:
                        # Bidirectional edges
                        edges.append([offset + rep_stock_i, offset + rep_stock_j])
                        edges.append([offset + rep_stock_j, offset + rep_stock_i])
        
        if edges:
            edge_index = torch.LongTensor(edges).T
        else:
            # Return minimal edge if no edges created
            edge_index = torch.LongTensor([[0], [0]])
        
        return edge_index
    
    def create_batch_edges(self, 
                          batch_size: int,
                          device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create all graph structures for a batch
        
        Args:
            batch_size: Number of samples in batch
            device: Device to place tensors on
            
        Returns:
            Tuple of (intra_sector_edges, inter_sector_edges, stock_to_sector_mapping)
        """
        # Create edge indices
        edge_index_intra = self.create_intra_sector_edges(batch_size)
        edge_index_inter = self.create_inter_sector_edges(batch_size)
        
        # Create stock-to-sector mapping for batch
        stock_to_sector_batch = self.stock_to_sector_tensor.repeat(batch_size)
        
        # Move to device
        edge_index_intra = edge_index_intra.to(device)
        edge_index_inter = edge_index_inter.to(device)
        stock_to_sector_batch = stock_to_sector_batch.to(device)
        
        return edge_index_intra, edge_index_inter, stock_to_sector_batch
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the graph structure
        
        Returns:
            Dictionary with graph statistics
        """
        # Create sample edges to get statistics
        edge_index_intra = self.create_intra_sector_edges(1)
        edge_index_inter = self.create_inter_sector_edges(1)
        
        stats = {
            'num_stocks': len(self.stocks),
            'num_sectors': len(self.sectors),
            'num_intra_edges': edge_index_intra.shape[1],
            'num_inter_edges': edge_index_inter.shape[1],
            'stocks_per_sector': {
                sector: len(stocks) 
                for sector, stocks in self.stocks_by_sector.items()
            },
            'avg_stocks_per_sector': np.mean([
                len(stocks) for stocks in self.stocks_by_sector.values()
            ])
        }
        
        return stats
    
    def visualize_graph_structure(self):
        """
        Print visual representation of graph structure
        """
        print("\n" + "="*60)
        print("GRAPH STRUCTURE VISUALIZATION")
        print("="*60)
        
        print("\n📊 Sector Distribution:")
        for sector, stocks in self.stocks_by_sector.items():
            stock_symbols = [self.idx_to_stock[idx] for idx in stocks[:5]]  # Show first 5
            if len(stocks) > 5:
                stock_symbols.append(f"... +{len(stocks)-5} more")
            print(f"  {sector:20s}: {len(stocks):3d} stocks | {', '.join(stock_symbols)}")
        
        stats = self.get_graph_statistics()
        
        print(f"\n📈 Graph Statistics:")
        print(f"  Total Stocks:       {stats['num_stocks']}")
        print(f"  Total Sectors:      {stats['num_sectors']}")
        print(f"  Intra-sector Edges: {stats['num_intra_edges']}")
        print(f"  Inter-sector Edges: {stats['num_inter_edges']}")
        print(f"  Avg Stocks/Sector:  {stats['avg_stocks_per_sector']:.1f}")
        
        print("\n" + "="*60)


# Test the graph constructor
if __name__ == "__main__":
    print("Testing Graph Constructor")
    print("-"*60)
    
    # Create sample data
    stocks = ['VCB', 'VNM', 'VIC', 'FPT', 'MBB', 'HPG', 'MSN', 'TCB', 'VHM', 'GAS']
    sector_mapping = {
        'VCB': 'Banking', 'MBB': 'Banking', 'TCB': 'Banking',
        'VNM': 'Consumer', 'MSN': 'Consumer',
        'VIC': 'Real Estate', 'VHM': 'Real Estate',
        'FPT': 'Technology',
        'HPG': 'Materials',
        'GAS': 'Energy'
    }
    
    # Initialize constructor
    graph_constructor = GraphConstructor(
        stocks=stocks,
        sector_mapping=sector_mapping,
        intra_sector_connectivity=1.0,
        inter_sector_connectivity=0.5
    )
    
    # Visualize structure
    graph_constructor.visualize_graph_structure()
    
    # Create batch edges
    batch_size = 2
    edge_index_intra, edge_index_inter, stock_to_sector = graph_constructor.create_batch_edges(batch_size)
    
    print(f"\n✅ Test Results:")
    print(f"  Intra-sector edges shape: {edge_index_intra.shape}")
    print(f"  Inter-sector edges shape: {edge_index_inter.shape}")
    print(f"  Stock-to-sector shape: {stock_to_sector.shape}")
    print(f"  Unique sectors: {torch.unique(stock_to_sector).tolist()}")
    
    print("\n✅ Graph Constructor working correctly!")