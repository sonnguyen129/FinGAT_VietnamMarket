"""Graph construction — intra-sector and inter-sector edges.

Both graphs are fully connected:
  - Intra-sector: every pair of stocks within the same sector
  - Inter-sector: every pair of sectors
"""

import os

import numpy as np
import torch


def build_intra_sector_edges(tickers: list, sector_mapping: dict) -> torch.Tensor:
    """Build fully-connected graph within each sector.

    Returns edge_index tensor of shape [2, num_edges] in PyG format.
    """
    # Group ticker indices by sector
    sectors = {}
    for i, ticker in enumerate(tickers):
        if ticker in sector_mapping:
            sector = sector_mapping[ticker]['industry']
            sectors.setdefault(sector, []).append(i)

    src, dst = [], []
    for sector, indices in sectors.items():
        for i in indices:
            for j in indices:
                if i != j:
                    src.append(i)
                    dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Validate: E_intra = sum(n_c * (n_c - 1)) for each sector c
    expected = sum(len(idx) * (len(idx) - 1) for idx in sectors.values())
    assert edge_index.shape[1] == expected, \
        f"Edge count mismatch: {edge_index.shape[1]} vs expected {expected}"

    print(f"Intra-sector edges: {edge_index.shape[1]} "
          f"({len(sectors)} sectors)")
    return edge_index


def build_inter_sector_edges(num_sectors: int) -> torch.Tensor:
    """Build fully-connected graph between all sectors.

    Returns edge_index tensor of shape [2, num_sectors * (num_sectors - 1)].
    """
    src, dst = [], []
    for i in range(num_sectors):
        for j in range(num_sectors):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    expected = num_sectors * (num_sectors - 1)
    assert edge_index.shape[1] == expected

    print(f"Inter-sector edges: {edge_index.shape[1]} "
          f"({num_sectors} sectors)")
    return edge_index


def build_sector_assignments(tickers: list, sector_mapping: dict) -> torch.Tensor:
    """Map each stock index to its sector index.

    Returns tensor of shape [num_stocks] with sector IDs.
    """
    # Build sector list from the mapping
    sector_set = set()
    for t in tickers:
        if t in sector_mapping:
            sector_set.add(sector_mapping[t]['industry'])
    sector_list = sorted(sector_set)
    sector_to_idx = {s: i for i, s in enumerate(sector_list)}

    assignments = torch.tensor([
        sector_to_idx[sector_mapping[t]['industry']]
        for t in tickers if t in sector_mapping
    ], dtype=torch.long)

    return assignments, sector_list


def build_fully_connected_edges(num_nodes: int) -> torch.Tensor:
    """Build fully-connected graph for num_nodes nodes (used by FinGAT-NT)."""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def save_edges(intra_edges: torch.Tensor, inter_edges: torch.Tensor,
               sector_assignments: torch.Tensor,
               path: str = "data/cache"):
    """Save all graph data as .npy files."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'hose_inner_edge.npy'), intra_edges.numpy())
    np.save(os.path.join(path, 'hose_outer_edge.npy'), inter_edges.numpy())
    np.save(os.path.join(path, 'hose_sector_assignments.npy'),
            sector_assignments.numpy())
    print(f"Edges saved to {path}/")


def load_edges(path: str = "data/cache") -> tuple:
    """Load graph data from .npy files.

    Returns (intra_edge_index, inter_edge_index, sector_assignments).
    """
    intra = torch.tensor(
        np.load(os.path.join(path, 'hose_inner_edge.npy')), dtype=torch.long)
    inter = torch.tensor(
        np.load(os.path.join(path, 'hose_outer_edge.npy')), dtype=torch.long)
    assignments = torch.tensor(
        np.load(os.path.join(path, 'hose_sector_assignments.npy')),
        dtype=torch.long)
    return intra, inter, assignments
