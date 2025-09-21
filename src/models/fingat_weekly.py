import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AttentiveGRU(nn.Module):
    """
    Attentive GRU over a sequence (batch_first=True).
    Input: (B, T, D)
    Output: (B, H)
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.1, attn_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_w1 = nn.Linear(hidden_size, attn_size)
        self.attn_w2 = nn.Linear(attn_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)
        a = torch.tanh(self.attn_w1(out))  # (B,T,A)
        a = self.dropout(a)
        score = self.attn_w2(a).squeeze(-1)  # (B,T)
        if lengths is not None:
            T = out.size(1)
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            score = score.masked_fill(mask, float('-inf'))
        alpha = torch.softmax(score, dim=1)  # (B,T)
        ctx = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (B,H)
        return ctx


class GraphAttentionLayer(nn.Module):
    """
    Lightweight GAT layer using dense adjacency A (no self loops by default).
    Input: X (B, N, Din), A (B, N, N) in {0,1}
    Output: (B, N, Dout)
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B,N,Din), adj: (B,N,N)
        B, N, _ = x.shape
        h = self.lin(x)  # (B,N,Dout*H)
        h = h.view(B, N, self.heads, self.out_dim)  # (B,N,H,Dh)
        # Compute attention logits e_ij for each head: a^T [h_i || h_j] ≈ (a_src·h_i) + (a_dst·h_j)
        e_src = torch.einsum('bnhd,hd->bnh', h, self.a_src)  # (B,N,H)
        e_dst = torch.einsum('bnhd,hd->bnh', h, self.a_dst)  # (B,N,H)
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (B,N,N,H)
        e = self.leakyrelu(e)
        # Mask with adjacency: set to -inf where no edge
        mask = (adj == 0).unsqueeze(-1)  # (B,N,N,1)
        e = e.masked_fill(mask, float('-inf'))
        # Softmax over neighbors j
        alpha = torch.softmax(e, dim=2)  # (B,N,N,H)
        alpha = self.dropout(alpha)
        # Weighted sum of neighbor features
        out = torch.einsum('bnnH,bnhd->bnhd', alpha, h)  # (B,N,H,Dh)
        out = out.reshape(B, N, self.heads * self.out_dim)  # (B,N,H*Dh)
        return out


class FinGATWeekly(nn.Module):
    """
    FinGAT with weekly short-term module, long-term dual streams, and sector-level.
    Expects inputs from MultiAssetWindowDataset collate: x (B,N,W,F), adj (B,2,N,N), sectors list per sample.
    """
    def __init__(self, input_dim: int, week_size: int = 5, weeks: int = 3,
                 d_week: int = 64, d_intra: int = 64, d_long_a: int = 64, d_long_g: int = 64,
                 d_sector: int = 64, heads_intra: int = 4, heads_inter: int = 4,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.week_size = week_size
        self.weeks = weeks
        # Stock-level short term a_i
        self.week_gru = AttentiveGRU(input_dim, d_week, num_layers=1, dropout=dropout, attn_size=d_week)
        # Intra-sector GAT per week to get g_i
        self.gat_intra = GraphAttentionLayer(d_week, d_intra // heads_intra, heads=heads_intra, dropout=dropout)
        # Long-term dual attentive GRUs over sequences {a_i} and {g_i}
        self.long_a = AttentiveGRU(d_week, d_long_a, num_layers=1, dropout=dropout, attn_size=d_long_a)
        self.long_g = AttentiveGRU(d_intra, d_long_g, num_layers=1, dropout=dropout, attn_size=d_long_g)
        # Sector-level: inter-sector GAT over pooled sector embeddings per week, then attentive GRU over 3 weeks
        self.gat_inter = GraphAttentionLayer(d_intra, d_sector // heads_inter, heads=heads_inter, dropout=dropout)
        self.sector_long = AttentiveGRU(d_sector, d_sector, num_layers=1, dropout=dropout, attn_size=d_sector)
        # Fusion and heads
        self.dropout = nn.Dropout(dropout)
        fusion_dim = d_long_a + d_long_g + d_sector
        self.fusion = nn.Linear(fusion_dim, fusion_dim)
        self.cls_head = nn.Linear(fusion_dim, num_classes)
        # Separate heads for ranking score and return regression (MAE)
        self.rank_head = nn.Linear(fusion_dim, 1)
        self.ret_head = nn.Linear(fusion_dim, 1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch['x']  # (B,N,W,F)
        adj = batch['adj']  # (B,2,N,N) [0]=intra, [1]=inter sectors (for assets, not sectors)
        mask = batch['asset_mask']  # (B,N)
        B, N, W, F = x.shape
        weeks = self.weeks
        assert W % self.week_size == 0 and W // self.week_size == weeks, "Window should be 3 weeks of 5 days"
        # Reshape into weeks
        xw = x.view(B, N, weeks, self.week_size, F)  # (B,N,3,5,F)
        # Compute week-level a_i per asset: apply week_gru to each week slice
        xw_flat = xw.reshape(B * N * weeks, self.week_size, F)
        a_flat = self.week_gru(xw_flat)  # (B*N*weeks, d_week)
        # Restore
        d_week = a_flat.size(-1)
        a = a_flat.view(B, N, weeks, d_week)  # (B,N,3,d_week)
        # Intra-sector GAT per week using adjacency channel 0
        adj_intra = adj[:, 0, :, :]  # (B,N,N)
        g_list = []
        for t in range(weeks):
            a_t = a[:, :, t, :]  # (B,N,d_week)
            g_t = self.gat_intra(a_t, adj_intra)  # (B,N,d_intra)
            g_list.append(g_t)
        g = torch.stack(g_list, dim=2)  # (B,N,3,d_intra)
        # Long-term attentive GRUs over sequences length=3 per asset
        a_seq = a.permute(0,1,2,3).reshape(B*N, weeks, d_week)
        g_seq = g.permute(0,1,2,3).reshape(B*N, weeks, g.size(-1))
        tau_a = self.long_a(a_seq).view(B, N, -1)  # (B,N,d_long_a)
        tau_g = self.long_g(g_seq).view(B, N, -1)  # (B,N,d_long_g)
        # Sector-level pooling per week from g_t, then inter-sector GAT on sector graph, then attentive GRU over 3 weeks
        # Build sector pooling masks per batch sample from sectors list
        sectors = batch['sectors']  # List[List[str]] length B
        sector_embeddings_seq = []  # list of length weeks, each (B,S_t,d_intra)
        sector_index_maps = []      # list of length weeks, each list of dicts per sample mapping sector->index
        for t in range(weeks):
            g_t = g[:, :, t, :]  # (B,N,d_intra)
            pooled_list = []
            index_maps = []
            for b in range(B):
                valid = mask[b]  # (N,)
                sec_b = sectors[b]
                # Collect sector indices for valid assets
                sec_to_idx = {}
                for i, ok in enumerate(valid.tolist()):
                    if not ok:
                        continue
                    s = sec_b[i]
                    if s not in sec_to_idx:
                        sec_to_idx[s] = []
                    sec_to_idx[s].append(i)
                index_maps.append(sec_to_idx)
                # Max-pool within each sector
                pooled = []
                for s, idxs in sec_to_idx.items():
                    idx_tensor = torch.tensor(idxs, device=g_t.device, dtype=torch.long)
                    pooled.append(g_t[b, idx_tensor, :].max(dim=0).values)
                if len(pooled) == 0:
                    pooled_tensor = torch.zeros((0, g_t.size(-1)), device=g_t.device)
                else:
                    pooled_tensor = torch.stack(pooled, dim=0)  # (S_b, d_intra)
                pooled_list.append(pooled_tensor)
            # Pad sectors across batch to same S_max
            S_max = max([p.shape[0] for p in pooled_list]) if pooled_list else 0
            if S_max == 0:
                sector_embeddings_seq.append(torch.zeros((B, 0, g_t.size(-1)), device=g_t.device))
                sector_index_maps.append(index_maps)
                continue
            padded = g_t.new_zeros((B, S_max, g_t.size(-1)))
            for b, p in enumerate(pooled_list):
                if p.shape[0] > 0:
                    padded[b, :p.shape[0], :] = p
            sector_embeddings_seq.append(padded)  # (B,S_max,d_intra)
            sector_index_maps.append(index_maps)
        # Inter-sector GAT per week on fully-connected sector graph
        sector_after_gat = []
        for t in range(weeks):
            z_t = sector_embeddings_seq[t]  # (B,S,d_intra)
            B_, S, D = z_t.shape
            if S == 0:
                sector_after_gat.append(z_t.new_zeros((B_, S, D)))
                continue
            # Build fully-connected adjacency without self loops
            adj_sec = z_t.new_ones((B_, S, S))
            adj_sec = adj_sec.fill_(1.0) - torch.eye(S, device=z_t.device).unsqueeze(0)
            s_t = self.gat_inter(z_t, adj_sec)  # (B,S,d_sector)
            sector_after_gat.append(s_t)
        # Attentive GRU over sector sequence length=3 per sector node separately is complex due to S varying across batch/time.
        # We'll aggregate per batch sample by averaging week representations before GRU fallback.
        # To follow spec closer, we will run attentive GRU by stacking weeks with padding per batch b.
        tau_sector_batch = []  # list length B, each (S_b, d_sector)
        for b in range(B):
            # Collect [S_t_b, d_sector] per t, pad to S_max_b across t
            S_b_list = [sector_after_gat[t][b].shape[0] for t in range(weeks)]
            S_max_b = max(S_b_list)
            if S_max_b == 0:
                tau_sector_batch.append(sector_after_gat[0].new_zeros((0, sector_after_gat[0].size(-1))))
                continue
            seq_tensor = []  # (weeks, S_max_b, d_sector)
            for t in range(weeks):
                z = sector_after_gat[t][b]  # (S_t, d_sector)
                pad = z.new_zeros((S_max_b, z.size(-1)))
                if z.shape[0] > 0:
                    pad[:z.shape[0], :] = z
                seq_tensor.append(pad)
            seq_tensor = torch.stack(seq_tensor, dim=0)  # (weeks, S_max_b, d_sector)
            # Apply GRU over weeks for each sector slot independently
            # Reshape to (S_max_b, weeks, d_sector)
            seq_swapped = seq_tensor.permute(1, 0, 2)
            tau = self.sector_long(seq_swapped)  # (S_max_b, d_sector)
            tau_sector_batch.append(tau)
        # Map each asset to its sector embedding (first slot matching its sector key order per index_maps[0])
        tau_sector_list = []
        for b in range(B):
            valid = mask[b]
            N_b = int(valid.sum().item())
            tau_b = tau_sector_batch[b]  # (S_max_b, d_sector) or (0, d_sector)
            if tau_b.shape[0] == 0:
                tau_assets = tau_b.new_zeros((N_b, tau_b.size(-1)))
            else:
                # Build sector order from index_maps of t=0 for consistency
                sec_to_idx = sector_index_maps[0][b]
                sector_keys = list(sec_to_idx.keys())
                # Map sector name -> position in tau_b (assumed same order as pooling loop)
                # In pooling loop, per b, sectors iterated in insertion order of dict; replicate here
                # Build mapping by iterating again
                s_pos = {}
                pos = 0
                for s in sector_keys:
                    s_pos[s] = pos
                    pos += 1
                # Assign per asset
                asset_emb = []
                for i, ok in enumerate(valid.tolist()):
                    if not ok:
                        continue
                    s = sectors[b][i]
                    p = s_pos.get(s, 0)
                    asset_emb.append(tau_b[p])
                tau_assets = torch.stack(asset_emb, dim=0)
            # Pad back to N with zeros
            out_pad = torch.zeros((N, tau_assets.size(-1)), device=x.device)
            out_pad[:tau_assets.shape[0], :] = tau_assets
            tau_sector_list.append(out_pad)
        tau_sector = torch.stack(tau_sector_list, dim=0)  # (B,N,d_sector)
        # Fusion
        fused = torch.cat([tau_g, tau_a, tau_sector], dim=-1)  # (B,N,D)
        fused = self.dropout(F.relu(self.fusion(fused)))
        # Heads
        movement_logits = self.cls_head(fused)  # (B,N,C)
        ranking_scores = self.rank_head(fused).squeeze(-1)  # (B,N)
        ret_pred = self.ret_head(fused).squeeze(-1)         # (B,N)
        return {
            'movement_logits': movement_logits,
            'movement_probs': torch.softmax(movement_logits, dim=-1),
            'ranking_scores': ranking_scores,
            'ret_pred': ret_pred
        }
