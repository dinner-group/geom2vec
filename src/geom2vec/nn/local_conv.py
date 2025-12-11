import torch
import torch.nn as nn
from torch import Tensor


import torch
import torch.nn as nn
from torch import Tensor


class LocalQKConv(nn.Module):
    """
    Local latent Q/K convolution that uses learned equivariant vectors.

    Geometry is computed in a reduced 'geom_channels' subspace of the
    vector channels, then used to produce per-edge scalar gates for Q/K.

    Args:
        hidden_channels: int, size of scalar & vector channel dim H.
        window_size:     int, sequence half-window w. Each token i connects
                         to j in [i-w, ..., i+w], j != i.
        eps:             float, numerical epsilon for safe normalization.
        chiral:          bool, if True use signed dihedral (pseudoscalar)
                         and break reflection equivariance (SO(3) only).
        geom_channels:   int or None. Size G of geometry subspace. If None,
                         defaults to 1 (fully pooled); typical values: 8, 16.
        mlp_hidden:      int, hidden size for small edge MLPs.

    Inputs:
        x_scalar: (B, N, H)     scalar features per token (x[:, :, 0])
        vec:      (B, N, 3, H)  vector features per token/channel (x[:, :, 1:])

    Outputs:
        q, k: (B, N, H)  local Q/K features built from sequence-local
                         latent geometry in vec.
    """

    def __init__(
        self,
        hidden_channels: int,
        window_size: int = 3,
        eps: float = 1e-8,
        chiral: bool = False,
        geom_channels: int = 16,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.window_size = window_size
        self.eps = eps
        self.chiral = chiral
        self.geom_channels = geom_channels or 1   # G
        self._edge_cache = {}

        H = hidden_channels
        G = self.geom_channels

        # Compress vector channel dim H -> geom dim G, shared across 3 coords.
        # This is equivariant: we only mix channels, not xyz coordinates.
        self.vec_to_geom = nn.Linear(H, G, bias=False)

        # Edge MLPs: take per-geom-channel invariants [ang, cos_phi(, sin_phi)] -> gate in geom space
        in_dim = 3 if chiral else 2
        mlp_hidden = max(mlp_hidden, 4)

        self.edge_mlp_geom_q = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.edge_mlp_geom_k = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

        self.expand_q = nn.Linear(G, H)
        self.expand_k = nn.Linear(G, H)

    def _make_seq_edges(self, N: int, device: torch.device) -> torch.Tensor:
        """Local sequence edges with window_size."""
        w = self.window_size
        if w <= 0 or N <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        key = (N, device.type, device.index if device.index is not None else -1)
        if key in self._edge_cache:
            return self._edge_cache[key]

        i = torch.arange(N, device=device)                # (N,)
        offsets = torch.arange(-w, w + 1, device=device)  # [-w..w]
        offsets = offsets[offsets != 0]                   # remove 0

        i_mat = i.view(N, 1)                              # (N,1)
        j_mat = i_mat + offsets.view(1, -1)               # (N,2w)

        mask = (j_mat >= 0) & (j_mat < N)
        src = i_mat.expand_as(j_mat)[mask]
        dst = j_mat[mask]

        edge_index = torch.stack([src, dst], dim=0).long()  # (2,E)
        self._edge_cache[key] = edge_index
        return edge_index

    # ------------------------------------------------------------------ #

    def _compute_qk(self, x_scalar: Tensor, vec: Tensor):
        assert x_scalar.dim() == 3, "x_scalar should be (B, N, H)"
        assert vec.dim() == 4, "vec should be (B, N, 3, H)"

        B, N, H = x_scalar.shape
        assert H == self.hidden_channels, "hidden_channels mismatch"
        device = x_scalar.device

        edge_index = self._make_seq_edges(N, device)
        if edge_index.numel() == 0:
            q = torch.zeros_like(x_scalar)
            k = torch.zeros_like(x_scalar)
            return q, k

        src, dst = edge_index  # (E,)
        E = src.numel()
        G = self.geom_channels

        vec_flat = vec.reshape(B * N * 3, H)              # (B*N*3, H)
        vec_geom_flat = self.vec_to_geom(vec_flat)        # (B*N*3, G)
        vec_geom = vec_geom_flat.view(B, N, 3, G)         

        v_i = vec_geom[:, src, :, :]                     
        v_j = vec_geom[:, dst, :, :]                      
        b_ij = v_j - v_i                                  # (B, E, 3, G)

        norm_b = b_ij.norm(dim=2, keepdim=True).clamp_min(self.eps)  
        e_ij = b_ij / norm_b                                          

        u_node_flat = torch.zeros(B, N, 3 * G, device=device, dtype=vec.dtype)
        e_flat = e_ij.reshape(B, E, 3 * G)                           
        u_node_flat.index_add_(dim=1, index=src, source=e_flat)       
        u_node = u_node_flat.view(B, N, 3, G)                       

        u_i = u_node[:, src, :, :]                                    # (B, E, 3, G)
        u_j = u_node[:, dst, :, :]

        dot_ui_e = (u_i * e_ij).sum(dim=2)                           
        norm_ui = u_i.norm(dim=2).clamp_min(self.eps)                
        ang_ij = (dot_ui_e / norm_ui).clamp(-1.0, 1.0)                

        # project u_i, u_j to plane perpendicular to e_ij
        proj_i = dot_ui_e.unsqueeze(2) * e_ij                        
        dot_uj_e = (u_j * e_ij).sum(dim=2)                          
        proj_j = dot_uj_e.unsqueeze(2) * e_ij                         

        u_i_perp = u_i - proj_i                                   
        u_j_perp = u_j - proj_j                                     

        dot_perp = (u_i_perp * u_j_perp).sum(dim=2)                  
        norm_perp_i = u_i_perp.norm(dim=2).clamp_min(self.eps)      
        norm_perp_j = u_j_perp.norm(dim=2).clamp_min(self.eps)       

        denom_cos = (norm_perp_i * norm_perp_j).clamp_min(self.eps)
        dih_cos = (dot_perp / denom_cos).clamp(-1.0, 1.0)             

        if self.chiral:
            # pseudoscalar sin via triple product (u_i_perp cross u_j_perp) dot e_ij
            cross_ij = torch.cross(u_i_perp, u_j_perp, dim=2)        
            triple = (cross_ij * e_ij).sum(dim=2)                   

            norm_e = e_ij.norm(dim=2).clamp_min(self.eps)            
            denom_sin = (norm_perp_i * norm_perp_j * norm_e).clamp_min(self.eps)
            dih_sin = (triple / denom_sin).clamp(-1.0, 1.0)          
            feats = torch.stack([ang_ij, dih_cos, dih_sin], dim=-1)   # (B, E, G, 3)
        else:
            feats = torch.stack([ang_ij, dih_cos], dim=-1)            # (B, E, G, 2)

        B_, E_, G_, D = feats.shape
        assert B_ == B and E_ == E and G_ == G
        feats_flat = feats.view(B * E * G, D)                         # (B*E*G, D)

        g_geom_q = self.edge_mlp_geom_q(feats_flat).view(B, E, G)    # (B, E, G)
        g_geom_k = self.edge_mlp_geom_k(feats_flat).view(B, E, G)

        g_full_q_logits = self.expand_q(g_geom_q.view(B * E, G)).view(B, E, H)
        g_full_k_logits = self.expand_k(g_geom_k.view(B * E, G)).view(B, E, H)

        g_q = torch.sigmoid(g_full_q_logits)                         
        g_k = torch.sigmoid(g_full_k_logits)                        

        s_j = x_scalar[:, dst, :]                                   

        m_q = g_q * s_j                                              
        m_k = g_k * s_j

        q = torch.zeros_like(x_scalar)                                
        k = torch.zeros_like(x_scalar)

        q.index_add_(dim=1, index=src, source=m_q)
        k.index_add_(dim=1, index=src, source=m_k)

        return q, k


    def forward(self, x_scalar: Tensor, vec: Tensor):
        """Return (q, k)."""
        return self._compute_qk(x_scalar, vec)

    def q_conv(self, x_scalar: Tensor, vec: Tensor):
        q, _ = self._compute_qk(x_scalar, vec)
        return q

    def k_conv(self, x_scalar: Tensor, vec: Tensor):
        _, k = self._compute_qk(x_scalar, vec)
        return k