import torch
import torch.nn as nn


class LocalQKConv(nn.Module):
    """
    Local latent Q/K convolution that uses learned equivariant vectors.

    Args:
        hidden_channels: int, size of the scalar (and vector) channel dimension H.
        window_size:     int, sequence half-window w. Each token i connects
                         to j in [i-w, ..., i+w], j != i.
        eps:             float, numerical epsilon for safe normalization.

    Inputs:
        x_scalar: (B, N, H)  scalar features per token (e.g. x[:, :, 0]).
        vec:      (B, N, 3, H)  vector features per token/channel (e.g. x[:, :, 1:]).

    Outputs:
        q, k: (B, N, H)  local Q/K features built from sequence-local
                         latent geometry in vec.
    """

    def __init__(self, hidden_channels, window_size=3, eps=1e-8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.window_size = window_size
        self.eps = eps
        self._edge_cache = {} 

        H = hidden_channels
        self.w_angle_q = nn.Parameter(torch.zeros(H))
        self.w_dih_q   = nn.Parameter(torch.zeros(H))
        self.b_q       = nn.Parameter(torch.zeros(H))

        self.w_angle_k = nn.Parameter(torch.zeros(H))
        self.w_dih_k   = nn.Parameter(torch.zeros(H))
        self.b_k       = nn.Parameter(torch.zeros(H))


    def _make_seq_edges(self, N: int, device: torch.device) -> torch.Tensor:
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

    def _compute_qk(self, x_scalar, vec):

        assert x_scalar.dim() == 3, "x_scalar should be (B, N, H)"
        assert vec.dim() == 4, "vec should be (B, N, 3, H)"
        B, N, H = x_scalar.shape
        device = x_scalar.device
        edge_index = self._make_seq_edges(N, device)
        src, dst = edge_index

        # per-edge directions e_ij[c] = normalize(vec_j[c] - vec_i[c]).
        v_i = vec[:, src, :, :]  # (B, E, 3, H)
        v_j = vec[:, dst, :, :]
        b_ij = v_j - v_i                       # (B,E,3,H)
        norm_b = b_ij.norm(dim=2, keepdim=True).clamp_min(self.eps)
        e_ij = b_ij / norm_b 

        # u_node[i,c] = sum_{j in N(i)} e_ij[c]
        B, E, _, H = e_ij.shape
        u_node_flat = torch.zeros(B, N, 3 * H, device=device, dtype=vec.dtype)
        e_flat = e_ij.reshape(B, E, 3 * H)  # (B, E, 3H)

        # src: (E,), indices along node dimension
        u_node_flat.index_add_(dim=1, index=src, source=e_flat)
        u_node = u_node_flat.view(B, N, 3, H)  # (B, N, 3, H)

        # Gather direction units at source/dest nodes for each edge
        u_i = u_node[:, src, :, :]  # (B, E, 3, H)
        u_j = u_node[:, dst, :, :]

        # ang_ij[c] = cos(angle between u_i[c] and e_ij[c])
        dot_ui_e = (u_i * e_ij).sum(dim=2)                     # (B, E, H)
        norm_ui = u_i.norm(dim=2, keepdim=False).clamp_min(self.eps)  # (B,E,H)
        ang_ij = (dot_ui_e / norm_ui).clamp(-1.0, 1.0)

        # Project u_i, u_j onto the plane orthogonal to e_ij, then use
        # normalized dot product between the projections.
        proj_i = dot_ui_e.unsqueeze(2) * e_ij                  # (B, E, 3, H)
        dot_uj_e = (u_j * e_ij).sum(dim=2)                     # (B, E, H)
        proj_j = dot_uj_e.unsqueeze(2) * e_ij                  # (B, E, 3, H)

        u_i_perp = u_i - proj_i
        u_j_perp = u_j - proj_j

        dot_perp = (u_i_perp * u_j_perp).sum(dim=2)            # (B, E, H)
        norm_perp_i = u_i_perp.norm(dim=2).clamp_min(self.eps)
        norm_perp_j = u_j_perp.norm(dim=2).clamp_min(self.eps)
        dih_ij = (dot_perp / (norm_perp_i * norm_perp_j)).clamp(-1.0, 1.0)

        # Broadcast learned per-channel weights to (B, E, H)
        w_a_q = self.w_angle_q.view(1, 1, H)
        w_d_q = self.w_dih_q.view(1, 1, H)
        b_q   = self.b_q.view(1, 1, H)

        w_a_k = self.w_angle_k.view(1, 1, H)
        w_d_k = self.w_dih_k.view(1, 1, H)
        b_k   = self.b_k.view(1, 1, H)

        g_q = torch.sigmoid(ang_ij * w_a_q + dih_ij * w_d_q + b_q)  # (B, E, H)
        g_k = torch.sigmoid(ang_ij * w_a_k + dih_ij * w_d_k + b_k)  # (B, E, H)

        s_j = x_scalar[:, dst, :]                     # (B, E, H)

        m_q = g_q * s_j                               # (B, E, H)
        m_k = g_k * s_j

        q = torch.zeros_like(x_scalar)                # (B, N, H)
        k = torch.zeros_like(x_scalar)

        q.index_add_(dim=1, index=src, source=m_q)
        k.index_add_(dim=1, index=src, source=m_k)

        return q, k
    
    def forward(self, x_scalar, vec):
        return self._compute_qk(x_scalar, vec)

    def q_conv(self, x_scalar, vec):
        q, _ = self.forward(x_scalar, vec)
        return q

    def k_conv(self, x_scalar, vec):
        _, k = self.forward(x_scalar, vec)
        return k
