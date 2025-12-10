import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EquivariantSelfAttention(nn.Module):
    """Torch implementation of the EMT Equivariant Self-Attention module."""

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        window_size: Optional[int] = None,
        vector_mixing: str = "add",
        vector_gating: bool = True,
    ) -> None:
        super().__init__()
        if vector_mixing not in {"add", "concat"}:
            raise ValueError("vector_mixing must be either 'add' or 'concat'")
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)

        self.alpha_dot = nn.Parameter(torch.tensor(1.0))
        self.alpha_norm = nn.Parameter(torch.tensor(1.0))

        self.gate_proj = nn.Linear(2 * hidden_channels, hidden_channels)
        self.vector_gating = vector_gating
        self.vector_mixing = vector_mixing
        if vector_mixing == "concat":
            self.vec_merge_2 = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.dim() != 4 or x.size(2) != 4:
            raise ValueError("Expected input of shape (B, N, 4, H)")

        B, N, _, H = x.shape
        if H != self.hidden_channels:
            raise ValueError("Input hidden dimension does not match module configuration.")

        x_scalar = x[:, :, 0].contiguous()
        vec = x[:, :, 1:].contiguous()
        vec_res = vec.clone()

        q = self.q_proj(x_scalar)
        k = self.k_proj(x_scalar)
        v = self.v_proj(x_scalar)

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale_factor = 1 / math.sqrt(self.head_dim)

        vec_proj_out = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec_proj_out, self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=-2).contiguous()
        vec_norm = vec.norm(dim=2).contiguous()

        vec_flat = vec.view(B, N, 3, self.num_heads, self.head_dim)
        vec_flat = vec_flat.permute(0, 3, 1, 2, 4).contiguous()
        vec_flat = vec_flat.view(B, self.num_heads, N, 3 * self.head_dim)

        if self.window_size is not None:
            win_len = 2 * self.window_size + 1
            q_local = q.reshape(B * self.num_heads, N, self.head_dim)
            k_local = k.reshape(B * self.num_heads, N, self.head_dim)
            v_local = v.reshape(B * self.num_heads, N, self.head_dim)
            vec_local = vec_flat.reshape(B * self.num_heads, N, 3 * self.head_dim)

            k_padded = F.pad(k_local, (0, 0, self.window_size, self.window_size))
            v_padded = F.pad(v_local, (0, 0, self.window_size, self.window_size))
            vec_padded = F.pad(vec_local, (0, 0, self.window_size, self.window_size))

            k_windows = k_padded.unfold(dimension=1, size=win_len, step=1)
            v_windows = v_padded.unfold(dimension=1, size=win_len, step=1)
            vec_windows = vec_padded.unfold(dimension=1, size=win_len, step=1)

            attn_scores = torch.einsum("bid, biwd -> biw", q_local, k_windows) * scale_factor
            if mask is not None:
                mask_padded = F.pad(mask.float(), (self.window_size, self.window_size), value=0.0)
                mask_windows = mask_padded.unfold(dimension=1, size=win_len, step=1)
                attn_scores = attn_scores.masked_fill(~mask_windows.bool(), float("-inf"))

            attn_weights = attn_scores.softmax(dim=-1).to(q_local.dtype)

            x_agg_local = torch.einsum("biw, biwd -> bid", attn_weights, v_windows)
            x_agg = x_agg_local.view(B, self.num_heads, N, self.head_dim)

            vec_attn_local = torch.einsum("biw, biwd -> bid", attn_weights, vec_windows)
            vec_aggr = vec_attn_local.view(B, self.num_heads, N, 3 * self.head_dim)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if mask is not None:
                attn_scores = attn_scores.masked_fill(
                    ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            attn_weights = attn_scores.softmax(dim=-1).to(q.dtype)
            x_agg = torch.matmul(attn_weights, v)
            vec_aggr = torch.matmul(attn_weights, vec_flat)

        x_out = x_agg.permute(0, 2, 1, 3).contiguous().view(B, N, H)
        vec_aggr = vec_aggr.view(B, self.num_heads, N, 3, self.head_dim)
        vec_aggr = (
            vec_aggr.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(B, N, 3, self.hidden_channels)
        )

        invariants = torch.cat([self.alpha_dot * vec_dot, self.alpha_norm * vec_norm], dim=-1)
        gate = torch.sigmoid(self.gate_proj(invariants)).unsqueeze(2)

        if self.vector_gating:
            if self.vector_mixing == "add":
                vec_combined = gate * vec_aggr + vec_res
            else:
                vec_combined = torch.cat([vec_aggr, vec_res], dim=-1)
                vec_combined = self.vec_merge_2(vec_combined)
        else:
            if self.vector_mixing == "add":
                vec_combined = vec_aggr + vec_res
            else:
                vec_combined = torch.cat([vec_aggr, vec_res], dim=-1)
                vec_combined = self.vec_merge_2(vec_combined)

        o1, o2, o3 = torch.split(self.o_proj(x_out), self.hidden_channels, dim=-1)
        x_updated = vec_dot * o1 + vec_norm * o2 + o3
        x_final = torch.cat([x_updated.unsqueeze(2), vec_combined], dim=2)

        if mask is not None:
            x_final = x_final * mask.unsqueeze(-1).unsqueeze(-1).float()

        return x_final, attn_weights

