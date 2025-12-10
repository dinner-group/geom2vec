import math
import triton

import torch
import torch.nn.functional as F
import torch.nn as nn

from .forward_kernel import fused_attention_forward
from .backward_kernel import (
    launch_preprocess_kernel,
    launch_dq_calculation_kernel,
    launch_dkdv_calculation_kernel,
)

# disable TF32 for Triton kernels
torch.backends.cuda.matmul.allow_tf32 = True

# triton disable TF32 for all kernel


class EquivariantSelfAttentionTritonFunction(torch.autograd.Function):
    """
    Autograd entry-point that calls the Triton fused forward kernel and the
    two fused backward kernels (dQ   and  dK/dV).
    """

    # ─────────────────────────── forward ────────────────────────────────
    @staticmethod
    def forward(ctx, q, k, v, vec_v_unpadded, qk_scale):
        # Triton forward now returns   O_s,  O_v,  M,  L,  O_v_padded
        O_s, O_v, M, L, O_v_pad = fused_attention_forward(
            q, k, v, vec_v_unpadded
        )

        # pad vec_v so backward kernels always see power-of-2 vector dim
        V_orig = vec_v_unpadded.shape[-1]
        V_pad  = O_v_pad.shape[-1]                  # already next_pow2
        vec_v_pad = (
            F.pad(vec_v_unpadded, (0, V_pad - V_orig))
            if V_pad != V_orig else vec_v_unpadded
        ).contiguous()

        # save everything required by backward
        ctx.save_for_backward(
            q, k, v, vec_v_pad,          # inputs (vec_v is padded)
            M.contiguous(), L.contiguous(),
            O_s.contiguous(), O_v_pad.contiguous()
        )
        ctx.qk_scale      = qk_scale
        ctx.orig_vec_dim  = V_orig
        return O_s, O_v                           # UNPADDED vector output

    # ─────────────────────────── backward ───────────────────────────────
    @staticmethod
    def backward(ctx, dO_s, dO_v_unpadded):
        # unpack saved tensors
        (q, k, v, vec_v_pad,
         M, L,
         O_s, O_v_pad) = ctx.saved_tensors

        # pad dO_v to the same power-of-2 width as vec_v_pad
        V_pad  = vec_v_pad.shape[-1]
        pad_sz = V_pad - ctx.orig_vec_dim
        dO_v_pad = (
            F.pad(dO_v_unpadded, (0, pad_sz))
            if pad_sz else dO_v_unpadded
        ).contiguous()

        # ── Delta (row-wise ⟨O, dO⟩) ───────────────────────────────────
        Delta = torch.empty_like(M, dtype=torch.float32)
        launch_preprocess_kernel(O_s, O_v_pad, dO_s, dO_v_pad, Delta)

        # allocate output grads
        dq  = torch.empty_like(q)
        dk  = torch.empty_like(k)
        dv  = torch.empty_like(v)
        dvec_pad = torch.empty_like(vec_v_pad)

        # ── dK & dV ─────────────────────────────────────────────────────
        launch_dkdv_calculation_kernel(
            q, k, v, vec_v_pad,
            dO_s, dO_v_pad,
            M, Delta,
            L_out=L,
            sm_scale=ctx.qk_scale,
            DK_out=dk, DV_s_out=dv, DV_v_out=dvec_pad
        )

        # ── dQ ──────────────────────────────────────────────────────────
        launch_dq_calculation_kernel(
            q, k, v, vec_v_pad,
            dO_s, dO_v_pad,
            M, Delta,
            L_out=L,
            sm_scale=ctx.qk_scale,
            DQ_out=dq)

        # un-pad vector gradient
        dvec = dvec_pad[..., :ctx.orig_vec_dim].contiguous()
        return dq, dk, dv, dvec, None


class EquivariantSelfAttentionTriton(nn.Module):
    """
    Equivariant Self-Attention using Triton fused kernel for full attention.
    Processes inputs of shape (B, N, 4, hidden_channels).
    NOTE: This version currently only supports full attention (window_size=None).
    NOTE: Masking during attention calculation is not implemented in the kernel.
          Only final output masking based on query positions is supported.
    """

    def __init__(self, hidden_channels, num_heads, window_size=None,
                 vector_mixing='add', vector_gating=True):
        super().__init__()
        # --- Initialize parameters exactly like the original class ---
        assert vector_mixing in ['add', 'concat'], "vector_mixing must be either 'add' or 'concat'"
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"

        # Check if window_size is used (currently not supported by Triton kernel)
        if window_size is not None:
            raise NotImplementedError("Sliding window attention is not yet implemented in the Triton kernel version.")
        self.window_size = window_size

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

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
        if vector_mixing == 'concat':
            self.vec_merge_2 = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, N, 4, hidden_channels)
            mask: Optional boolean tensor of shape (B, N) where True indicates valid positions.
                  NOTE: Only used for final output masking, not during attention.
        Returns:
            x_final: Tensor of shape (B, N, 4, hidden_channels)
            attn_weights: None (Triton kernel does not return attention weights)
        """
        B, N, _, H = x.shape
        assert H == self.hidden_channels
        assert self.window_size is None, "Triton kernel only supports full attention"

        # --- Preprocessing (Identical to original) ---
        x_scalar = x[:, :, 0].contiguous()
        vec = x[:, :, 1:].contiguous()
        vec_res = vec.clone()

        q = self.q_proj(x_scalar)
        k = self.k_proj(x_scalar)
        v = self.v_proj(x_scalar)

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, H, N, Dh)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, H, N, Dh)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, H, N, Dh)

        # Vector processing (Identical to original)
        vec_proj_out = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec_proj_out, self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=-2).contiguous()
        vec_norm = vec.norm(dim=2).contiguous()

        # Prepare vec_v for the Triton kernel (vector branch value)
        # Shape required: (B, H, N, 3*Dh)
        vec_flat = vec.view(B, N, 3, self.num_heads, self.head_dim) # (B, N, 3, H, Dh)
        vec_flat = vec_flat.permute(0, 3, 1, 2, 4) # (B, H, N, 3, Dh)
        vec_v_kernel = vec_flat.contiguous().view(B, self.num_heads, N, 3 * self.head_dim) # (B, H, N, 3*Dh)

        # --- Call Fused Triton Kernel ---
        # fused_attention_forward expects (B, H, N, Dim) inputs
        # Outputs: out_scalar (B, H, N, Dh), out_vector (B, H, N, 3*Dh)
        # if mask is not None:
        #      print("WARNING: Mask input provided to EquivariantSelfAttentionTriton, "
        #            "but the Triton kernel does not currently support attention masking. "
        #            "Only final output masking will be applied.")

        # Ensure inputs are contiguous if the kernel wrapper expects them
        # (Our wrapper currently calculates strides assuming contiguous)
        # out_scalar, out_vector = fused_attention_forward(
        #     q.contiguous(), k.contiguous(), v.contiguous(), vec_v_kernel.contiguous()
        # )

        # apply the EquivariantSelfAttentionTritonFunction
        qk_scale = 1.0 / math.sqrt(self.head_dim)
        out_scalar, out_vector = EquivariantSelfAttentionTritonFunction.apply(
            q.contiguous(), k.contiguous(), v.contiguous(), vec_v_kernel.contiguous(),
            qk_scale
        )

        # --- Map Kernel Outputs ---
        # out_scalar is the aggregated scalar value, equivalent to x_agg
        # Shape: (B, H, N, Dh)
        x_agg = out_scalar

        # out_vector is the aggregated vector value, equivalent to vec_aggr
        # Shape: (B, H, N, 3*Dh)
        vec_aggr = out_vector

        # --- Postprocessing (Identical to original, using x_agg and vec_aggr) ---
        x_out = x_agg.permute(0, 2, 1, 3).contiguous().view(B, N, H) # (B, N, H)
        vec_aggr = vec_aggr.view(B, self.num_heads, N, 3, self.head_dim) \
                        .permute(0, 2, 3, 1, 4) \
                        .contiguous() \
                        .view(B, N, 3, H)

        invariants = torch.cat([self.alpha_dot * vec_dot, self.alpha_norm * vec_norm], dim=-1)
        gate = torch.sigmoid(self.gate_proj(invariants)).unsqueeze(2)

        if self.vector_gating:
            if self.vector_mixing == 'add':
                vec_combined = gate * vec_aggr + vec_res
            elif self.vector_mixing == 'concat':
                vec_combined = torch.cat([gate * vec_aggr, vec_res], dim=-1)
                vec_combined = self.vec_merge_2(vec_combined)
        else:
            if self.vector_mixing == 'add':
                vec_combined = vec_aggr + vec_res
            elif self.vector_mixing == 'concat':
                vec_combined = torch.cat([vec_aggr, vec_res], dim=-1)
                vec_combined = self.vec_merge_2(vec_combined)

        o1, o2, o3 = torch.split(self.o_proj(x_out), self.hidden_channels, dim=-1)
        x_updated = vec_dot * o1 + vec_norm * o2 + o3

        x_final = torch.cat([x_updated.unsqueeze(2), vec_combined], dim=2)

        # Final output masking based on query positions
        if mask is not None:
            mask_query = mask.unsqueeze(-1).unsqueeze(-1).float() # (B, N, 1, 1)
            x_final = x_final * mask_query

        # Triton kernel doesn't return attention weights
        return x_final, None

