import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

preprocess_configs = [
    triton.Config({'BLOCK_M': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
]

bwd_configs = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
]

@triton.autotune(configs=preprocess_configs, key=["HEAD_DIM", "VEC_DIM_PADDED"])
@triton.jit
def _attn_bwd_preprocess_fused(
    O_scalar, O_vector, dO_scalar, dO_vector, Delta,
    stride_osz, stride_osh, stride_osm, stride_osk,
    stride_ovz, stride_ovh, stride_ovm, stride_ovv,
    stride_dosz, stride_dosh, stride_dosm, stride_dosk,
    stride_dovz, stride_dovh, stride_dovm, stride_dovv,
    stride_dz, stride_dh, stride_dn,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    VEC_DIM_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,          # autotuner picks this
):

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_z = pid_bh // H
    off_h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX

    # Offsets --------------------------------------------------------------
    o_s_offset = off_z * stride_osz + off_h * stride_osh
    o_v_offset = off_z * stride_ovz + off_h * stride_ovh
    do_s_offset = off_z * stride_dosz + off_h * stride_dosh
    do_v_offset = off_z * stride_dovz + off_h * stride_dovh
    delta_offset = off_z * stride_dz + off_h * stride_dh

    # Block pointers -------------------------------------------------------
    O_s_ptr = tl.make_block_ptr(O_scalar + o_s_offset, (N_CTX, HEAD_DIM),
                                (stride_osm, stride_osk), (start_m, 0),
                                (BLOCK_M, HEAD_DIM), order=(1, 0))
    dO_s_ptr = tl.make_block_ptr(dO_scalar + do_s_offset, (N_CTX, HEAD_DIM),
                                 (stride_dosm, stride_dosk), (start_m, 0),
                                 (BLOCK_M, HEAD_DIM), order=(1, 0))
    O_v_ptr = tl.make_block_ptr(O_vector + o_v_offset, (N_CTX, VEC_DIM_PADDED),
                                (stride_ovm, stride_ovv), (start_m, 0),
                                (BLOCK_M, VEC_DIM_PADDED), order=(1, 0))
    dO_v_ptr = tl.make_block_ptr(dO_vector + do_v_offset, (N_CTX, VEC_DIM_PADDED),
                                 (stride_dovm, stride_dovv), (start_m, 0),
                                 (BLOCK_M, VEC_DIM_PADDED), order=(1, 0))

    # Loads (float32) ------------------------------------------------------
    o_s  = tl.load(O_s_ptr,  boundary_check=(0,), padding_option="zero").to(tl.float32)
    do_s = tl.load(dO_s_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    o_v  = tl.load(O_v_ptr,  boundary_check=(0,), padding_option="zero").to(tl.float32)
    do_v = tl.load(dO_v_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    delta_s = tl.sum(o_s * do_s, 1)
    delta_v = tl.sum(o_v * do_v, 1)
    delta   = delta_s + delta_v

    # Store ---------------------------------------------------------------
    Delta_ptrs = Delta + delta_offset + offs_m * stride_dn
    tl.store(Delta_ptrs, delta, mask=mask_m)

@triton.autotune(configs=bwd_configs, key=["HEAD_DIM", "VEC_DIM_PADDED"])
@triton.jit
def _attn_bwd_fused_dq(
    Q, K, V_scalar, V_vector,
    sm_scale,
    DO_scalar, DO_vector,
    M, Delta, L,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vsz, stride_vsh, stride_vsn, stride_vsk,
    stride_vvz, stride_vvh, stride_vvn, stride_vvv,
    stride_dosz, stride_dosh, stride_dosm, stride_dosk,
    stride_dovz, stride_dovh, stride_dovm, stride_dovv,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_mz, stride_mh, stride_mn,
    stride_deltaz, stride_deltah, stride_deltan,
    stride_lz,     stride_lh,     stride_ln,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    VEC_DIM_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bhid = tl.program_id(2)
    off_z = bhid // H
    off_h = bhid % H
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh
    vv_offset = off_z * stride_vvz + off_h * stride_vvh
    dos_offset = off_z * stride_dosz + off_h * stride_dosh
    dov_offset = off_z * stride_dovz + off_h * stride_dovh
    dq_offset = off_z * stride_dqz + off_h * stride_dqh
    m_offset = off_z * stride_mz + off_h * stride_mh
    delta_offset = off_z * stride_deltaz + off_h * stride_deltah
    l_offset     = off_z * stride_lz     + off_h * stride_lh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)

    # Build masks
    mask_m = offs_m < N_CTX             # shape [BLOCK_M]
    mask2d = tl.expand_dims(mask_m, 1) # shape [BLOCK_M, 1]

    # Dimension indices
    offs_k = tl.arange(0, HEAD_DIM)
    offs_v = tl.arange(0, VEC_DIM_PADDED)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q, dO_s, dO_v blocks -> float32 (using 2D mask)
    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs, mask=mask2d, other=0.0).to(tl.float32) # Ensure float32
    dO_s_ptrs = DO_scalar + dos_offset + offs_m[:, None] * stride_dosm + offs_k[None, :] * stride_dosk
    do_s = tl.load(dO_s_ptrs, mask=mask2d, other=0.0).to(tl.float32) # Ensure float32
    dO_v_ptrs = DO_vector + dov_offset + offs_m[:, None] * stride_dovm + offs_v[None, :] * stride_dovv
    do_v = tl.load(dO_v_ptrs, mask=mask2d, other=0.0).to(tl.float32) # Ensure float32

    # Load M, Delta, L using block pointers (boundary checked)
    M_block_ptr = tl.make_block_ptr(
        base=M + m_offset, shape=(N_CTX,), strides=(stride_mn,),
        offsets=(start_m,), block_shape=(BLOCK_M,), order=(0,)
    )
    m_val = tl.load(M_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) # Use "zero" padding, or adjust if needed

    Delta_block_ptr = tl.make_block_ptr(
        base=Delta + delta_offset, shape=(N_CTX,), strides=(stride_deltan,),
        offsets=(start_m,), block_shape=(BLOCK_M,), order=(0,)
    )
    delta_m = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < N_CTX)

        K_T_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        kT = tl.load(K_T_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

        V_s_ptrs = V_scalar + vs_offset + offs_n[:, None] * stride_vsn + offs_d[None, :] * stride_vsk
        v_s = tl.load(V_s_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        V_v_ptrs = V_vector + vv_offset + offs_n[:, None] * stride_vvn + offs_v[None, :] * stride_vvv
        v_v = tl.load(V_v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, kT, allow_tf32=False)
        scaled_qk = qk * sm_scale

        p = tl.math.exp(scaled_qk - m_val[:, None])

        L_block_ptr = tl.make_block_ptr(
                base = L + l_offset,
                shape = (N_CTX,),
                strides = (stride_ln,),
                offsets = (start_m,),
                block_shape = (BLOCK_M,),
                order = (0,),
            )
        l_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        l_row = l_block.to(tl.float32)

        p_norm = p / (l_row[:, None] + 1e-7)

        # --- Accumulate dQ using float32 ---
        dp_s = tl.dot(do_s, tl.trans(v_s), allow_tf32=False)
        dp_v = tl.dot(do_v, tl.trans(v_v), allow_tf32=False)
        dp = dp_s + dp_v # float32

        # Calculate dS = P * (dP - delta) using float32
        ds = p_norm * (dp - delta_m[:, None])

        # Calculate tmp_dq = dS @ K using float32
        K_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k_block = tl.load(K_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        tmp_dq = tl.dot(ds, k_block, allow_tf32=False)
        dq = dq + tmp_dq

    # # --- Store final dQ ---
    dq *= sm_scale
    DQ_ptrs = DQ + dq_offset + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
    tl.store(DQ_ptrs, dq.to(DQ.dtype.element_ty), mask=mask2d)


@triton.autotune(configs=bwd_configs, key=["HEAD_DIM", "VEC_DIM_PADDED"])
@triton.jit
def _attn_bwd_fused(
    Q, K, V_scalar, V_vector,
    sm_scale,
    DO_scalar, DO_vector,
    M, Delta, L,
    DQ, DK, DV_scalar, DV_vector,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vsz, stride_vsh, stride_vsn, stride_vsk,
    stride_vvz, stride_vvh, stride_vvn, stride_vvv,
    stride_dosz, stride_dosh, stride_dosm, stride_dosk,
    stride_dovz, stride_dovh, stride_dovm, stride_dovv,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvsz, stride_dvsh, stride_dvsn, stride_dvsk,
    stride_dvvz, stride_dvvh, stride_dvvn, stride_dvvv,
    stride_mz, stride_mh, stride_mn,
    stride_deltaz, stride_deltah, stride_deltan,
    stride_lz,     stride_lh,     stride_ln,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    VEC_DIM_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs and Offsets
    pid_n = tl.program_id(0)
    bhid = tl.program_id(2)
    off_z = bhid // H
    off_h = bhid % H

    # Input/Output Offsets
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh
    vv_offset = off_z * stride_vvz + off_h * stride_vvh
    dos_offset = off_z * stride_dosz + off_h * stride_dosh
    dov_offset = off_z * stride_dovz + off_h * stride_dovh
    dk_offset = off_z * stride_dkz + off_h * stride_dkh
    dvs_offset = off_z * stride_dvsz + off_h * stride_dvsh
    dvv_offset = off_z * stride_dvvz + off_h * stride_dvvh
    m_offset = off_z * stride_mz + off_h * stride_mh
    delta_offset = off_z * stride_deltaz + off_h * stride_deltah
    l_offset     = off_z * stride_lz     + off_h * stride_lh

    # Key Block Info
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = (offs_n < N_CTX)

    # Dimension Indices
    offs_k = tl.arange(0, HEAD_DIM)
    offs_v = tl.arange(0, VEC_DIM_PADDED)

    # Accumulators (float32)
    dv_s = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_v = tl.zeros([BLOCK_N, VEC_DIM_PADDED], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Load K, V_scalar, V_vector blocks -> float32
    K_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    k = tl.load(K_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    V_s_ptrs = V_scalar + vs_offset + offs_n[:, None] * stride_vsn + offs_k[None, :] * stride_vsk
    v_s = tl.load(V_s_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    V_v_ptrs = V_vector + vv_offset + offs_n[:, None] * stride_vvn + offs_v[None, :] * stride_vvv
    v_v = tl.load(V_v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    # Inner loop over Query dimension (M)
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = (offs_m < N_CTX)

        # Load Q block -> float32
        Q_ptrs = Q + q_offset + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
        qT = tl.load(Q_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)

        # --- P^T Recomputation ---
        qkT = tl.dot(k, qT, allow_tf32=False) # float32 inputs/output
        scaled_qkT = qkT * sm_scale # float32

        # --- Load M using block pointer with boundary check ---
        M_block_ptr = tl.make_block_ptr(
            base=M + m_offset, shape=(N_CTX,), strides=(stride_mn,),
            offsets=(start_m,), block_shape=(BLOCK_M,), order=(0,)
        )
        # Use padding_option="zero" to fill out-of-bounds with 0.0
        m_val = tl.load(M_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)


        m_val_expanded = tl.expand_dims(m_val, 0)
        pT = tl.math.exp(scaled_qkT - m_val_expanded)

        L_block_ptr = tl.make_block_ptr(
            base=L + l_offset, shape=(N_CTX,), strides=(stride_ln,),
            offsets=(start_m,), block_shape=(BLOCK_M,), order=(0,)
        )
        # Use padding_option="zero". Add epsilon later before division.
        l_row = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        l_row_expanded = tl.expand_dims(l_row, 0)
        pT_normalized = pT / (l_row_expanded + 1e-7)
        # --- End P^T Recomputation ---

        mask_m_2d = tl.expand_dims(mask_m, 1)
        dO_s_ptrs = DO_scalar + dos_offset + offs_m[:, None] * stride_dosm + offs_k[None, :] * stride_dosk
        do_s = tl.load(dO_s_ptrs, mask=mask_m_2d, other=0.0).to(tl.float32)
        dO_v_ptrs = DO_vector + dov_offset + offs_m[:, None] * stride_dovm + offs_v[None, :] * stride_dovv
        do_v = tl.load(dO_v_ptrs, mask=mask_m_2d, other=0.0).to(tl.float32)

        # --- Accumulate dV (using float32) ---
        tmp_dv_s = tl.dot(pT_normalized, do_s, allow_tf32=False)
        tmp_dv_v = tl.dot(pT_normalized, do_v, allow_tf32=False)
        dv_s += tmp_dv_s
        dv_v += tmp_dv_v

        Delta_block_ptr = tl.make_block_ptr(
            base=Delta + delta_offset, shape=(N_CTX,), strides=(stride_deltan,),
            offsets=(start_m,), block_shape=(BLOCK_M,), order=(0,)
        )
        delta_m = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # Calculate dpT = V dO^T using float32
        dpT_s = tl.dot(v_s, tl.trans(do_s), allow_tf32=False)
        dpT_v = tl.dot(v_v, tl.trans(do_v), allow_tf32=False)
        dpT = dpT_s + dpT_v

        delta_m_expanded = tl.expand_dims(delta_m, 0)
        dsT = pT_normalized * (dpT - delta_m_expanded)

        tmp_dk = tl.dot(dsT, tl.trans(qT), allow_tf32=False)
        dk += tmp_dk

    # --- Store final gradients ---
    DV_s_ptrs = DV_scalar + dvs_offset + offs_n[:, None] * stride_dvsn + offs_k[None, :] * stride_dvsk
    tl.store(DV_s_ptrs, dv_s.to(DV_scalar.dtype.element_ty), mask=mask_n[:, None])
    DV_v_ptrs = DV_vector + dvv_offset + offs_v[None, :] * stride_dvvv + offs_n[:, None] * stride_dvvn
    tl.store(DV_v_ptrs, dv_v.to(DV_vector.dtype.element_ty), mask=mask_n[:, None])

    dk *= sm_scale
    DK_ptrs = DK + dk_offset + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DK_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])


def launch_preprocess_kernel(O_s: torch.Tensor, O_v: torch.Tensor,
                             dO_s: torch.Tensor, dO_v: torch.Tensor,
                             Delta_out: torch.Tensor):
    """Run the autotuned Δ-preprocess kernel."""
    B, H, N_CTX, HEAD_DIM = O_s.shape
    VEC_DIM_PADDED = O_v.shape[-1]

    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), B * H)

    _attn_bwd_preprocess_fused[grid](
        O_s, O_v, dO_s, dO_v, Delta_out,
        *O_s.stride(), *O_v.stride(), *dO_s.stride(), *dO_v.stride(),
        *Delta_out.stride(),
        B, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        VEC_DIM_PADDED=VEC_DIM_PADDED,
    )


def launch_dkdv_calculation_kernel(Q: torch.Tensor, K: torch.Tensor,
                                   V_s: torch.Tensor, V_v: torch.Tensor,
                                   dO_s: torch.Tensor, dO_v: torch.Tensor,
                                   M: torch.Tensor, Delta: torch.Tensor,
                                   sm_scale: float, L_out: torch.Tensor,
                                   DK_out: torch.Tensor, DV_s_out: torch.Tensor,
                                   DV_v_out: torch.Tensor):
    """Autotuned DK / dV kernel."""
    B, H, N_CTX, HEAD_DIM = Q.shape
    VEC_DIM_PADDED = V_v.shape[-1]
    DQ_dummy = torch.empty_like(Q)

    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N']), 1, B * H)

    _attn_bwd_fused[grid](
        Q, K, V_s, V_v, sm_scale, dO_s, dO_v, M, Delta, L_out,
        DQ_dummy, DK_out, DV_s_out, DV_v_out,
        *Q.stride(), *K.stride(), *V_s.stride(), *V_v.stride(),
        *dO_s.stride(), *dO_v.stride(),
        *DQ_dummy.stride(), *DK_out.stride(),
        *DV_s_out.stride(), *DV_v_out.stride(),
        *M.stride(), *Delta.stride(), *L_out.stride(),
        B, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        VEC_DIM_PADDED=VEC_DIM_PADDED,
    )


def launch_dq_calculation_kernel(Q: torch.Tensor, K: torch.Tensor,
                                 V_s: torch.Tensor, V_v: torch.Tensor,
                                 dO_s: torch.Tensor, dO_v: torch.Tensor,
                                 M: torch.Tensor, Delta: torch.Tensor,
                                 sm_scale: float, L_out: torch.Tensor,
                                 DQ_out: torch.Tensor):
    """Autotuned dQ kernel."""
    B, H, N_CTX, HEAD_DIM = Q.shape
    VEC_DIM_PADDED = V_v.shape[-1]

    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), 1, B * H)

    _attn_bwd_fused_dq[grid](
        Q, K, V_s, V_v, sm_scale, dO_s, dO_v, M, Delta, L_out, DQ_out,
        *Q.stride(), *K.stride(), *V_s.stride(), *V_v.stride(),
        *dO_s.stride(), *dO_v.stride(), *DQ_out.stride(),
        *M.stride(), *Delta.stride(), *L_out.stride(),
        B, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        VEC_DIM_PADDED=VEC_DIM_PADDED,
    )

