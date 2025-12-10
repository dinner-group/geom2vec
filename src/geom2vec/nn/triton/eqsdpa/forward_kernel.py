import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

configs = [
    triton.Config(
        {
            'BLOCK_M': 64,
            'BLOCK_N': 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_M': 32,
            'BLOCK_N': 64,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            'BLOCK_M': 128,
            'BLOCK_N': 32,
        },
        num_warps=8,
        num_stages=3,
    ),
]

@triton.autotune(configs=configs, key=['HEAD_DIM', 'VEC_DIM_PADDED'])
@triton.jit
def _fused_attn_fwd_kernel(
    # --- pointers -----------------------------------------------------------
    Q, K, V, vec_V,
    Out_scalar, Out_vector,
    M_out,              # max-value for each row   (B, H, N)
    L_out,              # NEW: soft-max denom l_i  (B, H, N)
    # --- strides ------------------------------------------------------------
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_vvz, stride_vvh, stride_vvn, stride_vvv,
    stride_osz, stride_osh, stride_osm, stride_osk,
    stride_ovz, stride_ovh, stride_ovm, stride_ovv,
    stride_moz, stride_moh, stride_mom,
    stride_loz, stride_loh, stride_lom,           # NEW
    # --- sizes --------------------------------------------------------------
    Z, H, N_CTX,
    HEAD_DIM  : tl.constexpr,
    VEC_DIM_PADDED : tl.constexpr,
    BLOCK_M   : tl.constexpr,         # will be injected by autotune
    BLOCK_N   : tl.constexpr,         # will be injected by autotune
    qk_scale  : tl.constexpr,
):
    """Fused attention forward kernel with autotune support."""
    start_m = tl.program_id(0)              # row-tile index (along N_CTX)
    off_hz   = tl.program_id(1)             # batch*head
    off_z    = off_hz // H                  # batch  index
    off_h    = off_hz %  H                  # head   index

    # ------------------------------------------------------------------ ptrs
    q_offset  = off_z * stride_qz  + off_h * stride_qh
    k_offset  = off_z * stride_kz  + off_h * stride_kh
    v_offset  = off_z * stride_vz  + off_h * stride_vh
    vv_offset = off_z * stride_vvz + off_h * stride_vvh
    os_offset = off_z * stride_osz + off_h * stride_osh
    ov_offset = off_z * stride_ovz + off_h * stride_ovh
    mo_offset = off_z * stride_moz + off_h * stride_moh
    lo_offset = off_z * stride_loz + off_h * stride_loh   # NEW

    # block pointers --------------------------------------------------------
    Q_ptr  = tl.make_block_ptr(base = Q  + q_offset,
                               shape=(N_CTX, HEAD_DIM),
                               strides=(stride_qm, stride_qk),
                               offsets=(start_m * BLOCK_M, 0),
                               block_shape=(BLOCK_M, HEAD_DIM),
                               order=(1, 0))
    K_ptr  = tl.make_block_ptr(base = K  + k_offset,
                               shape=(HEAD_DIM, N_CTX),
                               strides=(stride_kk, stride_kn),
                               offsets=(0, 0),
                               block_shape=(HEAD_DIM, BLOCK_N),
                               order=(0, 1))
    V_ptr  = tl.make_block_ptr(base = V  + v_offset,
                               shape=(N_CTX, HEAD_DIM),
                               strides=(stride_vn, stride_vk),
                               offsets=(0, 0),
                               block_shape=(BLOCK_N, HEAD_DIM),
                               order=(1, 0))
    VV_ptr = tl.make_block_ptr(base = vec_V + vv_offset,
                               shape=(N_CTX, VEC_DIM_PADDED),
                               strides=(stride_vvn, stride_vvv),
                               offsets=(0, 0),
                               block_shape=(BLOCK_N, VEC_DIM_PADDED),
                               order=(1, 0))
    OS_ptr = tl.make_block_ptr(base = Out_scalar + os_offset,
                               shape=(N_CTX, HEAD_DIM),
                               strides=(stride_osm, stride_osk),
                               offsets=(start_m * BLOCK_M, 0),
                               block_shape=(BLOCK_M, HEAD_DIM),
                               order=(1, 0))
    OV_ptr = tl.make_block_ptr(base = Out_vector + ov_offset,
                               shape=(N_CTX, VEC_DIM_PADDED),
                               strides=(stride_ovm, stride_ovv),
                               offsets=(start_m * BLOCK_M, 0),
                               block_shape=(BLOCK_M, VEC_DIM_PADDED),
                               order=(1, 0))
    M_ptr  = tl.make_block_ptr(base = M_out + mo_offset,
                               shape=(N_CTX,),
                               strides=(stride_mom,),
                               offsets=(start_m * BLOCK_M,),
                               block_shape=(BLOCK_M,),
                               order=(0,))
    L_ptr  = tl.make_block_ptr(base = L_out + lo_offset,  # NEW
                               shape=(N_CTX,),
                               strides=(stride_lom,),
                               offsets=(start_m * BLOCK_M,),
                               block_shape=(BLOCK_M,),
                               order=(0,))

    # ---------------------------------------------------------------- accum
    acc_s   = tl.zeros([BLOCK_M, HEAD_DIM],     dtype=tl.float32)
    acc_v   = tl.zeros([BLOCK_M, VEC_DIM_PADDED], dtype=tl.float32)
    l_i     = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i     = tl.full ([BLOCK_M], -float('inf'), dtype=tl.float32)

    q = tl.load(Q_ptr)                                   # (BLOCK_M, D)

    # ---------------------------------------------------------------- main
    lo = 0
    hi = N_CTX
    K_ptr  = tl.advance(K_ptr,  (0, lo))
    V_ptr  = tl.advance(V_ptr,  (lo, 0))
    VV_ptr = tl.advance(VV_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_ptr, boundary_check=(0,), padding_option="zero")
        vv = tl.load(VV_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.dot(q, k, allow_tf32=False) * qk_scale

        # --- Mask QK scores for padding ---
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        qk = tl.where(mask_n[None, :], qk, -float('inf'))

        # numerically-stable soft-max (tile-wise update)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p     = tl.math.exp(qk - m_new[:, None])
        alpha = tl.math.exp(m_i - m_new)

        p_f32 = p
        v_f32 = v.to(tl.float32)
        vv_f32 = vv.to(tl.float32)

        tmp_s = tl.dot(p_f32, v_f32, allow_tf32=False)
        tmp_v = tl.dot(p_f32, vv_f32, allow_tf32=False)
        sum_p = tl.sum(p_f32, 1)

        l_i = alpha * l_i + sum_p
        acc_s = alpha[:, None] * acc_s + tmp_s
        acc_v = alpha[:, None] * acc_v + tmp_v

        m_i   = m_new

        # advance block pointers
        K_ptr  = tl.advance(K_ptr,  (0, BLOCK_N))
        V_ptr  = tl.advance(V_ptr,  (BLOCK_N, 0))
        VV_ptr = tl.advance(VV_ptr, (BLOCK_N, 0))

    # ---------------------------------------------------------------- store
    inv_l   = 1.0 / (l_i + 1e-7)
    out_s   = acc_s * inv_l[:, None]
    out_v   = acc_v * inv_l[:, None]

    tl.store(OS_ptr, out_s.to(Out_scalar.dtype.element_ty), boundary_check=(0,))
    tl.store(OV_ptr, out_v.to(Out_vector.dtype.element_ty), boundary_check=(0,))
    tl.store(M_ptr,  m_i,                        boundary_check=(0,))
    tl.store(L_ptr,  l_i,                        boundary_check=(0,))  # NEW


def fused_attention_forward(q, k, v, vec_v):
    """Launch the autotuned fused attention kernel and return the outputs."""
    B, H, N_CTX, HEAD_DIM = q.shape
    _, _, _, VEC_DIM = vec_v.shape
    VEC_DIM_PADDED = triton.next_power_of_2(VEC_DIM)
    pad = VEC_DIM_PADDED - VEC_DIM
    vec_v_pad = F.pad(vec_v, (0, pad)) if pad else vec_v

    out_scalar = torch.empty_like(q)
    out_vec_pad = torch.empty((B, H, N_CTX, VEC_DIM_PADDED),
                              dtype=q.dtype, device=q.device)
    M_out = torch.empty((B, H, N_CTX), dtype=torch.float32, device=q.device)
    L_out = torch.empty_like(M_out)

    qk_scale = 1.0 / math.sqrt(HEAD_DIM)

    # grid function takes META dict chosen by autotune
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), B * H)

    _fused_attn_fwd_kernel[grid](
        # pointers
        q, k, v, vec_v_pad,
        out_scalar, out_vec_pad,
        M_out, L_out,
        # strides
        *q.stride(), *k.stride(), *v.stride(),
        *vec_v_pad.stride(),
        *out_scalar.stride(), *out_vec_pad.stride(),
        *M_out.stride(),
        *L_out.stride(),
        # sizes / meta
        B, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        VEC_DIM_PADDED=VEC_DIM_PADDED,
        qk_scale=qk_scale,
    )

    out_vector = out_vec_pad[..., :VEC_DIM].contiguous()
    return out_scalar, out_vector, M_out, L_out, out_vec_pad

