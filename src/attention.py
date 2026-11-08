"""Task 4: reference scaled dot-product attention and tiled online-softmax attention.

Shapes: q [B, H, Lq, D], k/v [B, H, Lk, D].
Causal convention: the Lq queries are the LAST Lq positions of a Lk-long sequence
(query i sits at position Lk - Lq + i). With Lq == Lk this is the usual causal mask;
with Lq == 1 it is exactly one KV-cache decode step (Task 3).
Inputs in fp16/bf16 are accumulated in fp32 (like real fused kernels), fp64 stays fp64.
"""
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def _compute_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32

def causal_mask(q_start, q_end, k_start, k_end, Lq, Lk, device):
    """True where the key is in the FUTURE of the query (must be masked out)."""
    q_pos = (Lk - Lq) + torch.arange(q_start, q_end, device=device)
    k_pos = torch.arange(k_start, k_end, device=device)
    return k_pos[None, :] > q_pos[:, None]

def reference_attention(q, k, v, causal=True, scale=None):
    """softmax(q k^T / sqrt(D)) v, materialising the full Lq x Lk score matrix."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    if causal:
        assert Lk >= Lq, "causal attention needs Lk >= Lq"
    cd = _compute_dtype(q)
    scale = (1.0 / math.sqrt(D)) if scale is None else scale
    s = torch.matmul(q.to(cd), k.to(cd).transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(causal_mask(0, Lq, 0, Lk, Lq, Lk, q.device), float("-inf"))
    m = s.max(dim=-1, keepdim=True).values          # stable softmax: subtract row max
    e = torch.exp(s - m)
    p = e / e.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.to(cd)).to(q.dtype)

def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64, scale=None):
    """Blocked attention with online softmax. Only a block_q x block_k tile of scores
    exists at any time; the Lq x Lk matrix is never built."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    if causal:
        assert Lk >= Lq, "causal attention needs Lk >= Lq"
    cd = _compute_dtype(q)
    scale = (1.0 / math.sqrt(D)) if scale is None else scale
    offset = Lk - Lq
    out = torch.empty_like(q)
    neg_inf = float("-inf")

    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        bq = qe - qs
        qb = q[:, :, qs:qe].to(cd)
        m = torch.full((B, H, bq, 1), neg_inf, dtype=cd, device=q.device)  # running max
        l = torch.zeros((B, H, bq, 1), dtype=cd, device=q.device)          # running sum
        acc = torch.zeros((B, H, bq, D), dtype=cd, device=q.device)        # running P @ V
        q_pos_min, q_pos_max = offset + qs, offset + qe - 1

        for ks in range(0, Lk, block_k):
            if causal and ks > q_pos_max:      # whole key block is in the future: skip
                break
            ke = min(ks + block_k, Lk)
            kb, vb = k[:, :, ks:ke].to(cd), v[:, :, ks:ke].to(cd)
            s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
            if causal and ke - 1 > q_pos_min:  # block touches the diagonal: mask inside
                s = s.masked_fill(causal_mask(qs, qe, ks, ke, Lq, Lk, q.device), neg_inf)

            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            # a row can be fully masked in this block (m_new = -inf): use 0 to avoid inf-inf = nan
            m_safe = torch.where(m_new == neg_inf, torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            alpha = torch.exp(m - m_safe)      # rescales everything accumulated so far
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, vb)
            m = m_new

        out[:, :, qs:qe] = (acc / l).to(q.dtype)
    return out

BACKENDS = {"math": SDPBackend.MATH,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION}

def sdpa(q, k, v, causal=True, backend=None):
    """PyTorch's built-in attention (fused kernels), for comparison only.
    Note: is_causal=True aligns the mask top-left, so only use it when Lq == Lk."""
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    with sdpa_kernel(BACKENDS[backend]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
