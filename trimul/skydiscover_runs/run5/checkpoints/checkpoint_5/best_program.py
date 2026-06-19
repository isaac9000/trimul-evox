# EVOLVE-BLOCK-START
"""
Optimized TriMul using fused projections and efficient outer product via bmm.
Key optimizations:
1. Fuse all 5 linear projections into one batched matmul to reduce kernel launches
2. Avoid module instantiation overhead - work directly with weight tensors
3. Use bfloat16 for the expensive outer product bmm to leverage tensor cores
4. Use contiguous memory layouts for better GPU utilization
5. Avoid redundant .to(float32) calls
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    """
    TriMul forward pass: LayerNorm -> fused 5-way projection -> sigmoid gating
    -> masked outer product via bmm in bfloat16 -> output LayerNorm -> gate -> project.
    All 5 projections fused into single matmul; outer product done as batched matmul.
    """
    input_tensor, mask, weights, config = data

    # Load weights in float32
    norm_w = weights['norm.weight'].to(torch.float32)
    norm_b = weights['norm.bias'].to(torch.float32)
    left_proj_w = weights['left_proj.weight'].to(torch.float32)
    right_proj_w = weights['right_proj.weight'].to(torch.float32)
    left_gate_w = weights['left_gate.weight'].to(torch.float32)
    right_gate_w = weights['right_gate.weight'].to(torch.float32)
    out_gate_w = weights['out_gate.weight'].to(torch.float32)
    out_norm_w = weights['to_out_norm.weight'].to(torch.float32)
    out_norm_b = weights['to_out_norm.bias'].to(torch.float32)
    to_out_w = weights['to_out.weight'].to(torch.float32)

    bs, seqlen, _, dim = input_tensor.shape
    hidden_dim = left_proj_w.shape[0]

    # LayerNorm input
    x = F.layer_norm(input_tensor.to(torch.float32), [dim], norm_w, norm_b)

    # Fuse all 5 projections into single matmul: [5*hidden, dim]
    fused_w = torch.cat([left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w], dim=0)
    x_flat = x.reshape(-1, dim)
    fused_out = F.linear(x_flat, fused_w).reshape(bs, seqlen, seqlen, 5 * hidden_dim)

    # Split projections
    left       = fused_out[..., :hidden_dim]
    right      = fused_out[..., hidden_dim:2*hidden_dim]
    left_gate  = fused_out[..., 2*hidden_dim:3*hidden_dim]
    right_gate = fused_out[..., 3*hidden_dim:4*hidden_dim]
    out_gate   = fused_out[..., 4*hidden_dim:5*hidden_dim]

    # Apply mask and sigmoid gates
    mask_exp = mask.unsqueeze(-1).to(torch.float32)
    left = left * mask_exp * torch.sigmoid(left_gate)
    right = right * mask_exp * torch.sigmoid(right_gate)
    out_gate_sig = torch.sigmoid(out_gate)

    # Outer product via bmm in bfloat16:
    # left:  [bs, i, k, d] -> permute -> [bs*d, i, k]
    # right: [bs, j, k, d] -> permute -> [bs*d, k, j]
    left_p  = left.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * hidden_dim, seqlen, seqlen)
    right_p = right.to(torch.bfloat16).permute(0, 3, 2, 1).reshape(bs * hidden_dim, seqlen, seqlen)

    # bmm: [bs*hidden, i, k] x [bs*hidden, k, j] -> [bs*hidden, i, j]
    out = torch.bmm(left_p, right_p).reshape(bs, hidden_dim, seqlen, seqlen).permute(0, 2, 3, 1).to(torch.float32)

    # Output LayerNorm, gate, and projection
    out = F.layer_norm(out, [hidden_dim], out_norm_w, out_norm_b)
    out = out * out_gate_sig
    out = F.linear(out, to_out_w)

    return out
# EVOLVE-BLOCK-END
