# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused 5-way projection in bf16, bmm outer product in bfloat16.
No module instantiation - work directly with weight tensors.
Uses TF32 for fp32 paths and bf16 for GEMM/bmm to leverage H100 tensor cores.
"""

import torch
import torch.nn.functional as F

# Enable TF32 on H100 for fp32 matmul (uses tensor cores ~3x faster, minimal precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def custom_kernel(data):
    """
    TriMul forward: LayerNorm (fp32) -> fused 5-way linear (bf16 GEMM) -> sigmoid gates ->
    masked outer product (bmm in bfloat16) -> output LayerNorm (fp32) -> gate -> project (bf16).
    All 5 projections fused into single bf16 matmul for H100 tensor core throughput.
    TF32 enabled for fp32 layer_norm and output projection paths.
    Contiguous() called before permute+reshape to ensure coalesced bmm inputs.
    """
    input_tensor, mask, weights, config = data

    norm_w = weights['norm.weight'].to(torch.float32)
    norm_b = weights['norm.bias'].to(torch.float32)
    out_norm_w = weights['to_out_norm.weight'].to(torch.float32)
    out_norm_b = weights['to_out_norm.bias'].to(torch.float32)
    to_out_w = weights['to_out.weight'].to(torch.float32)

    bs, seqlen, _, dim = input_tensor.shape
    hidden_dim = weights['left_proj.weight'].shape[0]

    # LayerNorm in fp32 for accuracy
    x = F.layer_norm(input_tensor.to(torch.float32), [dim], norm_w, norm_b)

    # Fuse all 5 projections into single bf16 matmul for tensor core throughput
    # Concatenate weights as bf16: [5*hidden, dim]
    fused_w_bf16 = torch.cat([
        weights['left_proj.weight'].to(torch.bfloat16),
        weights['right_proj.weight'].to(torch.bfloat16),
        weights['left_gate.weight'].to(torch.bfloat16),
        weights['right_gate.weight'].to(torch.bfloat16),
        weights['out_gate.weight'].to(torch.bfloat16),
    ], dim=0)

    # bf16 GEMM: [bs*seqlen*seqlen, 5*hidden]
    x_flat_bf16 = x.reshape(-1, dim).to(torch.bfloat16)
    fused_out = F.linear(x_flat_bf16, fused_w_bf16).reshape(bs, seqlen, seqlen, 5 * hidden_dim)

    # Split projections (still in bf16)
    left       = fused_out[..., :hidden_dim]
    right      = fused_out[..., hidden_dim:2*hidden_dim]
    left_gate  = fused_out[..., 2*hidden_dim:3*hidden_dim]
    right_gate = fused_out[..., 3*hidden_dim:4*hidden_dim]
    out_gate   = fused_out[..., 4*hidden_dim:5*hidden_dim]

    # Apply mask and sigmoid gates in bf16
    mask_exp = mask.unsqueeze(-1).to(torch.bfloat16)
    left = left * mask_exp * torch.sigmoid(left_gate)
    right = right * mask_exp * torch.sigmoid(right_gate)
    out_gate_sig = torch.sigmoid(out_gate).to(torch.float32)

    # Outer product via bmm in bfloat16:
    # Make contiguous before permute to ensure coalesced memory access
    # left:  [bs, i, k, d] -> contiguous -> [bs*d, i, k]
    # right: [bs, j, k, d] -> contiguous -> [bs*d, k, j]
    left_p  = left.contiguous().permute(0, 3, 1, 2).reshape(bs * hidden_dim, seqlen, seqlen)
    right_p = right.contiguous().permute(0, 3, 2, 1).reshape(bs * hidden_dim, seqlen, seqlen)

    # bmm: [bs*hidden, i, k] x [bs*hidden, k, j] -> [bs*hidden, i, j]
    out = torch.bmm(left_p, right_p).reshape(bs, hidden_dim, seqlen, seqlen).permute(0, 2, 3, 1).to(torch.float32)

    # Output LayerNorm in fp32 for accuracy
    out = F.layer_norm(out, [hidden_dim], out_norm_w, out_norm_b)
    out = out * out_gate_sig
    out = F.linear(out, to_out_w)

    return out
# EVOLVE-BLOCK-END
