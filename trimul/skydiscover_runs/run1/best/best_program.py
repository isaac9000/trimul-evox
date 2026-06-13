# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused 5-way linear projection in BF16, BF16 outer product via bmm.
Fuses left_proj, right_proj, left_gate, right_gate, out_gate into one matmul.
Uses BF16 throughout projections to maximize H100 tensor core throughput.
Enables TF32 for FP32 matmuls. Pre-transposes right tensor for contiguous bmm.
"""

import torch
import torch.nn.functional as F

# Enable TF32 for H100 tensor cores on FP32 matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def custom_kernel(data):
    """
    TriMul forward pass:
    1. F.layer_norm for input norm (float32), then cast to BF16
    2. Single fused BF16 matmul for all 5 projections (left, right, lg, rg, og)
    3. Outer product via BF16 bmm: reshape to (bs*d, i, k) x (bs*d, j, k)^T
       - right_t transposed as (bs*d, k, j) via permute(0,3,2,1) for contiguous layout
    4. F.layer_norm (FP32) + gate + projection
    """
    input_tensor, mask, weights, config = data

    norm_w = weights['norm.weight'].float()
    norm_b = weights['norm.bias'].float()
    ton_w = weights['to_out_norm.weight'].float()
    ton_b = weights['to_out_norm.bias'].float()
    to_w = weights['to_out.weight'].float()

    bs, seq_len, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]

    # Input LayerNorm (float32) then cast to BF16 for projections
    x_f32 = F.layer_norm(input_tensor.float(), [dim], norm_w, norm_b)
    x_bf16 = x_f32.to(torch.bfloat16)

    # Fuse all 5 linear projections into one BF16 matmul: (5*hidden_dim, dim)
    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).to(torch.bfloat16)

    # Single BF16 matmul: (bs, seq, seq, 5*hidden_dim)
    all_proj = F.linear(x_bf16, fused_w)

    # Split projections
    left     = all_proj[..., :hidden_dim]
    right    = all_proj[..., hidden_dim:2*hidden_dim]
    lg       = all_proj[..., 2*hidden_dim:3*hidden_dim]
    rg       = all_proj[..., 3*hidden_dim:4*hidden_dim]
    out_gate = all_proj[..., 4*hidden_dim:]

    # Apply mask and gating (in BF16)
    mask_bf = mask.to(torch.bfloat16).unsqueeze(-1)
    left  = left  * mask_bf * torch.sigmoid(lg)
    right = right * mask_bf * torch.sigmoid(rg)
    out_gate = torch.sigmoid(out_gate)

    # Outer product: (bs, i, k, d), (bs, j, k, d) -> (bs, i, j, d)
    # Use matmul on (bs, i, k, d) treated as (bs*i, k, d) x (bs*j, k, d)^T
    # Reshape left: (bs*i, k, d), right: (bs*j, d, k) then matmul -> (bs*i, k, k)?
    # Better: permute to (bs, d, i, k) and (bs, d, j, k), then matmul
    left_t  = left.permute(0, 3, 1, 2).contiguous()   # (bs, d, i, k)
    right_t = right.permute(0, 3, 1, 2).contiguous()  # (bs, d, j, k)

    # matmul: (bs, d, i, k) x (bs, d, k, j) -> (bs, d, i, j)
    out_t = torch.matmul(left_t, right_t.transpose(-1, -2))

    # Reshape: (bs, d, i, j) -> (bs, i, j, d)
    out = out_t.permute(0, 2, 3, 1).contiguous().float()

    # Output LayerNorm + gate + projection
    out = F.layer_norm(out, [hidden_dim], ton_w, ton_b)
    out = out * out_gate.float()
    return F.linear(out, to_w)
# EVOLVE-BLOCK-END
