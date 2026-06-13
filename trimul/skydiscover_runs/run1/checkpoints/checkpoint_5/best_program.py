# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused 5-way linear projection, BF16 outer product via bmm.
Fuses left_proj, right_proj, left_gate, right_gate, out_gate into one matmul.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    """
    TriMul forward pass:
    1. F.layer_norm for input norm
    2. Single fused matmul for all 5 projections (left, right, lg, rg, og)
    3. Outer product via BF16 bmm: reshape to (bs*d, i, k) x (bs*d, k, j)
    4. F.layer_norm + out projection
    """
    input_tensor, mask, weights, config = data

    norm_w = weights['norm.weight'].float()
    norm_b = weights['norm.bias'].float()
    ton_w = weights['to_out_norm.weight'].float()
    ton_b = weights['to_out_norm.bias'].float()
    to_w = weights['to_out.weight'].float()

    bs, seq_len, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]

    # Input LayerNorm (float32)
    x = F.layer_norm(input_tensor.float(), [dim], norm_w, norm_b)

    # Fuse all 5 linear projections into one matmul
    # Stack weights: (5*hidden_dim, dim)
    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).float()

    # Single matmul: (bs, seq, seq, 5*hidden_dim)
    all_proj = F.linear(x, fused_w)

    # Split projections
    left     = all_proj[..., :hidden_dim]
    right    = all_proj[..., hidden_dim:2*hidden_dim]
    lg       = all_proj[..., 2*hidden_dim:3*hidden_dim]
    rg       = all_proj[..., 3*hidden_dim:4*hidden_dim]
    out_gate = all_proj[..., 4*hidden_dim:]

    # Apply mask
    mask_f = mask.float().unsqueeze(-1)
    left = left * mask_f
    right = right * mask_f

    # Gating
    left  = left  * torch.sigmoid(lg)
    right = right * torch.sigmoid(rg)
    out_gate = torch.sigmoid(out_gate)

    # Outer product: (bs, i, k, d), (bs, j, k, d) -> (bs, i, j, d)
    # Reshape to (bs*d, i, k) for bmm
    left_t  = left.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * hidden_dim, seq_len, seq_len)
    right_t = right.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * hidden_dim, seq_len, seq_len)

    # bmm: (bs*d, i, k) x (bs*d, k, j) -> (bs*d, i, j)
    out_t = torch.bmm(left_t, right_t.transpose(1, 2))

    # Reshape: (bs*d, i, j) -> (bs, d, i, j) -> (bs, i, j, d)
    out = out_t.reshape(bs, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1).float()

    # Output LayerNorm + gate + projection
    out = F.layer_norm(out, [hidden_dim], ton_w, ton_b)
    out = out * out_gate
    return F.linear(out, to_w)
# EVOLVE-BLOCK-END
