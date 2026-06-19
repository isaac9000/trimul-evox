# EVOLVE-BLOCK-START
"""
Optimized TriMul: TF32 enabled, fused 4+1 bf16 GEMM projection, bmm outer product in bfloat16,
torch.compile(max-autotune, fullgraph=True) on core computation for kernel fusion.
Key optimizations:
1. TF32 enabled globally for fp32 matmul paths (tensor cores ~3x faster)
2. bf16 reduced precision reduction enabled for further throughput
3. 4-way fused GEMM for left/right/left_gate/right_gate + separate out_gate (less register pressure)
4. Use bfloat16 for the expensive outer product bmm to leverage tensor cores
5. torch.compile(max-autotune, fullgraph=True) on core computation for additional kernel fusion
6. .contiguous() before permute to ensure coalesced memory access for bmm
"""

import torch
import torch.nn.functional as F

# Enable TF32 on H100 for fp32 matmul (uses tensor cores ~3x faster)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def _trimul_core(x_flat_bf16, fused_w4_bf16, out_gate_w_bf16, mask_exp,
                 bs, seqlen, hidden_dim, out_norm_w, out_norm_b, to_out_w):
    """
    Core TriMul: 4-way fused bf16 GEMM (left/right/left_gate/right_gate) + separate out_gate GEMM,
    sigmoid gating, masked bmm outer product in bf16, fp32 LayerNorm, gate, project.
    Designed for torch.compile(max-autotune) fusion on H100.
    """
    # 4-way fused GEMM: left, right, left_gate, right_gate
    fused4 = F.linear(x_flat_bf16, fused_w4_bf16).reshape(bs, seqlen, seqlen, 4 * hidden_dim)
    left       = fused4[..., :hidden_dim]
    right      = fused4[..., hidden_dim:2*hidden_dim]
    left_gate  = fused4[..., 2*hidden_dim:3*hidden_dim]
    right_gate = fused4[..., 3*hidden_dim:4*hidden_dim]

    # Separate out_gate GEMM (reduces register pressure vs 5-way fused)
    out_gate_raw = F.linear(x_flat_bf16, out_gate_w_bf16).reshape(bs, seqlen, seqlen, hidden_dim)

    # Apply mask and sigmoid gates in bf16
    left = left * mask_exp * torch.sigmoid(left_gate)
    right = right * mask_exp * torch.sigmoid(right_gate)
    out_gate_sig = torch.sigmoid(out_gate_raw).to(torch.float32)

    # Outer product via bmm in bfloat16
    left_p  = left.contiguous().permute(0, 3, 1, 2).reshape(bs * hidden_dim, seqlen, seqlen)
    right_p = right.contiguous().permute(0, 3, 2, 1).reshape(bs * hidden_dim, seqlen, seqlen)
    out = torch.bmm(left_p, right_p).reshape(bs, hidden_dim, seqlen, seqlen).permute(0, 2, 3, 1).to(torch.float32)

    # Output LayerNorm in fp32 for accuracy
    out = F.layer_norm(out, [hidden_dim], out_norm_w, out_norm_b)
    out = out * out_gate_sig
    out = F.linear(out, to_out_w)
    return out


_compiled_trimul_core = torch.compile(_trimul_core, mode='max-autotune', fullgraph=True)


def custom_kernel(data):
    """
    TriMul forward: LayerNorm (fp32) -> split 4+1 fused linear (bf16 GEMM) -> sigmoid gates ->
    masked outer product (bmm in bfloat16) -> output LayerNorm (fp32) -> gate -> project (fp32).
    TF32 enabled for all fp32 matmul paths. bf16 GEMM for projections for tensor core throughput.
    4-way fused GEMM for left/right proj+gate, separate GEMM for out_gate to reduce register pressure.
    torch.compile(max-autotune) applied to core computation for additional fusion.
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

    # Fuse 4 projections (left, right, left_gate, right_gate) into single bf16 matmul
    fused_w4_bf16 = torch.cat([
        weights['left_proj.weight'].to(torch.bfloat16),
        weights['right_proj.weight'].to(torch.bfloat16),
        weights['left_gate.weight'].to(torch.bfloat16),
        weights['right_gate.weight'].to(torch.bfloat16),
    ], dim=0)

    out_gate_w_bf16 = weights['out_gate.weight'].to(torch.bfloat16)

    x_flat_bf16 = x.reshape(-1, dim).to(torch.bfloat16)
    mask_exp = mask.unsqueeze(-1).to(torch.bfloat16)

    out = _compiled_trimul_core(x_flat_bf16, fused_w4_bf16, out_gate_w_bf16, mask_exp,
                                bs, seqlen, hidden_dim, out_norm_w, out_norm_b, to_out_w)
    return out
# EVOLVE-BLOCK-END
