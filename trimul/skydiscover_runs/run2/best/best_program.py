# EVOLVE-BLOCK-START
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fused TriMul: single combined matmul for all 5 projections,
        fused mask+sigmoid gating, bf16 outer-product einsum, out norm/gate/proj."""
        hd = self.left_proj.weight.shape[0]

        x = self.norm(x)

        combined = torch.nn.functional.linear(x, self._fused_w)
        left, right, left_gate, right_gate, out_gate = combined.split(hd, dim=-1)

        b, n, _, _ = x.shape
        mask = mask.unsqueeze(-1)
        left = (left * mask).mul_(torch.sigmoid(left_gate)).to(torch.bfloat16)
        right = (right * mask).mul_(torch.sigmoid(right_gate)).to(torch.bfloat16)

        d = left.shape[-1]
        lb = left.permute(0, 3, 1, 2).reshape(b * d, n, n)
        rb = right.permute(0, 3, 2, 1).reshape(b * d, n, n)
        out = torch.bmm(lb, rb).reshape(b, d, n, n).permute(0, 2, 3, 1).to(torch.float32)

        out = self.to_out_norm(out)
        out = out.mul_(torch.sigmoid(out_gate))
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    trimul._fused_w = torch.cat([
        trimul.left_proj.weight,
        trimul.right_proj.weight,
        trimul.left_gate.weight,
        trimul.right_gate.weight,
        trimul.out_gate.weight,
    ], dim=0).contiguous()

    with torch.no_grad():
        output = trimul(input_tensor, mask).to(torch.float32)

    return output
# EVOLVE-BLOCK-END
