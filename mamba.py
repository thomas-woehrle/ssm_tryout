"""
Adapted (with parts copied) from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
"""

from dataclasses import dataclass
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


@dataclass
class ModelArgs:
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, args: ModelArgs):
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits


class ResidualBlock(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, args: ModelArgs):
        self.args = ModelArgs
        self.norm = RMSNorm(d_model=args.d_model)
        self.mamba_block = MambaBlock(args)

    def forward(self, x: torch.Tensor):
        y = self.mamba_block(self.norm(x))
        y = y + x

        return y


class MambaBlock(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, args: ModelArgs):
        self.args = args

        # non-ssm stuff
        self.in_proj = nn.Linear(args.d_model, args.d_inner)
        self.residual_in_proj = nn.Linear(args.d_model, args.d_inner)
        self.out_proj = nn.Linear(args.d_inner, args.d_model)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.use_conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # selective projections
        self.s_B = nn.Linear(args.d_inner, args.d_state)
        self.s_C = nn.Linear(args.d_inner, args.d_state)
        self.s_Delta_linear = nn.Linear(args.d_inner, 1)
        self.tau_Delta_param = nn.Parameter(torch.zeros(args.d_inner))

        # other ssm stuff
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x: Input of shape (b, l, d)
        """
        x = self.in_proj(x)
        res_x = self.residual_in_proj(x)

        seq_l = x.shape[1]
        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :seq_l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)
        res_x = F.silu(res_x)

        y = self.s6_ssm(x)
        y = y * res_x
        y = self.out_proj(y)

        return y

    def s6_ssm(self, x: torch.Tensor) -> torch.Tensor:
        B = self.s_B(x)
        C = self.s_C(x)
        Delta = F.softplus(self.tau_Delta_param + self.s_Delta_linear(x))

        A = -torch.exp(self.A_log)

        y = self.selective_scan(x, Delta, A, B, C, self.D)

        return y

    @staticmethod
    def selective_scan(
        u: torch.Tensor,
        Delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.tensor,
    ) -> torch.Tensor:
        (b, seq_l, d_inner) = u.shape
        d_state = A.shape[1]

        bar_A = torch.exp(einsum(Delta, A, "b l d_in, d_in n -> b l d_in n"))
        bar_B_u = einsum(Delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        h = torch.zeros(b, d_inner, d_state)
        ys = []

        for i in range(seq_l):
            h = bar_A[:, i] * h + bar_B_u[:, i]
            y = einsum(C[:, i], h, "b n, b d_in n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + D * u

        return y


class RMSNorm(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output
