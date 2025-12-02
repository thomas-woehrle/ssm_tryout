"""
Adapted (with some parts copied) from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange


class MambaBlock:
    def __init__(
        self, d_model: int, d_inner: int, d_state: int, use_conv_bias: bool, d_conv
    ):
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        # non-ssm stuff
        self.in_proj = nn.Linear(d_model, d_inner)
        self.residual_in_proj = nn.Linear(d_model, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=use_conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # selective projections
        self.s_B = nn.Linear(d_inner, d_state)
        self.s_C = nn.Linear(d_inner, d_state)
        self.s_Delta_linear = nn.Linear(d_inner, 1)
        self.tau_Delta_param = nn.Parameter(torch.zeros(d_inner))

        # other ssm stuff
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

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
