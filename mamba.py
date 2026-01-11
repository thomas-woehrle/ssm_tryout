"""
Adapted (with parts copied) from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
"""

from dataclasses import dataclass
import json
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
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
        Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(
                model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
            )
            return json.load(open(resolved_archive_file))  # pyright: ignore[reportArgumentType]

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(
                model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
            )
            return torch.load(
                resolved_archive_file,  # pyright: ignore[reportArgumentType]
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data["d_model"],
            n_layer=config_data["n_layer"],
            vocab_size=config_data["vocab_size"],
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model


class ResidualBlock(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm = RMSNorm(d_model=args.d_model)
        self.mixer = MambaBlock(args)

    def forward(self, x: torch.Tensor):
        y = self.mixer(self.norm(x))
        y = y + x

        return y


class MambaBlock(nn.Module):
    """Source: https://github.com/johnma2006/mamba-minimal/blob/master/model.py"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # non-ssm stuff

        # projection for ssm stream and residual stream
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        assert isinstance(args.dt_rank, int)
        self.x_proj = nn.Linear(
            args.d_inner, args.dt_rank + args.d_state * 2, bias=False
        )

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = torch.arange(1, args.d_state + 1).repeat(args.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x: Input of shape (b, l, d)
        """
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )

        seq_l = x.shape[1]
        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :seq_l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)
        res = F.silu(res)

        y = self.s6_ssm(x)
        y = y * res
        y = self.out_proj(y)

        return y

    def s6_ssm(self, x: torch.Tensor) -> torch.Tensor:
        stuff = self.x_proj(x)

        A = -torch.exp(self.A_log)

        Delta, B, C = stuff.split(
            split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1
        )

        Delta = self.dt_proj(Delta)
        Delta = F.softplus(Delta)

        y = self.selective_scan(x, Delta, A, B, C, self.D)

        return y

    @staticmethod
    def selective_scan(
        u: torch.Tensor,
        Delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        (b, seq_l, d_inner) = u.shape
        d_state = A.shape[1]

        bar_A = torch.exp(einsum(Delta, A, "b l d_in, d_in n -> b l d_in n"))
        bar_B_u = einsum(Delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        h = torch.zeros(b, d_inner, d_state)
        ys = []

        for i in range(seq_l):
            h = bar_A[:, i] * h + bar_B_u[:, i]
            y = einsum(C[:, i], h, "b n, b d_in n -> b d_in")
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
