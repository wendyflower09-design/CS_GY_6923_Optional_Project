from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import mup


@dataclass
class UPTransformerConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool = True


class UPCausalSelfAttention(nn.Module):
    def __init__(self, config: UPTransformerConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd_dim = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_dim)

        y = self.resid_dropout(self.c_proj(y))

        return y


class UPMLP(nn.Module):
    def __init__(self, config: UPTransformerConfig):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class UPBlock(nn.Module):
    def __init__(self, config: UPTransformerConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = UPCausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = UPMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class UPTransformerLM(nn.Module):
    def __init__(self, config: UPTransformerConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([UPBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = mup.MuReadout(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / ((2 * config.n_layer) ** 0.5),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = idx.size()

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence length {seq_len}, "
                f"block_size is only {self.config.block_size}."
            )

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss


def build_up_model_with_base_shapes(
    model_config_dict,
    vocab_size: int,
) -> UPTransformerLM:
    target_config = UPTransformerConfig(
        vocab_size=vocab_size,
        block_size=model_config_dict["block_size"],
        n_layer=model_config_dict["n_layer"],
        n_head=model_config_dict["n_head"],
        n_embd=model_config_dict["n_embd"],
        dropout=model_config_dict["dropout"],
        bias=model_config_dict["bias"],
    )

    base_width = model_config_dict["n_head"] * 16
    delta_width = model_config_dict["n_head"] * 32

    base_config = UPTransformerConfig(
        vocab_size=vocab_size,
        block_size=model_config_dict["block_size"],
        n_layer=model_config_dict["n_layer"],
        n_head=model_config_dict["n_head"],
        n_embd=base_width,
        dropout=model_config_dict["dropout"],
        bias=model_config_dict["bias"],
    )

    delta_config = UPTransformerConfig(
        vocab_size=vocab_size,
        block_size=model_config_dict["block_size"],
        n_layer=model_config_dict["n_layer"],
        n_head=model_config_dict["n_head"],
        n_embd=delta_width,
        dropout=model_config_dict["dropout"],
        bias=model_config_dict["bias"],
    )

    base_model = UPTransformerLM(base_config)
    delta_model = UPTransformerLM(delta_config)
    target_model = UPTransformerLM(target_config)

    base_shapes = mup.make_base_shapes(
        base_model,
        delta_model,
        savefile=None,
    )

    mup.set_base_shapes(
        target_model,
        base_shapes,
    )

    return target_model
