"""A small, hookable transformer (Nanda-style); hooks are named by full module path."""

from __future__ import annotations

import torch
from torch import nn

from grokking_tda.data.modular import TaskMeta
from grokking_tda.models.hooks import HookedModule, HookPoint

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.hook_z = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, seq, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(n, seq, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        attn = scores.softmax(dim=-1)
        z = attn @ v  # (n, heads, seq, d_head)
        z = self.hook_z(z.transpose(1, 2).reshape(n, seq, -1))
        return self.W_O(z)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act: str) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.act = _ACTIVATIONS[act]()
        self.fc_out = nn.Linear(d_mlp, d_model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.hook_post(self.act(self.hook_pre(self.fc_in(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, act: str, use_ln: bool) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp, act)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hook_attn_out(self.attn(self.ln1(x)))
        x = x + self.hook_mlp_out(self.mlp(self.ln2(x)))
        return x


class GrokkingTransformer(HookedModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_mlp: int = 512,
        act: str = "relu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.hook_embed = HookPoint()
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_mlp, act, use_layernorm)
            for _ in range(n_layers)
        )
        self.ln_final = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.hook_resid_final = HookPoint()
        self.unembed = nn.Linear(d_model, num_classes, bias=False)
        self.modulus = num_classes
        self.hidden_hook = "hook_resid_final"  # what AnalysisCfg.representation="hidden" reads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.hook_embed(self.embed(tokens) + self.pos_embed[: tokens.shape[1]])
        for block in self.blocks:
            x = block(x)
        final = self.hook_resid_final(self.ln_final(x)[:, -1, :])  # answer position
        return self.unembed(final)

    def embedding_matrix(self) -> torch.Tensor:
        # the p residue rows; the slice drops the "=" token at index p
        return self.embed.weight[: self.modulus].detach()


def build_transformer(model_cfg, meta: TaskMeta) -> GrokkingTransformer:
    return GrokkingTransformer(
        vocab_size=meta.vocab_size,
        num_classes=meta.num_classes,
        seq_len=meta.seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_mlp=model_cfg.d_mlp,
        act=model_cfg.act,
        use_layernorm=model_cfg.use_layernorm,
    )
