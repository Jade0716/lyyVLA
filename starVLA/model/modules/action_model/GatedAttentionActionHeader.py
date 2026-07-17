"""Attention action head for token-condition inputs.

This head keeps the simple ``predict_action(condition_tokens)`` interface used
by the current ActionToken/TwoChunk frameworks, while replacing adaptive
pooling with learned action chunk queries and multi-head attention blocks.
"""

import math

import torch
import torch.nn as nn


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class GatedAttentionBlock(nn.Module):
    """Residual MLP block with action-query self attention and condition attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_rope: bool = True,
        condition_group_names: tuple[str, ...] | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by num_heads {num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.condition_group_names = tuple(condition_group_names or ())
        self.separate_condition_paths = len(self.condition_group_names) > 0

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_condition = nn.Linear(dim, dim)
        self.v_condition = nn.Linear(dim, dim)
        if self.separate_condition_paths:
            self.k_condition_groups = nn.ModuleDict(
                {name: nn.Linear(dim, dim) for name in self.condition_group_names}
            )
            self.v_condition_groups = nn.ModuleDict(
                {name: nn.Linear(dim, dim) for name in self.condition_group_names}
            )
        self.o_proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionEmbedding(self.head_dim) if use_rope else None

    def _to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_condition(
        self,
        condition: torch.Tensor,
        key_layer: nn.Linear,
        value_layer: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_len = condition.shape[1]
        key = self._to_heads(key_layer(condition))
        value = self._to_heads(value_layer(condition))
        if self.rope is not None:
            cos, sin = self.rope(seq_len=cond_len, device=condition.device, dtype=condition.dtype)
            _, key = apply_rope(key, key, cos, sin)
        return key, value

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        condition_groups: dict[str, torch.Tensor] | None = None,
        attention_debug_spans: dict[str, tuple[int, int]] | None = None,
        condition_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        batch, action_len, hidden_dim = x.shape

        q = self._to_heads(self.q_proj(x))
        k_self = self._to_heads(self.k_self(x))
        v_self = self._to_heads(self.v_self(x))

        if self.rope is not None:
            cos, sin = self.rope(seq_len=action_len, device=x.device, dtype=x.dtype)
            q, k_self = apply_rope(q, k_self, cos, sin)

        attn_scores = [torch.matmul(q, k_self.transpose(-2, -1))]
        values = [v_self]

        group_lens: dict[str, int] = {}
        if self.separate_condition_paths and condition_groups:
            for name in self.condition_group_names:
                group = condition_groups.get(name)
                if group is None or group.shape[1] == 0:
                    group_lens[name] = 0
                    continue
                group = group.to(device=x.device, dtype=x.dtype)
                k_group, v_group = self._project_condition(
                    group,
                    self.k_condition_groups[name],
                    self.v_condition_groups[name],
                )
                attn_scores.append(torch.matmul(q, k_group.transpose(-2, -1)))
                values.append(v_group)
                group_lens[name] = int(group.shape[1])
        elif condition is not None and condition.shape[1] > 0:
            k_condition, v_condition = self._project_condition(
                condition,
                self.k_condition,
                self.v_condition,
            )
            attn_scores.append(torch.matmul(q, k_condition.transpose(-2, -1)))
            values.append(v_condition)

        scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        if condition_attention_mask is not None:
            if self.separate_condition_paths and condition_groups:
                raise ValueError(
                    "condition_attention_mask is only supported for the shared condition path."
                )
            if condition is None or condition_attention_mask.shape != condition.shape[:2]:
                raise ValueError(
                    "condition_attention_mask must have shape [B, condition_len], got "
                    f"{tuple(condition_attention_mask.shape)} for condition "
                    f"{None if condition is None else tuple(condition.shape)}."
                )
            self_mask = torch.ones(
                (batch, action_len),
                dtype=torch.bool,
                device=x.device,
            )
            key_mask = torch.cat(
                [self_mask, condition_attention_mask.to(device=x.device, dtype=torch.bool)],
                dim=1,
            )
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        value = torch.cat(values, dim=2)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch, action_len, hidden_dim)
        output = self.o_proj(output)
        output = self.ffn(output + x)
        if attention_debug_spans is None:
            return output

        stats = {
            "self": float(weights[..., :action_len].sum(dim=-1).mean().detach().float().cpu().item())
        }
        if self.separate_condition_paths and condition_groups:
            offset = action_len
            for name in self.condition_group_names:
                length = int(group_lens.get(name, 0))
                if length <= 0:
                    stats[name] = 0.0
                    continue
                stats[name] = float(
                    weights[..., offset : offset + length]
                    .sum(dim=-1)
                    .mean()
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                offset += length
            if attention_debug_spans is not None:
                for name in attention_debug_spans:
                    stats.setdefault(name, 0.0)
            return output, stats

        condition_offset = action_len
        for name, span in attention_debug_spans.items():
            start, end = int(span[0]), int(span[1])
            if end <= start:
                stats[name] = 0.0
                continue
            stats[name] = float(
                weights[..., condition_offset + start : condition_offset + end]
                .sum(dim=-1)
                .mean()
                .detach()
                .float()
                .cpu()
                .item()
            )
        return output, stats


class GatedAttentionActionHead(nn.Module):
    """Predict a fixed action chunk from condition tokens via gated attention."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        action_dim: int = 7,
        NUM_ACTIONS_CHUNK: int = 8,
        num_blocks: int = 8,
        num_heads: int = 8,
        use_rope: bool = True,
        adapter_token_count: int | None = None,
        coarse_condition_query: bool = False,
        coarse_action_side_tokens: bool = False,
        separate_condition_paths: bool = False,
        condition_group_names: tuple[str, ...] | list[str] | None = None,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        # Kept for backward-compatible configs. The current head attends over
        # all condition tokens directly instead of splitting a gated adapter path.
        self.adapter_token_count = adapter_token_count
        self.coarse_condition_query = bool(coarse_condition_query)
        self.coarse_action_side_tokens = bool(coarse_action_side_tokens)
        self.separate_condition_paths = bool(separate_condition_paths)
        self.condition_group_names = tuple(
            condition_group_names or ("action_token", "coarse_idct", "memory", "dino")
        )
        self.last_attention_debug = None

        query_dim = input_dim * action_dim
        self.action_chunk_embeddings = nn.Parameter(torch.zeros(NUM_ACTIONS_CHUNK, query_dim))
        nn.init.normal_(self.action_chunk_embeddings, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        if self.coarse_condition_query:
            self.coarse_query_proj = nn.Linear(action_dim, hidden_dim)
        if self.coarse_action_side_tokens:
            self.coarse_action_side_proj = nn.Linear(action_dim, hidden_dim)
            self.coarse_action_side_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.condition_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GatedAttentionBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    use_rope=use_rope,
                    condition_group_names=self.condition_group_names
                    if self.separate_condition_paths
                    else None,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        coarse_actions: torch.Tensor | None = None,
        condition_groups: dict[str, torch.Tensor] | None = None,
        attention_debug_spans: dict[str, tuple[int, int]] | None = None,
        condition_layers: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        condition_group_layers: list[dict[str, torch.Tensor]] | tuple[dict[str, torch.Tensor], ...] | None = None,
        condition_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        use_layer_groups = self.separate_condition_paths and condition_group_layers is not None
        condition = None if use_layer_groups else self.condition_proj(actions_hidden_states)
        if self.separate_condition_paths and condition_groups and not use_layer_groups:
            condition_groups = {
                name: self.condition_proj(group)
                for name, group in condition_groups.items()
                if group is not None
            }
        if condition_layers is not None and not use_layer_groups:
            condition_layers = [self.condition_proj(layer) for layer in condition_layers]
        if use_layer_groups:
            projected_groups = {}
            projected_group_layers = []
            for layer_groups in condition_group_layers:
                projected_layer = {}
                for name, group in layer_groups.items():
                    if group is None:
                        continue
                    cache_key = id(group)
                    if cache_key not in projected_groups:
                        projected_groups[cache_key] = self.condition_proj(group)
                    projected_layer[name] = projected_groups[cache_key]
                projected_group_layers.append(projected_layer)
            condition_group_layers = projected_group_layers
        query = self.action_chunk_embeddings.to(
            device=actions_hidden_states.device,
            dtype=actions_hidden_states.dtype,
        )
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.query_proj(self.query_norm(query))
        if coarse_actions is not None:
            if not (self.coarse_condition_query or self.coarse_action_side_tokens):
                raise ValueError(
                    "coarse_actions were provided but both coarse_condition_query and "
                    "coarse_action_side_tokens are disabled."
                )
            if coarse_actions.shape[:2] != x.shape[:2] or coarse_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    "coarse_actions must have shape [B, NUM_ACTIONS_CHUNK, action_dim], "
                    f"got {tuple(coarse_actions.shape)} for query shape {tuple(x.shape)} "
                    f"and action_dim={self.action_dim}."
                )
            coarse_actions = coarse_actions.to(device=x.device, dtype=x.dtype)
            if self.coarse_condition_query:
                x = x + self.coarse_query_proj(coarse_actions)
            if self.coarse_action_side_tokens:
                coarse_tokens = self.coarse_action_side_proj(coarse_actions)
                type_embedding = self.coarse_action_side_type_embedding.to(
                    device=coarse_tokens.device,
                    dtype=coarse_tokens.dtype,
                )
                x = torch.cat([x, coarse_tokens + type_embedding], dim=1)
        elif self.coarse_action_side_tokens:
            raise ValueError("coarse_action_side_tokens=true requires coarse_actions.")

        attention_debug_layers = []
        for layer_idx, block in enumerate(self.blocks):
            block_condition = condition
            if condition_layers is not None:
                block_condition = condition_layers[min(layer_idx, len(condition_layers) - 1)]
            block_condition_groups = condition_groups if self.separate_condition_paths else None
            if self.separate_condition_paths and condition_group_layers is not None:
                block_condition_groups = condition_group_layers[min(layer_idx, len(condition_group_layers) - 1)]
            if attention_debug_spans is None:
                x = block(
                    x,
                    condition=block_condition,
                    condition_groups=block_condition_groups,
                    condition_attention_mask=condition_attention_mask,
                )
            else:
                x, layer_stats = block(
                    x,
                    condition=block_condition,
                    condition_groups=block_condition_groups,
                    condition_attention_mask=condition_attention_mask,
                    attention_debug_spans=attention_debug_spans,
                )
                attention_debug_layers.append(layer_stats)
        x = x[:, : self.NUM_ACTIONS_CHUNK, :]
        actions = self.output_proj(self.output_norm(x))
        self.last_attention_debug = attention_debug_layers if attention_debug_spans is not None else None
        return actions

    def forward(
        self,
        actions_hidden_states: torch.Tensor,
        coarse_actions: torch.Tensor | None = None,
        condition_groups: dict[str, torch.Tensor] | None = None,
        attention_debug_spans: dict[str, tuple[int, int]] | None = None,
        condition_layers: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        condition_group_layers: list[dict[str, torch.Tensor]] | tuple[dict[str, torch.Tensor], ...] | None = None,
        condition_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_action(
            actions_hidden_states,
            coarse_actions=coarse_actions,
            condition_groups=condition_groups,
            condition_attention_mask=condition_attention_mask,
            attention_debug_spans=attention_debug_spans,
            condition_layers=condition_layers,
            condition_group_layers=condition_group_layers,
        )
