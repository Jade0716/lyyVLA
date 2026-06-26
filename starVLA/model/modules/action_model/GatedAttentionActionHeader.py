"""VLA-Adapter-style gated attention action head for token-condition inputs.

This head keeps the simple ``predict_action(condition_tokens)`` interface used
by the current ActionToken/TwoChunk frameworks, while replacing adaptive
pooling with learned action chunk queries and gated multi-head attention blocks.
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
    """Residual MLP block with self, task, and gated adapter attention."""

    def __init__(self, dim: int, num_heads: int = 8, use_rope: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by num_heads {num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

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
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
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
        task_condition: torch.Tensor | None = None,
        adapter_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, action_len, hidden_dim = x.shape

        q = self._to_heads(self.q_proj(x))
        k_self = self._to_heads(self.k_self(x))
        v_self = self._to_heads(self.v_self(x))

        if self.rope is not None:
            cos, sin = self.rope(seq_len=action_len, device=x.device, dtype=x.dtype)
            q, k_self = apply_rope(q, k_self, cos, sin)

        attn_scores = [torch.matmul(q, k_self.transpose(-2, -1))]
        values = [v_self]

        if task_condition is not None and task_condition.shape[1] > 0:
            k_task, v_task = self._project_condition(
                task_condition,
                self.k_condition,
                self.v_condition,
            )
            attn_scores.append(torch.matmul(q, k_task.transpose(-2, -1)))
            values.append(v_task)

        if adapter_condition is not None and adapter_condition.shape[1] > 0:
            k_adapter, v_adapter = self._project_condition(
                adapter_condition,
                self.k_condition,
                self.v_condition,
            )
            ratio_g = torch.tanh(self.gating_factor)
            attn_scores.append(torch.matmul(q, k_adapter.transpose(-2, -1)) * ratio_g)
            values.append(v_adapter)

        scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        value = torch.cat(values, dim=2)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch, action_len, hidden_dim)
        output = self.o_proj(output)
        return self.ffn(output + x)


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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.adapter_token_count = adapter_token_count

        query_dim = input_dim * action_dim
        self.action_chunk_embeddings = nn.Parameter(torch.zeros(NUM_ACTIONS_CHUNK, query_dim))
        nn.init.normal_(self.action_chunk_embeddings, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.condition_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GatedAttentionBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    use_rope=use_rope,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        condition = self.condition_proj(actions_hidden_states)
        adapter_condition = None
        task_condition = condition
        if self.adapter_token_count is not None and self.adapter_token_count > 0:
            adapter_count = min(int(self.adapter_token_count), condition.shape[1])
            adapter_condition = condition[:, :adapter_count, :]
            task_condition = condition[:, adapter_count:, :]
        query = self.action_chunk_embeddings.to(
            device=actions_hidden_states.device,
            dtype=actions_hidden_states.dtype,
        )
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.query_proj(self.query_norm(query))
        for block in self.blocks:
            x = block(
                x,
                task_condition=task_condition,
                adapter_condition=adapter_condition,
            )
        return self.output_proj(self.output_norm(x))

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)
