# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Flow-matching action head that uses DiT.forward's built-in cross/self attention
interleaving.

This differs from LayerwiseFM_ActionHeader:
  - LayerwiseFM manually loops over transformer_blocks and passes
    encoder_hidden_states on every block, so every block acts as cross-attn.
  - This head calls DiT.forward(...), so when diffusion_model_cfg has
    interleave_self_attention=True, odd blocks run self-attention and even
    blocks run cross-attention as implemented in cross_attention_dit.py.

The filename intentionally follows the requested name: crossself_AcitonHeader.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        return self.layer3(x)


@dataclass
class CrossSelfFlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    state_dim: int = field(default=7)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=4)
    num_target_vision_tokens: int = field(default=32)
    vl_condition_mode: str = field(default="last")
    vl_condition_layer_index: int = field(default=-1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class CrossSelfFlowmatchingActionHead(nn.Module):
    """
    Flow-matching head with DiT-managed cross/self attention interleaving.

    If `vl_embs` is a list, the conditioning tensor is selected according to:
      - vl_condition_mode="last"   : use vl_embs[vl_condition_layer_index]
      - vl_condition_mode="mean"   : average all layer tensors
      - vl_condition_mode="concat" : concatenate layers on sequence dimension

    `concat` preserves more VLM layers but can be much heavier.
    """

    def __init__(self, full_config, **kwargs):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        action_model_type = config.get("action_model_type", None)
        if action_model_type in DiTConfig:
            diffusion_model_cfg = {**DiTConfig[action_model_type], **diffusion_model_cfg}

        if diffusion_model_cfg.get("input_embedding_dim", None) is None:
            raise ValueError(
                "crossself_AcitonHeader requires diffusion_model_cfg.input_embedding_dim "
                "or action_model_type in {'DiT-B', 'DiT-L'}."
            )
        diffusion_model_cfg.setdefault("interleave_self_attention", True)

        self.input_embedding_dim = int(diffusion_model_cfg["input_embedding_dim"])
        self.model = DiT(**diffusion_model_cfg)
        self.dit_out_hidden_size = int(self.model.config.output_dim)

        self.hidden_size = int(config.get("hidden_size", self.input_embedding_dim))
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.num_inference_timesteps = int(config.num_inference_timesteps)
        self.num_timestep_buckets = int(config.num_timestep_buckets)

        self.vl_condition_mode = config.get("vl_condition_mode", "last")
        self.vl_condition_layer_index = int(config.get("vl_condition_layer_index", -1))

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.get("state_dim", None)
            else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.dit_out_hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _select_encoder_hidden_states(self, vl_embs, encoder_attention_mask=None):
        if not isinstance(vl_embs, (list, tuple)):
            return vl_embs, encoder_attention_mask

        if len(vl_embs) == 0:
            raise ValueError("vl_embs list is empty.")

        if self.vl_condition_mode == "last":
            return vl_embs[self.vl_condition_layer_index], encoder_attention_mask

        if self.vl_condition_mode == "mean":
            return torch.stack(list(vl_embs), dim=0).mean(dim=0), encoder_attention_mask

        if self.vl_condition_mode == "concat":
            encoder_hidden_states = torch.cat(list(vl_embs), dim=1)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.repeat(1, len(vl_embs))
            return encoder_hidden_states, encoder_attention_mask

        raise ValueError(
            f"Unsupported vl_condition_mode={self.vl_condition_mode!r}; "
            "expected one of {'last', 'mean', 'concat'}."
        )

    def _build_sa_embeddings(self, actions, timesteps, state_features):
        action_features = self.action_encoder(actions, timesteps)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        if state_features is not None:
            return torch.cat((state_features, future_tokens, action_features), dim=1)
        return torch.cat((future_tokens, action_features), dim=1)

    def forward(self, vl_embs, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        """
        vl_embs: Tensor [B, seq, H] or list of layer tensors [B, seq, H]
        actions: Tensor [B, action_horizon, action_dim]
        """
        encoder_hidden_states, encoder_attention_mask = self._select_encoder_hidden_states(
            vl_embs, encoder_attention_mask
        )

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        state_features = self.state_encoder(state) if state is not None else None
        sa_embs = self._build_sa_embeddings(noisy_trajectory, t_discretized, state_features)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(self, vl_embs, state: torch.Tensor = None, encoder_attention_mask=None) -> torch.Tensor:
        encoder_hidden_states, encoder_attention_mask = self._select_encoder_hidden_states(
            vl_embs, encoder_attention_mask
        )
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=encoder_hidden_states.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
            )
            sa_embs = self._build_sa_embeddings(actions, timesteps_tensor, state_features)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    return CrossSelfFlowmatchingActionHead(full_config=config)
