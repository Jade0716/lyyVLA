import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_StateMemory")
class Qwen_GR00T_ActionToken_TwoChunk_StateMemory(Qwen_GR00T_ActionToken_TwoChunk):
    """TwoChunk with a compact causal state-history bank for the fast controller."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        memory_cfg = self.config.framework.get("state_memory", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.state_memory_stride = int(memory_cfg.get("stride", self.vision_refresh_steps))
        self.state_memory_recent_size = int(memory_cfg.get("recent_size", 4))
        self.state_memory_embed_dim = int(memory_cfg.get("embed_dim", 128))
        self.state_memory_noise_std = float(memory_cfg.get("noise_std", 0.02))
        self.state_memory_noise_ar = float(memory_cfg.get("noise_ar", 0.9))
        self.state_memory_noise_enabled = bool(memory_cfg.get("noise_enabled", True))
        self.state_memory_preserve_gripper = bool(memory_cfg.get("preserve_gripper", True))
        if self.state_memory_recent_size <= 0:
            raise ValueError(
                f"state_memory.recent_size must be positive, got {self.state_memory_recent_size}."
            )
        if not 0.0 <= self.state_memory_noise_ar < 1.0:
            raise ValueError(
                f"state_memory.noise_ar must be in [0, 1), got {self.state_memory_noise_ar}."
            )
        if self.state_memory_stride != self.vision_refresh_steps:
            raise ValueError(
                "State memory must update at the fast-controller refresh rate: "
                f"state_memory.stride={self.state_memory_stride}, "
                f"vision_refresh_steps={self.vision_refresh_steps}."
            )

        state_dim = int(self.config.framework.action_model.state_dim)
        embed_dim = self.state_memory_embed_dim
        self.state_memory_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_summary_gru = nn.GRUCell(embed_dim, embed_dim)
        self.state_summary_projection = nn.Linear(embed_dim, hidden_size)
        self.state_recent_projection = nn.Linear(embed_dim, hidden_size)
        self.state_memory_age_embedding = nn.Parameter(
            torch.randn(self.state_memory_recent_size, hidden_size) * 0.02
        )
        self.state_memory_type_embedding = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.state_memory_valid_embedding = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.state_memory_empty_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        self._state_memory_recent = None
        self._state_memory_summary = None
        self._state_memory_summary_valid = None

    def _augment_state_history(self, states: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.state_memory_noise_enabled or states.shape[0] == 0:
            return states

        noise = torch.zeros_like(states)
        running = torch.zeros_like(states[0])
        innovation_scale = max(1.0 - self.state_memory_noise_ar**2, 0.0) ** 0.5
        for index in range(states.shape[0]):
            epsilon = torch.randn_like(running) * self.state_memory_noise_std
            running = self.state_memory_noise_ar * running + innovation_scale * epsilon
            noise[index] = running
        if self.state_memory_preserve_gripper and noise.shape[-1] > 0:
            noise[:, -1] = 0
        return states + noise

    def _memory_tokens_from_parts(
        self,
        summary: torch.Tensor,
        recent_states: torch.Tensor,
        summary_valid: bool,
    ) -> torch.Tensor:
        hidden_size = self.state_memory_empty_token.shape[-1]
        tokens = self.state_memory_empty_token.new_zeros(
            1 + self.state_memory_recent_size,
            hidden_size,
        )

        summary_token = self.state_summary_projection(summary)
        summary_token = summary_token + self.state_memory_type_embedding[0]
        summary_token = summary_token + self.state_memory_valid_embedding[int(summary_valid)]
        if not summary_valid:
            summary_token = summary_token + self.state_memory_empty_token[0]
        tokens[0] = summary_token

        recent_count = min(recent_states.shape[0], self.state_memory_recent_size)
        pad_count = self.state_memory_recent_size - recent_count
        tokens[1:] = (
            self.state_memory_empty_token
            + self.state_memory_type_embedding[1]
            + self.state_memory_valid_embedding[0]
            + self.state_memory_age_embedding
        )
        if recent_count:
            recent_states = recent_states[-recent_count:]
            recent_embed = self.state_memory_encoder(recent_states)
            recent_tokens = self.state_recent_projection(recent_embed)
            recent_tokens = recent_tokens + self.state_memory_type_embedding[1]
            recent_tokens = recent_tokens + self.state_memory_valid_embedding[1]
            recent_tokens = recent_tokens + self.state_memory_age_embedding[pad_count:]
            tokens[1 + pad_count :] = recent_tokens
        return tokens

    def _encode_state_history(self, states: torch.Tensor) -> torch.Tensor:
        old_states = states[:-self.state_memory_recent_size]
        recent_states = states[-self.state_memory_recent_size :]

        summary = states.new_zeros(self.state_memory_embed_dim)
        for state in old_states:
            summary = self.state_summary_gru(
                self.state_memory_encoder(state.unsqueeze(0)).squeeze(0),
                summary,
            )
        return self._memory_tokens_from_parts(
            summary=summary,
            recent_states=recent_states,
            summary_valid=old_states.shape[0] > 0,
        )

    def _training_state_memory_tokens(
        self,
        examples: List[dict],
        num_refreshes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        per_sample_histories = []
        per_sample_sequences = []
        for example in examples:
            history = torch.as_tensor(example["state_history"], device=device, dtype=dtype)
            sequence = torch.as_tensor(example["state_memory_sequence"], device=device, dtype=dtype)
            combined = self._augment_state_history(torch.cat([history, sequence[:num_refreshes]], dim=0))
            per_sample_histories.append(combined[: history.shape[0]])
            per_sample_sequences.append(combined[history.shape[0] :])

        refresh_major_tokens = []
        for refresh_index in range(num_refreshes):
            for sample_index in range(len(examples)):
                history = per_sample_histories[sample_index]
                if refresh_index:
                    history = torch.cat(
                        [history, per_sample_sequences[sample_index][:refresh_index]],
                        dim=0,
                    )
                refresh_major_tokens.append(self._encode_state_history(history))
        return torch.stack(refresh_major_tokens, dim=0)

    def _build_action_condition(
        self,
        action_token_hidden: torch.Tensor,
        frame_images: List,
        dino_image_tensors: torch.Tensor | None = None,
        state_memory_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition_hidden = super()._build_action_condition(
            action_token_hidden=action_token_hidden,
            frame_images=frame_images,
            dino_image_tensors=dino_image_tensors,
        )
        if state_memory_tokens is None:
            return condition_hidden
        return torch.cat(
            [
                condition_hidden,
                state_memory_tokens.to(
                    device=condition_hidden.device,
                    dtype=condition_hidden.dtype,
                ),
            ],
            dim=1,
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        if examples and (
            "image_sequence" not in examples[0]
            or "state_history" not in examples[0]
        ):
            return super().forward(examples=examples, **kwargs)

        image_sequences = [example["image_sequence"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
        num_refreshes = self._valid_training_refreshes(image_sequences, actions)
        if num_refreshes == 0:
            raise ValueError("StateMemory TwoChunk received no valid action chunks.")

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)
        long_chunk_len = min(self.motion_dct_chunk_len, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(
            action_token_hidden,
            actions,
            long_chunk_len,
        )
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss
        action_loss_weight = self._action_loss_weight(kwargs.get("train_step", None))

        coarse_long_action = self._predict_coarse_action(action_token_hidden, long_chunk_len)
        if self.detach_idct_condition:
            coarse_long_action = coarse_long_action.detach()
        coarse_long_action = self._coarse_with_gripper_pad(
            coarse_long_action,
            action_dim=actions.shape[-1],
        )

        flat_frame_images = []
        residual_action_targets = []
        for refresh_index in range(num_refreshes):
            start = refresh_index * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            flat_frame_images.extend(
                [image_sequence[refresh_index] for image_sequence in image_sequences]
            )
            residual_action_targets.append(
                actions[:, start:end, :] - coarse_long_action[:, start:end, :]
            )

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        state_memory_tokens = self._training_state_memory_tokens(
            examples=examples,
            num_refreshes=num_refreshes,
            device=flat_action_token_hidden.device,
            dtype=flat_action_token_hidden.dtype,
        )
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            state_memory_tokens=state_memory_tokens,
        )
        residual_action_targets = torch.cat(residual_action_targets, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_residual_actions = self.action_model.predict_action(fused_hidden)
            action_loss = self.l1_loss(pred_residual_actions, residual_action_targets)

        total_loss = action_loss_weight * action_loss + weighted_motion_dct_loss
        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "action_dit_loss_weight": action_loss_weight,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
        }

    def _reset_predict_cache(self) -> None:
        super()._reset_predict_cache()
        self._state_memory_recent = None
        self._state_memory_summary = None
        self._state_memory_summary_valid = None

    def _ensure_predict_state_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._state_memory_recent is not None and len(self._state_memory_recent) == batch_size:
            return
        self._state_memory_recent = [[] for _ in range(batch_size)]
        self._state_memory_summary = torch.zeros(
            batch_size,
            self.state_memory_embed_dim,
            device=device,
            dtype=dtype,
        )
        self._state_memory_summary_valid = [False] * batch_size

    def _read_predict_state_memory(self) -> torch.Tensor:
        tokens = []
        for batch_index, recent in enumerate(self._state_memory_recent):
            if recent:
                recent_tensor = torch.stack(recent, dim=0)
            else:
                recent_tensor = self._state_memory_summary.new_empty(
                    0,
                    int(self.config.framework.action_model.state_dim),
                )
            tokens.append(
                self._memory_tokens_from_parts(
                    summary=self._state_memory_summary[batch_index],
                    recent_states=recent_tensor,
                    summary_valid=self._state_memory_summary_valid[batch_index],
                )
            )
        return torch.stack(tokens, dim=0)

    def _commit_predict_states(self, examples: List[dict]) -> None:
        for batch_index, example in enumerate(examples):
            state = torch.as_tensor(
                example["state"],
                device=self._state_memory_summary.device,
                dtype=self._state_memory_summary.dtype,
            ).reshape(-1, int(self.config.framework.action_model.state_dim))[-1]
            recent = self._state_memory_recent[batch_index]
            if len(recent) == self.state_memory_recent_size:
                oldest = recent.pop(0)
                encoded = self.state_memory_encoder(oldest.unsqueeze(0)).squeeze(0)
                self._state_memory_summary[batch_index] = self.state_summary_gru(
                    encoded,
                    self._state_memory_summary[batch_index],
                )
                self._state_memory_summary_valid[batch_index] = True
            recent.append(state.detach())

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        if type(examples) is not list:
            examples = [examples]
        if examples and "image_sequence" in examples[0]:
            return super().predict_action(examples, **kwargs)

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        reset_cache = bool(kwargs.get("reset_cache", False))
        if reset_cache:
            self._reset_predict_cache()
        batch_size = len(examples)
        slow_refresh = self._should_refresh_action_token(instructions, batch_size)
        slow_time_s = 0.0
        if slow_refresh:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
            )
            self._sync_cuda_if_needed()
            slow_start = time.perf_counter()
            self._cached_action_token_hidden = self._encode_action_token_hidden(
                batch_images,
                instructions,
                qwen_inputs=qwen_inputs,
            ).detach()
            self._cached_coarse_action = self._predict_coarse_action(
                self._cached_action_token_hidden,
                self.motion_dct_chunk_len,
            ).detach()
            self._cached_instruction_key = tuple(instructions)
            self._cached_batch_size = batch_size
            self._predict_call_count = 0
            self._sync_cuda_if_needed()
            slow_time_s = time.perf_counter() - slow_start

        start = self._predict_call_count * self.fast_chunk_size
        end = start + self.fast_chunk_size
        if end > self.motion_dct_chunk_len:
            self._predict_call_count = 0
            start = 0
            end = self.fast_chunk_size

        self._ensure_predict_state_memory(
            batch_size=batch_size,
            device=self._cached_action_token_hidden.device,
            dtype=self._cached_action_token_hidden.dtype,
        )
        state_memory_tokens = self._read_predict_state_memory()
        dino_image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        self._sync_cuda_if_needed()
        fast_start = time.perf_counter()
        fused_hidden = self._build_action_condition(
            self._cached_action_token_hidden,
            batch_images,
            dino_image_tensors=dino_image_tensors,
            state_memory_tokens=state_memory_tokens,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_residual_actions = self.action_model.predict_action(fused_hidden)
        coarse_action = self._coarse_with_gripper_pad(
            self._cached_coarse_action,
            pred_residual_actions.shape[-1],
        )
        pred_actions = pred_residual_actions + coarse_action[:, start:end, :]
        self._sync_cuda_if_needed()
        fast_time_s = time.perf_counter() - fast_start

        self._commit_predict_states(examples)
        self._predict_call_count += 1
        return {
            "normalized_actions": pred_actions.detach().float().cpu().numpy(),
            "inference_timing": {
                "slow_refresh": slow_refresh,
                "slow_time_s": slow_time_s,
                "fast_time_s": fast_time_s,
                "model_inference_time_s": slow_time_s + fast_time_s,
                "timing_scope": "model_only_after_preprocess",
                "fast_chunk_size": self.fast_chunk_size,
                "language_refresh_steps": self.language_refresh_steps,
                "vision_refresh_steps": self.vision_refresh_steps,
                "state_memory_recent_size": self.state_memory_recent_size,
            },
        }
