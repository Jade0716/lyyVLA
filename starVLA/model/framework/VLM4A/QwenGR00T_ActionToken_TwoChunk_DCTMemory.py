import math
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dct, idct

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import _as_bool
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _as_noise_range(value) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"noise range must contain exactly two values, got {value}.")
    low, high = float(parts[0]), float(parts[1])
    if low < 0 or high < 0 or high < low:
        raise ValueError(f"invalid noise range {value}; expected 0 <= low <= high.")
    return low, high


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_DCTMemory")
class Qwen_GR00T_ActionToken_TwoChunk_DCTMemory(Qwen_GR00T_ActionToken_TwoChunk):
    """TwoChunk with cached DCT motion memory tokens.

    The dataloader provides clean GT memory during training:
      - dct_memory_summary: [summary_keep_freq, action_dim]
      - dct_memory_recent:  [chunk_keep_freq, action_dim]

    During inference, an online bank is updated from previously predicted action
    chunks. Memory tokens are appended to the fast action-head condition, leaving
    the slow VLM action-token path unchanged.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        memory_cfg = self.config.framework.get("dct_memory", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)

        self.dct_memory_chunk_len = int(memory_cfg.get("chunk_len", 32))
        self.dct_memory_chunk_keep_freq = int(memory_cfg.get("chunk_keep_freq", 4))
        self.dct_memory_summary_keep_freq = int(memory_cfg.get("summary_keep_freq", 8))
        self.dct_memory_action_dim = int(memory_cfg.get("action_dim", action_dim))
        self.dct_memory_recent_mode = str(memory_cfg.get("recent_mode", "dct"))
        self.dct_memory_raw_recent = self.dct_memory_recent_mode in {"raw", "raw_actions"}
        self.dct_memory_noise_enabled = _as_bool(memory_cfg.get("noise_enabled", True))
        self.dct_memory_summary_noise_std = float(memory_cfg.get("summary_noise_std", memory_cfg.get("noise_std", 0.01)))
        self.dct_memory_recent_noise_std = float(memory_cfg.get("recent_noise_std", memory_cfg.get("noise_std", 0.01)))
        self.dct_memory_summary_noise_std_range = _as_noise_range(memory_cfg.get("summary_noise_std_range", None))
        self.dct_memory_recent_noise_std_range = _as_noise_range(memory_cfg.get("recent_noise_std_range", None))
        self.dct_memory_count_embed_dim = int(memory_cfg.get("summary_count_embed_dim", 128))
        self.coarse_condition_query = bool(getattr(self.action_model, "coarse_condition_query", False))

        self.dct_summary_proj = nn.Linear(self.dct_memory_action_dim, hidden_size, bias=False)
        self.dct_recent_proj = nn.Linear(self.dct_memory_action_dim, hidden_size, bias=False)
        self.dct_summary_count_mlp = nn.Sequential(
            nn.Linear(self.dct_memory_count_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dct_memory_type_embedding = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        if self.dct_memory_raw_recent:
            self.dct_recent_position_embedding = nn.Parameter(
                torch.randn(self.dct_memory_chunk_len, hidden_size) * 0.02
            )

        self._online_summary_dct = None
        self._online_recent_dct = None
        self._online_recent_actions = None
        self._online_pending_actions = None
        self._online_summary_pending_actions = None
        self._online_completed_chunks = 0
        self._dct_basis_cache = nn.ModuleDict()

    def _reset_predict_cache(self) -> None:
        super()._reset_predict_cache()
        self._online_summary_dct = None
        self._online_recent_dct = None
        self._online_recent_actions = None
        self._online_pending_actions = None
        self._online_summary_pending_actions = None
        self._online_completed_chunks = 0

    def _dct_matrix(self, length: int, keep_freq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = f"dct_{length}_{keep_freq}"
        if not hasattr(self, key):
            basis = dct(np.eye(length, dtype=np.float32), type=2, axis=0, norm="ortho")[:keep_freq]
            self.register_buffer(key, torch.from_numpy(basis), persistent=False)
        return getattr(self, key).to(device=device, dtype=dtype)

    def _idct_matrix(self, length: int, keep_freq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = f"idct_{length}_{keep_freq}"
        if not hasattr(self, key):
            coeff_eye = np.zeros((keep_freq, length), dtype=np.float32)
            coeff_eye[:, :keep_freq] = np.eye(keep_freq, dtype=np.float32)
            basis = idct(coeff_eye, type=2, n=length, axis=1, norm="ortho").T
            self.register_buffer(key, torch.from_numpy(basis), persistent=False)
        return getattr(self, key).to(device=device, dtype=dtype)

    def _torch_dct(self, signal: torch.Tensor, keep_freq: int) -> torch.Tensor:
        basis = self._dct_matrix(signal.shape[1], keep_freq, signal.device, signal.dtype)
        return torch.einsum("kt,btd->bkd", basis, signal)

    def _torch_idct(self, coeff: torch.Tensor, length: int, keep_freq: int) -> torch.Tensor:
        basis = self._idct_matrix(length, keep_freq, coeff.device, coeff.dtype)
        return torch.einsum("tk,bkd->btd", basis, coeff[:, :keep_freq, :])

    def _merge_summary_with_recent(
        self,
        summary_dct: torch.Tensor,
        summary_count: int,
        recent_dct: torch.Tensor,
    ) -> torch.Tensor:
        prev_len = int(summary_count) * self.dct_memory_chunk_len
        prev = self._torch_idct(summary_dct, prev_len, self.dct_memory_summary_keep_freq)
        recent = self._torch_idct(recent_dct, self.dct_memory_chunk_len, self.dct_memory_chunk_keep_freq)
        merged = torch.cat([prev, recent], dim=1)
        return self._torch_dct(merged, self.dct_memory_summary_keep_freq)

    def _examples_dct_memory(self, examples: List[dict], device: torch.device, dtype: torch.dtype):
        batch_size = len(examples)
        summary_shape = (batch_size, self.dct_memory_summary_keep_freq, self.dct_memory_action_dim)
        recent_len = self.dct_memory_chunk_len if self.dct_memory_raw_recent else self.dct_memory_chunk_keep_freq
        recent_shape = (batch_size, recent_len, self.dct_memory_action_dim)

        if not examples or "dct_memory_summary" not in examples[0]:
            summary = torch.zeros(summary_shape, device=device, dtype=dtype)
            recent = torch.zeros(recent_shape, device=device, dtype=dtype)
            summary_valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            recent_valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            summary_count = torch.zeros((batch_size,), device=device, dtype=dtype)
            return summary, recent, summary_valid, recent_valid, summary_count

        summary = torch.as_tensor(
            np.asarray([example["dct_memory_summary"] for example in examples]),
            device=device,
            dtype=dtype,
        )
        recent = torch.as_tensor(
            np.asarray([example["dct_memory_recent"] for example in examples]),
            device=device,
            dtype=dtype,
        )
        summary_valid = torch.as_tensor(
            np.asarray([example.get("dct_memory_summary_valid", False) for example in examples]),
            device=device,
            dtype=torch.bool,
        )
        recent_valid = torch.as_tensor(
            np.asarray([example.get("dct_memory_recent_valid", False) for example in examples]),
            device=device,
            dtype=torch.bool,
        )
        summary_count = torch.as_tensor(
            np.asarray([example.get("dct_memory_summary_count", 0) for example in examples]),
            device=device,
            dtype=dtype,
        )
        return summary, recent, summary_valid, recent_valid, summary_count

    def _summary_count_embedding(
        self,
        summary_count: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.log1p(summary_count.float()).unsqueeze(-1)
        half_dim = self.dct_memory_count_embed_dim // 2
        if half_dim <= 0:
            raise ValueError(f"summary_count_embed_dim must be positive, got {self.dct_memory_count_embed_dim}.")
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=summary_count.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = value * frequency.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.dct_memory_count_embed_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dct_memory_count_embed_dim - embedding.shape[-1]))
        mlp_dtype = next(self.dct_summary_count_mlp.parameters()).dtype
        return self.dct_summary_count_mlp(embedding.to(dtype=mlp_dtype)).to(dtype=dtype)

    def _augment_memory_dct(
        self,
        summary: torch.Tensor,
        recent: torch.Tensor,
        summary_valid: torch.Tensor,
        recent_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or not self.dct_memory_noise_enabled:
            return summary, recent

        def _noise_scale(
            tensor: torch.Tensor,
            fixed_std: float,
            std_range: Optional[tuple[float, float]],
        ) -> Optional[torch.Tensor]:
            if std_range is not None:
                low, high = std_range
                shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
                return tensor.new_empty(shape).uniform_(low, high)
            if fixed_std > 0:
                return tensor.new_tensor(fixed_std)
            return None

        summary_scale = _noise_scale(
            summary,
            self.dct_memory_summary_noise_std,
            self.dct_memory_summary_noise_std_range,
        )
        if summary_scale is not None:
            noise = torch.randn_like(summary) * summary_scale
            summary = summary + noise * summary_valid[:, None, None].to(dtype=summary.dtype)

        recent_scale = _noise_scale(
            recent,
            self.dct_memory_recent_noise_std,
            self.dct_memory_recent_noise_std_range,
        )
        if recent_scale is not None:
            noise = torch.randn_like(recent) * recent_scale
            recent = recent + noise * recent_valid[:, None, None].to(dtype=recent.dtype)
        return summary, recent

    def _dct_memory_tokens(
        self,
        summary: torch.Tensor,
        recent: torch.Tensor,
        summary_valid: torch.Tensor,
        recent_valid: torch.Tensor,
        summary_count: torch.Tensor,
    ) -> torch.Tensor:
        summary = summary[..., : self.dct_memory_action_dim]
        recent = recent[..., : self.dct_memory_action_dim]
        summary, recent = self._augment_memory_dct(summary, recent, summary_valid, recent_valid)

        summary_tokens = self.dct_summary_proj(
            summary.to(dtype=self.dct_summary_proj.weight.dtype)
        ).to(dtype=summary.dtype)
        recent_tokens = self.dct_recent_proj(
            recent.to(dtype=self.dct_recent_proj.weight.dtype)
        ).to(dtype=recent.dtype)
        summary_count_embed = self._summary_count_embedding(
            summary_count,
            dtype=summary_tokens.dtype,
        )

        summary_mask = summary_valid[:, None, None].to(dtype=summary_tokens.dtype)
        recent_mask = recent_valid[:, None, None].to(dtype=recent_tokens.dtype)
        summary_tokens = (
            summary_tokens
            + self.dct_memory_type_embedding[0].view(1, 1, -1)
            + summary_count_embed[:, None, :]
        ) * summary_mask
        recent_tokens = recent_tokens + self.dct_memory_type_embedding[1].view(1, 1, -1)
        if self.dct_memory_raw_recent:
            recent_tokens = recent_tokens + self.dct_recent_position_embedding.to(
                device=recent_tokens.device,
                dtype=recent_tokens.dtype,
            ).view(1, self.dct_memory_chunk_len, -1)
        recent_tokens = recent_tokens * recent_mask
        return torch.cat([summary_tokens, recent_tokens], dim=1)

    def _build_action_condition_with_memory(
        self,
        action_token_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        frame_images: List,
        dino_image_tensors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dino_hidden = self._encode_dino_hidden_states(
            batch_images=frame_images,
            dtype=action_token_hidden.dtype,
            image_tensors=dino_image_tensors,
        ).to(device=action_token_hidden.device)
        return torch.cat([action_token_hidden, memory_tokens, dino_hidden], dim=1)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ):
        if examples and "image_sequence" not in examples[0]:
            return super().forward(examples=examples, **kwargs)

        image_sequences = [example["image_sequence"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        actions = torch.tensor(
            np.array(actions),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
        num_refreshes = self._valid_training_refreshes(image_sequences, actions)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk DCTMemory forward received no valid action chunks. "
                f"actions.shape={tuple(actions.shape)}, fast_chunk_size={self.fast_chunk_size}, "
                f"vision_refresh_steps={self.vision_refresh_steps}, motion_dct_chunk_len={self.motion_dct_chunk_len}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        qwen_first_frame_images = self._to_qwen_batch_images(first_frame_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=qwen_first_frame_images,
            instructions=instructions,
        )
        action_token_hidden = self._encode_action_token_hidden(
            qwen_first_frame_images,
            instructions,
            qwen_inputs=qwen_inputs,
        )

        summary, recent, summary_valid, recent_valid, summary_count = self._examples_dct_memory(
            examples,
            device=action_token_hidden.device,
            dtype=action_token_hidden.dtype,
        )
        memory_tokens = self._dct_memory_tokens(
            summary,
            recent,
            summary_valid,
            recent_valid,
            summary_count,
        )

        long_chunk_len = min(self.motion_dct_chunk_len, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(action_token_hidden, actions, long_chunk_len)
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss
        action_loss_weight = self._action_loss_weight(kwargs.get("train_step", None))

        coarse_long_action = self._predict_coarse_action(action_token_hidden, long_chunk_len)
        if self.detach_idct_condition:
            coarse_long_action = coarse_long_action.detach()
        coarse_long_action = self._coarse_with_gripper_pad(coarse_long_action, action_dim=actions.shape[-1])

        flat_frame_images = []
        residual_action_targets = []
        coarse_action_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunk = coarse_long_action[:, start:end, :]
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            residual_action_targets.append(actions[:, start:end, :] - coarse_chunk)
            coarse_action_chunks.append(coarse_chunk)

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        flat_memory_tokens = memory_tokens.repeat(num_refreshes, 1, 1)
        dino_image_tensors = self.dino_encoder.prepare_dino_input(flat_frame_images)
        fused_hidden = self._build_action_condition_with_memory(
            flat_action_token_hidden,
            flat_memory_tokens,
            flat_frame_images,
            dino_image_tensors=dino_image_tensors,
        )
        residual_action_targets = torch.cat(residual_action_targets, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )
        flat_coarse_actions = torch.cat(coarse_action_chunks, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_residual_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=flat_coarse_actions if self.coarse_condition_query else None,
            )
            action_loss = self.l1_loss(pred_residual_actions, residual_action_targets)

        total_loss = action_loss_weight * action_loss + weighted_motion_dct_loss
        output = {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "action_dit_loss_weight": action_loss_weight,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
            "dct_memory_summary_valid": summary_valid.float().mean(),
            "dct_memory_recent_valid": recent_valid.float().mean(),
        }
        if kwargs.get("log_actiontoken_grad_norm", False):
            output.update(
                {
                    "grad_norm/action_token/action_dit_loss": self._grad_norm_wrt_hidden(
                        action_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/motion_dct_loss": self._grad_norm_wrt_hidden(
                        motion_dct_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/weighted_motion_dct_loss": self._grad_norm_wrt_hidden(
                        weighted_motion_dct_loss, action_token_hidden
                    ),
                }
            )
        return output

    def _online_memory_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        summary_shape = (batch_size, self.dct_memory_summary_keep_freq, self.dct_memory_action_dim)
        if self.dct_memory_raw_recent:
            recent_shape = (batch_size, self.dct_memory_chunk_len, self.dct_memory_action_dim)
            summary = (
                self._online_summary_dct.to(device=device, dtype=dtype)
                if self._online_summary_dct is not None
                else torch.zeros(summary_shape, device=device, dtype=dtype)
            )
            recent = (
                self._online_recent_actions.to(device=device, dtype=dtype)
                if self._online_recent_actions is not None
                else torch.zeros(recent_shape, device=device, dtype=dtype)
            )
            summary_valid = torch.full(
                (batch_size,),
                self._online_completed_chunks > 0,
                device=device,
                dtype=torch.bool,
            )
            recent_valid = torch.ones((batch_size,), device=device, dtype=torch.bool)
            summary_count = torch.full(
                (batch_size,),
                self._online_completed_chunks,
                device=device,
                dtype=dtype,
            )
        else:
            recent_shape = (batch_size, self.dct_memory_chunk_keep_freq, self.dct_memory_action_dim)
            if self._online_recent_dct is None:
                summary = torch.zeros(summary_shape, device=device, dtype=dtype)
                recent = torch.zeros(recent_shape, device=device, dtype=dtype)
                summary_valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
                recent_valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
                summary_count = torch.zeros((batch_size,), device=device, dtype=dtype)
            else:
                summary = (
                    self._online_summary_dct.to(device=device, dtype=dtype)
                    if self._online_summary_dct is not None
                    else torch.zeros(summary_shape, device=device, dtype=dtype)
                )
                recent = self._online_recent_dct.to(device=device, dtype=dtype)
                summary_valid = torch.full(
                    (batch_size,),
                    self._online_completed_chunks >= 2,
                    device=device,
                    dtype=torch.bool,
                )
                recent_valid = torch.ones((batch_size,), device=device, dtype=torch.bool)
                summary_count = torch.full(
                    (batch_size,),
                    max(self._online_completed_chunks - 1, 0),
                    device=device,
                    dtype=dtype,
                )
        return self._dct_memory_tokens(
            summary,
            recent,
            summary_valid,
            recent_valid,
            summary_count,
        )

    def _update_online_memory(self, pred_actions: torch.Tensor) -> None:
        actions = pred_actions.detach().to(dtype=torch.float32)
        if actions.shape[-1] > self.dct_memory_action_dim:
            actions = actions[..., : self.dct_memory_action_dim]
        elif actions.shape[-1] < self.dct_memory_action_dim:
            pad = actions.new_zeros(*actions.shape[:-1], self.dct_memory_action_dim - actions.shape[-1])
            actions = torch.cat([actions, pad], dim=-1)
        if actions.shape[-1] >= 7:
            gripper = actions[..., 6]
            gripper_out_of_train_range = bool(torch.any((gripper < 0.0) | (gripper > 1.0)).item())
            if gripper_out_of_train_range or bool(getattr(self, "_debug_online_memory_actions", False)):
                preview = actions[0, : min(4, actions.shape[1])].detach().float().cpu().numpy()
                print(
                    "[TwoChunkDCTMemory] store_memory_actions "
                    f"shape={tuple(actions.shape)} first_rows={preview.tolist()} "
                    f"gripper_minmax=({float(gripper.min()):.4f}, {float(gripper.max()):.4f}) "
                    f"gripper_out_of_train_range={gripper_out_of_train_range}"
                )

        if self.dct_memory_raw_recent:
            if self._online_recent_actions is None:
                self._online_recent_actions = actions.new_zeros(
                    actions.shape[0],
                    self.dct_memory_chunk_len,
                    self.dct_memory_action_dim,
                )
            self._online_recent_actions = torch.cat(
                [self._online_recent_actions.to(actions.device), actions],
                dim=1,
            )[:, -self.dct_memory_chunk_len :, :]

            if self._online_summary_pending_actions is None:
                self._online_summary_pending_actions = actions
            else:
                self._online_summary_pending_actions = torch.cat(
                    [self._online_summary_pending_actions.to(actions.device), actions],
                    dim=1,
                )

            while self._online_summary_pending_actions.shape[1] >= self.dct_memory_chunk_len:
                chunk = self._online_summary_pending_actions[:, : self.dct_memory_chunk_len, :]
                self._online_summary_pending_actions = self._online_summary_pending_actions[:, self.dct_memory_chunk_len :, :]
                chunk_dct = self._torch_dct(chunk, self.dct_memory_chunk_keep_freq)
                if self._online_completed_chunks == 0:
                    summary = chunk_dct.new_zeros(
                        chunk_dct.shape[0],
                        self.dct_memory_summary_keep_freq,
                        self.dct_memory_action_dim,
                    )
                    copy_len = min(self.dct_memory_chunk_keep_freq, self.dct_memory_summary_keep_freq)
                    summary[:, :copy_len, :] = chunk_dct[:, :copy_len, :]
                    self._online_summary_dct = summary
                else:
                    self._online_summary_dct = self._merge_summary_with_recent(
                        self._online_summary_dct.to(chunk_dct.device),
                        self._online_completed_chunks,
                        chunk_dct,
                    )
                self._online_completed_chunks += 1
            return

        if self._online_pending_actions is None:
            self._online_pending_actions = actions
        else:
            self._online_pending_actions = torch.cat([self._online_pending_actions.to(actions.device), actions], dim=1)

        while self._online_pending_actions.shape[1] >= self.dct_memory_chunk_len:
            chunk = self._online_pending_actions[:, : self.dct_memory_chunk_len, :]
            self._online_pending_actions = self._online_pending_actions[:, self.dct_memory_chunk_len :, :]
            new_recent = self._torch_dct(chunk, self.dct_memory_chunk_keep_freq)

            if self._online_completed_chunks == 0:
                self._online_recent_dct = new_recent
            elif self._online_completed_chunks == 1:
                summary = new_recent.new_zeros(
                    new_recent.shape[0],
                    self.dct_memory_summary_keep_freq,
                    self.dct_memory_action_dim,
                )
                copy_len = min(self.dct_memory_chunk_keep_freq, self.dct_memory_summary_keep_freq)
                summary[:, :copy_len, :] = self._online_recent_dct[:, :copy_len, :].to(new_recent.device)
                self._online_summary_dct = summary
                self._online_recent_dct = new_recent
            else:
                self._online_summary_dct = self._merge_summary_with_recent(
                    self._online_summary_dct.to(new_recent.device),
                    self._online_completed_chunks - 1,
                    self._online_recent_dct.to(new_recent.device),
                )
                self._online_recent_dct = new_recent
            self._online_completed_chunks += 1

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        if type(examples) is not list:
            examples = [examples]

        if examples and "image_sequence" in examples[0]:
            return self._predict_action_window(examples, **kwargs)

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        batch_size = len(examples)
        debug_twochunk = bool(kwargs.get("debug_twochunk", False))
        self._debug_online_memory_actions = debug_twochunk
        reset_cache = bool(kwargs.get("reset_cache", False))
        if reset_cache:
            self._reset_predict_cache()

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

        dino_image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        self._sync_cuda_if_needed()
        fast_start = time.perf_counter()
        memory_tokens = self._online_memory_tokens(
            batch_size,
            self._cached_action_token_hidden.device,
            self._cached_action_token_hidden.dtype,
        )
        fused_hidden = self._build_action_condition_with_memory(
            self._cached_action_token_hidden,
            memory_tokens,
            batch_images,
            dino_image_tensors=dino_image_tensors,
        )
        action_dim = int(self.config.framework.action_model.action_dim)
        coarse_action = self._coarse_with_gripper_pad(self._cached_coarse_action, action_dim)
        coarse_chunk = coarse_action[:, start:end, :]
        with torch.autocast("cuda", dtype=torch.float32):
            pred_residual_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=coarse_chunk if self.coarse_condition_query else None,
            )
        pred_actions = pred_residual_actions + coarse_chunk
        self._update_online_memory(pred_actions)
        self._sync_cuda_if_needed()
        fast_time_s = time.perf_counter() - fast_start

        self._predict_call_count += 1
        if debug_twochunk:
            hidden_mode = "new_slow_action_hidden" if slow_refresh else "reuse_slow_action_hidden"
            print(
                "[TwoChunkDCTMemoryFramework] "
                f"{hidden_mode}; reset_cache={reset_cache}; "
                f"fast_refresh_every={self.fast_chunk_size} env steps; "
                f"slow_refresh_every={self.language_refresh_steps} env steps; "
                f"predict_call_count={self._predict_call_count}; "
                f"online_completed_chunks={self._online_completed_chunks}; "
                f"slow_time={slow_time_s:.4f}s; fast_time={fast_time_s:.4f}s; "
                f"fused_hidden_shape={tuple(fused_hidden.shape)}; action_shape={tuple(pred_actions.shape)}"
            )
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
            "inference_timing": {
                "slow_refresh": slow_refresh,
                "slow_time_s": slow_time_s,
                "fast_time_s": fast_time_s,
                "model_inference_time_s": slow_time_s + fast_time_s,
                "timing_scope": "model_only_after_preprocess",
                "fast_chunk_size": self.fast_chunk_size,
                "language_refresh_steps": self.language_refresh_steps,
                "vision_refresh_steps": self.vision_refresh_steps,
                "online_dct_memory_chunks": self._online_completed_chunks,
            },
        }

    @torch.inference_mode()
    def _predict_action_window(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        image_sequences = [self._to_pil_nested(example["image_sequence"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            image_sequences = resize_images(image_sequences, target_size=train_obs_image_size)

        num_refreshes = self._valid_prediction_refreshes(image_sequences, examples)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk DCTMemory predict_action received no valid action windows. "
                f"fast_chunk_size={self.fast_chunk_size}, vision_refresh_steps={self.vision_refresh_steps}, "
                f"motion_dct_chunk_len={self.motion_dct_chunk_len}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)
        coarse_long_action = self._coarse_with_gripper_pad(
            self._predict_coarse_action(action_token_hidden, self.motion_dct_chunk_len),
            action_dim=int(self.config.framework.action_model.action_dim),
        )

        summary, recent, summary_valid, recent_valid, summary_count = self._examples_dct_memory(
            examples,
            device=action_token_hidden.device,
            dtype=action_token_hidden.dtype,
        )
        memory_tokens = self._dct_memory_tokens(
            summary,
            recent,
            summary_valid,
            recent_valid,
            summary_count,
        )

        flat_frame_images = []
        for refresh_i in range(num_refreshes):
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        flat_memory_tokens = memory_tokens.repeat(num_refreshes, 1, 1)
        coarse_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunks.append(coarse_long_action[:, start:end, :])
        coarse_chunks = torch.stack(coarse_chunks, dim=1)

        fused_hidden = self._build_action_condition_with_memory(
            flat_action_token_hidden,
            flat_memory_tokens,
            flat_frame_images,
        )
        batch_size = len(examples)
        flat_coarse_actions = coarse_chunks.permute(1, 0, 2, 3).reshape(
            num_refreshes * batch_size,
            self.fast_chunk_size,
            -1,
        ).to(device=fused_hidden.device, dtype=fused_hidden.dtype)
        pred_residual_actions = self.action_model.predict_action(
            fused_hidden,
            coarse_actions=flat_coarse_actions if self.coarse_condition_query else None,
        )

        pred_residual_actions = pred_residual_actions.view(num_refreshes, batch_size, self.fast_chunk_size, -1)
        pred_residual_actions = pred_residual_actions.permute(1, 0, 2, 3)

        pred_actions = pred_residual_actions + coarse_chunks
        pred_actions = pred_actions.reshape(batch_size, num_refreshes * self.fast_chunk_size, -1)
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
