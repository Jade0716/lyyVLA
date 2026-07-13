"""TwoChunk v2: action head predicts full actions without memory.

The DCT head still predicts a low-frequency coarse trajectory and exposes its
IDCT chunk as condition tokens. Unlike the original TwoChunk framework, the fast
action head is trained to output final actions directly instead of residuals,
and inference returns the action-head output directly. In short, coarse is a
condition hint only; there is no final ``coarse + residual`` skip.
"""

import time
from typing import List

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_V2")
class Qwen_GR00T_ActionToken_TwoChunk_V2(Qwen_GR00T_ActionToken_TwoChunk):
    """TwoChunk variant where action head predicts full actions, no memory."""

    @staticmethod
    def _state_for_refresh(state_tensor: torch.Tensor | None, refresh_i: int) -> torch.Tensor | None:
        if state_tensor is None:
            return None
        if state_tensor.ndim == 2:
            if refresh_i != 0:
                raise ValueError(
                    "TwoChunk V2 include_state requires refresh-aligned state sequence [B,T,D] "
                    f"for multi-refresh training, got [B,D] and refresh_i={refresh_i}."
                )
            return state_tensor
        if state_tensor.ndim == 3:
            if refresh_i >= state_tensor.shape[1]:
                raise ValueError(
                    "TwoChunk V2 include_state requires state sequence length >= num_refreshes; "
                    f"got state length {state_tensor.shape[1]} and refresh_i={refresh_i}."
                )
            return state_tensor[:, refresh_i, :]
        raise ValueError(f"Expected state tensor shape [B,D] or [B,T,D], got {tuple(state_tensor.shape)}.")

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ):
        if examples and "image_sequence" not in examples[0]:
            return super().forward(examples=examples, **kwargs)

        image_sequences = [self._to_pil_nested(example["image_sequence"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example.get("state") for example in examples] if self.include_state_condition else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            image_sequences = resize_images(image_sequences, target_size=train_obs_image_size)

        actions = torch.tensor(
            np.array(actions),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
        state_tensor = None
        if self.include_state_condition:
            if states is None or any(state is None for state in states):
                raise ValueError("include_state=true requires every training example to contain `state`.")
            state_tensor = torch.tensor(
                np.array(states),
                device=self.action_query_token.device,
                dtype=self.action_query_token.dtype,
            )
        num_refreshes = self._valid_training_refreshes(image_sequences, actions)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk V2 forward received no valid action chunks. "
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

        long_chunk_len = min(self.motion_dct_chunk_len, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(action_token_hidden, actions, long_chunk_len)
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss
        action_loss_weight = self._action_loss_weight(kwargs.get("train_step", None))

        coarse_long_action = self._predict_coarse_action(action_token_hidden, long_chunk_len)
        if self.detach_idct_condition:
            coarse_long_action = coarse_long_action.detach()
        coarse_long_action = self._coarse_with_gripper_pad(coarse_long_action, action_dim=actions.shape[-1])

        flat_frame_images = []
        action_targets = []
        coarse_action_chunks = []
        state_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunk = coarse_long_action[:, start:end, :]
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            action_targets.append(actions[:, start:end, :])
            coarse_action_chunks.append(coarse_chunk)
            if state_tensor is not None:
                state_chunks.append(self._state_for_refresh(state_tensor, refresh_i))

        flat_action_token_hidden = self._repeat_slow_token_hidden(action_token_hidden, num_refreshes)
        flat_states = torch.cat(state_chunks, dim=0) if state_chunks else None
        flat_coarse_actions = torch.cat(coarse_action_chunks, dim=0).to(
            device=flat_action_token_hidden.device,
            dtype=flat_action_token_hidden.dtype,
        )
        dino_image_tensors = self.dino_encoder.prepare_dino_input(flat_frame_images)
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            dino_image_tensors=dino_image_tensors,
            coarse_actions=flat_coarse_actions,
            state=flat_states,
        )
        action_targets = torch.cat(action_targets, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )
        flat_coarse_actions = flat_coarse_actions.to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=flat_coarse_actions if self.coarse_actions_for_action_head else None,
                condition_groups=getattr(self, "_last_action_condition_groups", None),
                condition_layers=getattr(self, "_last_action_condition_layers", None),
                condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
                attention_debug_spans=None,
            )
            self._clear_temporary_action_conditions()
            action_loss = self.l1_loss(pred_actions, action_targets)

        total_loss = action_loss_weight * action_loss + weighted_motion_dct_loss
        output = {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "action_dit_loss_weight": action_loss_weight,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
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
        state = None
        if self.include_state_condition:
            states = [example.get("state") for example in examples]
            if any(item is None for item in states):
                raise ValueError("include_state=true requires every inference example to contain `state`.")
            state = torch.tensor(
                np.array(states),
                device=self.action_query_token.device,
                dtype=self.action_query_token.dtype,
            )
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0, :]
        debug_twochunk = bool(kwargs.get("debug_twochunk", False))
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
        action_dim = int(self.config.framework.action_model.action_dim)
        coarse_action = self._coarse_with_gripper_pad(self._cached_coarse_action, action_dim)
        coarse_chunk = coarse_action[:, start:end, :]
        fused_hidden = self._build_action_condition(
            self._cached_action_token_hidden,
            batch_images,
            dino_image_tensors=dino_image_tensors,
            coarse_actions=coarse_chunk,
            state=state,
        )
        attention_debug = bool(kwargs.get("attention_debug", False))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=coarse_chunk if self.coarse_actions_for_action_head else None,
                condition_groups=getattr(self, "_last_action_condition_groups", None),
                condition_layers=getattr(self, "_last_action_condition_layers", None),
                condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
                attention_debug_spans=self._last_attention_debug_spans if attention_debug else None,
            )
        self._sync_cuda_if_needed()
        fast_time_s = time.perf_counter() - fast_start

        self._predict_call_count += 1
        if debug_twochunk:
            hidden_mode = "new_slow_action_hidden" if slow_refresh else "reuse_slow_action_hidden"
            print(
                "[TwoChunkV2Framework] "
                f"{hidden_mode}; reset_cache={reset_cache}; "
                f"fast_refresh_every={self.fast_chunk_size} env steps; "
                f"slow_refresh_every={self.language_refresh_steps} env steps; "
                f"predict_call_count={self._predict_call_count}; "
                f"slow_time={slow_time_s:.4f}s; fast_time={fast_time_s:.4f}s; "
                f"fused_hidden_shape={tuple(fused_hidden.shape)}; action_shape={tuple(pred_actions.shape)}"
            )
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        result = {
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
            },
        }
        if attention_debug:
            result["attention_debug"] = {
                "layers": getattr(self.action_model, "last_attention_debug", None),
                "spans": getattr(self, "_last_attention_debug_spans", None),
            }
        return result

    @torch.inference_mode()
    def _predict_action_window(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        image_sequences = [self._to_pil_nested(example["image_sequence"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state_sequence = None
        if self.include_state_condition:
            states = [example.get("state") for example in examples]
            if any(item is None for item in states):
                raise ValueError("include_state=true requires every prediction-window example to contain `state`.")
            state_sequence = torch.tensor(
                np.array(states),
                device=self.action_query_token.device,
                dtype=self.action_query_token.dtype,
            )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            image_sequences = resize_images(image_sequences, target_size=train_obs_image_size)

        num_refreshes = self._valid_prediction_refreshes(image_sequences, examples)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk V2 predict_action received no valid action windows. "
                f"fast_chunk_size={self.fast_chunk_size}, vision_refresh_steps={self.vision_refresh_steps}, "
                f"motion_dct_chunk_len={self.motion_dct_chunk_len}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)
        coarse_long_action = self._coarse_with_gripper_pad(
            self._predict_coarse_action(action_token_hidden, self.motion_dct_chunk_len),
            action_dim=int(self.config.framework.action_model.action_dim),
        )

        flat_frame_images = []
        state_chunks = []
        for refresh_i in range(num_refreshes):
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            if state_sequence is not None:
                state_chunks.append(self._state_for_refresh(state_sequence, refresh_i))

        flat_action_token_hidden = self._repeat_slow_token_hidden(action_token_hidden, num_refreshes)
        flat_states = torch.cat(state_chunks, dim=0) if state_chunks else None
        batch_size = len(examples)
        coarse_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunks.append(coarse_long_action[:, start:end, :])
        coarse_chunks = torch.stack(coarse_chunks, dim=1)
        flat_coarse_actions = coarse_chunks.permute(1, 0, 2, 3).reshape(
            num_refreshes * batch_size,
            self.fast_chunk_size,
            -1,
        ).to(device=flat_action_token_hidden.device, dtype=flat_action_token_hidden.dtype)
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            coarse_actions=flat_coarse_actions,
            state=flat_states,
        )
        flat_coarse_actions = flat_coarse_actions.to(device=fused_hidden.device, dtype=fused_hidden.dtype)
        pred_actions = self.action_model.predict_action(
            fused_hidden,
            coarse_actions=flat_coarse_actions if self.coarse_actions_for_action_head else None,
            condition_groups=getattr(self, "_last_action_condition_groups", None),
            condition_layers=getattr(self, "_last_action_condition_layers", None),
            condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
            attention_debug_spans=None,
        )

        pred_actions = pred_actions.view(num_refreshes, batch_size, self.fast_chunk_size, -1)
        pred_actions = pred_actions.permute(1, 0, 2, 3)
        pred_actions = pred_actions.reshape(batch_size, num_refreshes * self.fast_chunk_size, -1)
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
