import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import (
    Qwen_GR00T_ActionToken,
    _as_bool,
)
from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk")
class Qwen_GR00T_ActionToken_TwoChunk(Qwen_GR00T_ActionToken):
    """
    Two-chunk GR00T ActionToken variant.

    A slow Qwen pass appends learnable motion queries and returns only those
    query hidden states. Fast action refreshes bypass the Qwen language model:
    they extract pre-LLM image embeddings from Qwen inputs, concatenate those
    image tokens with the cached action-query hidden states, and condition the
    GR00T DiT on that sequence.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        self.language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 16))
        self.vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", self.action_horizon))
        self.fast_chunk_size = self.vision_refresh_steps
        self.twochunk_window_size = int(qwenvl_cfg.get("twochunk_window_size", self.language_refresh_steps))
        self.motion_dct_keep_freq = int(qwenvl_cfg.get("motion_dct_keep_freq", 4))
        self.motion_dct_chunk_len = int(qwenvl_cfg.get("motion_dct_chunk_len", self.language_refresh_steps))
        self.use_motion_dct_loss = _as_bool(qwenvl_cfg.get("use_motion_dct_loss", True))
        self.motion_dct_loss_weight = float(qwenvl_cfg.get("motion_dct_loss_weight", 1.0))
        self.action_model.action_horizon = self.fast_chunk_size
        if hasattr(self.action_model, "config"):
            self.action_model.config.action_horizon = self.fast_chunk_size

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)
        self.motion_dct_action_dim = max(action_dim - 1, 1)
        self.action_query_token = nn.Parameter(torch.randn(1, self.motion_dct_keep_freq, hidden_size) * 0.02)
        self.motion_dct_head = None
        if self.use_motion_dct_loss:
            self.motion_dct_head = L1RegressionActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                action_dim=self.motion_dct_action_dim,
                NUM_ACTIONS_CHUNK=self.motion_dct_keep_freq,
            )

        self._cached_action_token_hidden = None
        self._cached_instruction_key = None
        self._cached_batch_size = None
        self._predict_call_count = 0

    def _token_id(self, attr: str, fallback: int | None = None) -> int | None:
        return getattr(self.qwen_vl_interface.model.config, attr, fallback)

    @staticmethod
    def _cat_feature_output(features) -> torch.Tensor:
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        if isinstance(features, (list, tuple)):
            return torch.cat(list(features), dim=0)
        return features

    def _call_image_features(self, qwen_inputs: dict):
        model = self.qwen_vl_interface.model
        kwargs = {
            "pixel_values": qwen_inputs["pixel_values"],
            "image_grid_thw": qwen_inputs.get("image_grid_thw", None),
        }
        try:
            return model.get_image_features(**kwargs, return_dict=True)
        except TypeError:
            return model.get_image_features(**kwargs)

    def _call_video_features(self, qwen_inputs: dict):
        model = self.qwen_vl_interface.model
        kwargs = {
            "pixel_values_videos": qwen_inputs["pixel_values_videos"],
            "video_grid_thw": qwen_inputs.get("video_grid_thw", None),
        }
        try:
            return model.get_video_features(**kwargs, return_dict=True)
        except TypeError:
            return model.get_video_features(**kwargs)

    @staticmethod
    def _pack_masked_tokens(inputs_embeds: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pieces = [inputs_embeds[i][token_mask[i]] for i in range(inputs_embeds.shape[0])]
        max_len = max(max(piece.shape[0], 1) for piece in pieces)
        out = inputs_embeds.new_zeros((inputs_embeds.shape[0], max_len, inputs_embeds.shape[-1]))
        out_mask = torch.zeros((inputs_embeds.shape[0], max_len), dtype=torch.bool, device=inputs_embeds.device)
        for i, piece in enumerate(pieces):
            if piece.numel() == 0:
                continue
            out[i, : piece.shape[0]] = piece
            out_mask[i, : piece.shape[0]] = True
        return out, out_mask

    def _extract_image_embeds_from_inputs(self, qwen_inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        image_token_id = self._token_id("image_token_id")
        if qwen_inputs.get("pixel_values", None) is not None and image_token_id is not None:
            image_features = self._cat_feature_output(self._call_image_features(qwen_inputs))
            image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            current_mask = input_ids == int(image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(
                current_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_features,
            )
            image_token_mask |= current_mask

        video_token_id = self._token_id("video_token_id")
        if qwen_inputs.get("pixel_values_videos", None) is not None and video_token_id is not None:
            video_features = self._cat_feature_output(self._call_video_features(qwen_inputs))
            video_features = video_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            current_mask = input_ids == int(video_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(
                current_mask.unsqueeze(-1).expand_as(inputs_embeds),
                video_features,
            )
            image_token_mask |= current_mask

        return self._pack_masked_tokens(inputs_embeds, image_token_mask)

    def _encode_action_token_hidden(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = self._append_action_query(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return qwenvl_outputs.hidden_states[-1][:, -self.motion_dct_keep_freq :, :]

    def _build_action_condition(
        self,
        action_token_hidden: torch.Tensor,
        frame_images: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        instructions = [""] * len(frame_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=frame_images, instructions=instructions)
        image_embeds, image_mask = self._extract_image_embeds_from_inputs(qwen_inputs)

        action_token_hidden = action_token_hidden.to(device=image_embeds.device, dtype=image_embeds.dtype)
        attention_mask = torch.cat(
            [
                torch.ones(
                    action_token_hidden.shape[0],
                    action_token_hidden.shape[1],
                    dtype=torch.bool,
                    device=action_token_hidden.device,
                ),
                image_mask,
            ],
            dim=1,
        )
        return torch.cat([action_token_hidden, image_embeds], dim=1), attention_mask

    def _flatten_big_chunk_images(self, image_sequences: List[List], num_refreshes: int) -> List[List]:
        big_chunk_images = []
        for image_sequence in image_sequences:
            images = []
            for refresh_images in image_sequence[:num_refreshes]:
                images.extend(refresh_images)
            big_chunk_images.append(images)
        return big_chunk_images

    @staticmethod
    def _first_refresh_images(image_sequences: List[List]) -> List[List]:
        return [image_sequence[0] for image_sequence in image_sequences]

    @staticmethod
    def _to_pil_nested(images):
        if isinstance(images, list):
            return [Qwen_GR00T_ActionToken_TwoChunk._to_pil_nested(image) for image in images]
        return to_pil_preserve(images)

    def _valid_training_refreshes(self, image_sequences: List[List], actions: torch.Tensor) -> int:
        max_refreshes = max(1, self.twochunk_window_size // max(self.vision_refresh_steps, 1))
        num_refreshes = min(min(len(seq) for seq in image_sequences), max_refreshes)
        while num_refreshes > 0:
            start = (num_refreshes - 1) * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            if end <= actions.shape[1]:
                return num_refreshes
            num_refreshes -= 1
        return 0

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
                "TwoChunk forward received no valid action chunks. "
                f"actions.shape={tuple(actions.shape)}, fast_chunk_size={self.fast_chunk_size}, "
                f"vision_refresh_steps={self.vision_refresh_steps}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)

        # Auxiliary low-frequency supervision: each slow motion token predicts
        # one DCT frequency for the non-gripper action dimensions.
        motion_chunk_len = min(self.motion_dct_chunk_len, num_refreshes * self.vision_refresh_steps, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(action_token_hidden, actions, motion_chunk_len)
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss

        flat_frame_images = []
        action_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            action_chunks.append(actions[:, start:end, :])

        # Fast DiT condition: repeat the slow motion-token hidden states for every
        # 4-step refresh, then concatenate it with that refresh's image embeds.
        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        encoder_hidden, encoder_attention_mask = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
        )
        actions_target = torch.cat(action_chunks, dim=0).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)

        if self.config and hasattr(self.config, "trainer"):
            repeated_diffusion_steps = int(self.config.trainer.get("repeated_diffusion_steps", 16))

        with torch.autocast("cuda", dtype=torch.float32):
            action_loss = self.action_model(
                encoder_hidden.repeat(repeated_diffusion_steps, 1, 1),
                actions_target.repeat(repeated_diffusion_steps, 1, 1),
                None,
                encoder_attention_mask=encoder_attention_mask.repeat(repeated_diffusion_steps, 1),
            )

        total_loss = action_loss + weighted_motion_dct_loss
        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
        }

    def _should_refresh_action_token(self, instructions: List[str], batch_size: int) -> bool:
        if self._cached_action_token_hidden is None:
            return True
        if self._cached_batch_size != batch_size:
            return True
        if self._cached_instruction_key != tuple(instructions):
            return True

        steps_since_refresh = self._predict_call_count * max(int(self.fast_chunk_size), 1)
        return steps_since_refresh % self.language_refresh_steps == 0

    def _reset_predict_cache(self) -> None:
        self._cached_action_token_hidden = None
        self._cached_instruction_key = None
        self._cached_batch_size = None
        self._predict_call_count = 0

    @staticmethod
    def _sync_cuda_if_needed() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
        reset_cache = bool(kwargs.get("reset_cache", False))
        if reset_cache:
            self._reset_predict_cache()

        slow_refresh = self._should_refresh_action_token(instructions, batch_size)
        slow_time_s = 0.0
        if slow_refresh:
            self._sync_cuda_if_needed()
            slow_start = time.perf_counter()
            self._cached_action_token_hidden = self._encode_action_token_hidden(batch_images, instructions).detach()
            self._sync_cuda_if_needed()
            slow_time_s = time.perf_counter() - slow_start
            self._cached_instruction_key = tuple(instructions)
            self._cached_batch_size = batch_size
            self._predict_call_count = 0

        self._sync_cuda_if_needed()
        fast_start = time.perf_counter()
        encoder_hidden, encoder_attention_mask = self._build_action_condition(
            self._cached_action_token_hidden,
            batch_images,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                encoder_hidden,
                None,
                encoder_attention_mask=encoder_attention_mask.to(dtype=torch.bool),
            )
        self._sync_cuda_if_needed()
        fast_time_s = time.perf_counter() - fast_start

        self._predict_call_count += 1
        if debug_twochunk:
            hidden_mode = "new_slow_action_hidden" if slow_refresh else "reuse_slow_action_hidden"
            print(
                "[TwoChunkFramework] "
                f"{hidden_mode}; reset_cache={reset_cache}; "
                f"fast_refresh_every={self.fast_chunk_size} env steps; "
                f"slow_refresh_every={self.language_refresh_steps} env steps; "
                f"predict_call_count={self._predict_call_count}; "
                f"slow_time={slow_time_s:.4f}s; "
                f"fast_time={fast_time_s:.4f}s; "
                f"encoder_hidden_shape={tuple(encoder_hidden.shape)}; "
                f"action_shape={tuple(pred_actions.shape)}"
            )
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _valid_prediction_refreshes(self, image_sequences: List[List], examples: List[dict]) -> int:
        max_refreshes = max(1, self.twochunk_window_size // max(self.vision_refresh_steps, 1))
        num_refreshes = min(min(len(seq) for seq in image_sequences), max_refreshes)
        if examples and "action" in examples[0]:
            action_len = min(np.asarray(example["action"]).shape[0] for example in examples)
            while num_refreshes > 0:
                end = (num_refreshes - 1) * self.vision_refresh_steps + self.fast_chunk_size
                if end <= action_len:
                    return num_refreshes
                num_refreshes -= 1
            return 0
        return num_refreshes

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
                "TwoChunk predict_action received no valid action windows. "
                f"fast_chunk_size={self.fast_chunk_size}, vision_refresh_steps={self.vision_refresh_steps}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)

        flat_frame_images = []
        for refresh_i in range(num_refreshes):
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        encoder_hidden, encoder_attention_mask = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                encoder_hidden,
                None,
                encoder_attention_mask=encoder_attention_mask.to(dtype=torch.bool),
            )

        batch_size = len(examples)
        pred_actions = pred_actions.view(num_refreshes, batch_size, self.fast_chunk_size, -1)
        pred_actions = pred_actions.permute(1, 0, 2, 3).reshape(batch_size, num_refreshes * self.fast_chunk_size, -1)
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
