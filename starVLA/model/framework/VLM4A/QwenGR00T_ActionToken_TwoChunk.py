from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class MotionDCTHead(nn.Module):
    def __init__(self, hidden_size=1024, keep_freq=8, action_dim=7):
        super().__init__()
        self.keep_freq = keep_freq
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, keep_freq * action_dim),
        )

    def forward(self, h_motion):
        if h_motion.dim() == 3:
            h_motion = h_motion[:, 0]
        out = self.net(h_motion)
        return out.view(-1, self.keep_freq, self.action_dim)


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk")
class Qwen_GR00T_ActionToken_TwoChunk(Qwen_GR00T_ActionToken):
    """
    Two-chunk GR00T ActionToken variant.

    A slow Qwen pass appends the learnable action query and returns only that
    query hidden state. Fast action refreshes bypass the Qwen language model:
    they extract pre-LLM image embeddings from Qwen inputs, concatenate those
    image tokens with the cached action-query hidden state, and condition the
    GR00T DiT on that sequence.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        self.language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 32))
        self.vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", self.action_horizon))
        self.twochunk_window_size = int(qwenvl_cfg.get("twochunk_window_size", self.language_refresh_steps))
        self.motion_dct_keep_freq = int(qwenvl_cfg.get("motion_dct_keep_freq", 8))
        self.motion_dct_loss_weight = float(qwenvl_cfg.get("motion_dct_loss_weight", 1.0))

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)
        self.motion_dct_head = MotionDCTHead(
            hidden_size=hidden_size,
            keep_freq=self.motion_dct_keep_freq,
            action_dim=action_dim,
        )

        if self.action_horizon != self.vision_refresh_steps:
            print(
                "[QwenGR00T_ActionToken_TwoChunk] action_horizon controls the predicted action chunk size. "
                f"Got action_horizon={self.action_horizon}, vision_refresh_steps={self.vision_refresh_steps}. "
                "Set them equal for a clean fast-refresh cadence."
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

    def _motion_dct_target(self, actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
        action_chunk = actions[:, :chunk_len, :].detach().float().cpu().numpy()
        low_dct_gt = dct(action_chunk, type=2, axis=1, norm="ortho")[:, : self.motion_dct_keep_freq, :]
        return torch.from_numpy(low_dct_gt).to(device=actions.device, dtype=actions.dtype)

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
            return qwenvl_outputs.hidden_states[-1][:, -1:, :]

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

    def _valid_training_refreshes(self, image_sequences: List[List], actions: torch.Tensor) -> int:
        max_refreshes = max(1, self.twochunk_window_size // max(self.vision_refresh_steps, 1))
        num_refreshes = min(min(len(seq) for seq in image_sequences), max_refreshes)
        while num_refreshes > 0:
            start = (num_refreshes - 1) * self.vision_refresh_steps
            end = start + self.action_horizon
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
                f"actions.shape={tuple(actions.shape)}, action_horizon={self.action_horizon}, "
                f"vision_refresh_steps={self.vision_refresh_steps}"
            )

        big_chunk_images = self._flatten_big_chunk_images(image_sequences, num_refreshes)
        action_token_hidden = self._encode_action_token_hidden(big_chunk_images, instructions)

        # Auxiliary low-frequency supervision: the slow action token predicts
        # the first DCT coefficients of the whole 32-step motion chunk.
        motion_chunk_len = min(num_refreshes * self.vision_refresh_steps, actions.shape[1])
        low_dct_gt = self._motion_dct_target(actions, motion_chunk_len).to(
            device=action_token_hidden.device,
            dtype=action_token_hidden.dtype,
        )
        low_dct_pred = self.motion_dct_head(action_token_hidden)
        motion_dct_loss = F.mse_loss(low_dct_pred.float(), low_dct_gt.float())

        flat_frame_images = []
        action_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.action_horizon
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            action_chunks.append(actions[:, start:end, :])

        # Fast DiT condition: repeat the slow action-token hidden for every
        # 4-step refresh, then concatenate it with that refresh's image embeds.
        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        encoder_hidden, encoder_attention_mask = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
        )
        actions_target = torch.cat(action_chunks, dim=0).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)

        repeated_diffusion_steps = (
            self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            if self.config and hasattr(self.config, "framework")
            else 4
        )

        with torch.autocast("cuda", dtype=torch.float32):
            action_loss = self.action_model(
                encoder_hidden.repeat(repeated_diffusion_steps, 1, 1),
                actions_target.repeat(repeated_diffusion_steps, 1, 1),
                None,
                encoder_attention_mask=encoder_attention_mask.repeat(repeated_diffusion_steps, 1),
            )

        total_loss = action_loss + self.motion_dct_loss_weight * motion_dct_loss
        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss.detach(),
            "motion_dct_loss": motion_dct_loss.detach(),
        }

    def _should_refresh_action_token(self, instructions: List[str], batch_size: int) -> bool:
        if self._cached_action_token_hidden is None:
            return True
        if self._cached_batch_size != batch_size:
            return True
        if self._cached_instruction_key != tuple(instructions):
            return True

        steps_since_refresh = self._predict_call_count * max(int(self.action_horizon), 1)
        return steps_since_refresh % self.language_refresh_steps == 0

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        batch_size = len(examples)
        if self._should_refresh_action_token(instructions, batch_size):
            self._cached_action_token_hidden = self._encode_action_token_hidden(batch_images, instructions).detach()
            self._cached_instruction_key = tuple(instructions)
            self._cached_batch_size = batch_size
            self._predict_call_count = 0

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

        self._predict_call_count += 1
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
