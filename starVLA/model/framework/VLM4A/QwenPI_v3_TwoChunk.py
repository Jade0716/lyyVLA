from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v3 import QwenPI_v3DefaultConfig, Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenPI_v3TwoChunkDefaultConfig(QwenPI_v3DefaultConfig):
    name: str = "QwenPI_v3_TwoChunk"


@FRAMEWORK_REGISTRY.register("QwenPI_v3_TwoChunk")
class Qwen_PI_v3_TwoChunk(Qwen_PI_v3):
    """
    QwenPI_v3 variant with a slower language refresh and a faster vision refresh.

    Inference policy:
      - `language_refresh_steps`: run the full Qwen-VL path and cache language-token
        hidden states.
      - `vision_refresh_steps`: run only the VLM vision tower, fuse those visual
        tokens with the cached language hidden states, then call the action DiT.

    For the current websocket policy server, `action_horizon` is also the server
    action chunk size. Set `framework.action_model.action_horizon` to the desired
    fast refresh interval, e.g. 4.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        self.language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 32))
        self.vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", 4))
        if self.action_horizon != self.vision_refresh_steps:
            print(
                "[QwenPI_v3_TwoChunk] action_horizon controls the websocket action chunk size. "
                f"Got action_horizon={self.action_horizon}, vision_refresh_steps={self.vision_refresh_steps}. "
                "Set them equal if you want the policy server to call this framework at the vision refresh rate."
            )

        vl_hidden_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        self.lang_vision_fusion_mlp = nn.Sequential(
            nn.LayerNorm(vl_hidden_dim),
            nn.Linear(vl_hidden_dim, vl_hidden_dim),
            nn.GELU(),
            nn.Linear(vl_hidden_dim, vl_hidden_dim),
        )

        self._cached_language_layers = None
        self._cached_language_mask = None
        self._cached_instruction_key = None
        self._cached_batch_size = None
        self._predict_call_count = 0

    def _token_id(self, attr: str, fallback: int | None = None) -> int | None:
        return getattr(self.qwen_vl_interface.model.config, attr, fallback)

    def _language_mask_from_inputs(self, qwen_inputs) -> torch.Tensor:
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", torch.ones_like(input_ids)).bool()
        language_mask = attention_mask.clone()

        for token_attr in ("image_token_id", "video_token_id"):
            token_id = self._token_id(token_attr)
            if token_id is not None:
                language_mask &= input_ids != int(token_id)

        return language_mask

    @staticmethod
    def _pack_masked_tokens(hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pieces = [hidden[i][mask[i]] for i in range(hidden.shape[0])]
        max_len = max(piece.shape[0] for piece in pieces)
        out = hidden.new_zeros((hidden.shape[0], max_len, hidden.shape[-1]))
        out_mask = torch.zeros((hidden.shape[0], max_len), dtype=torch.bool, device=hidden.device)
        for i, piece in enumerate(pieces):
            out[i, : piece.shape[0]] = piece
            out_mask[i, : piece.shape[0]] = True
        return out, out_mask

    def _split_vision_tokens_by_sample(self, qwen_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", torch.ones_like(input_ids)).bool()
        device = input_ids.device

        image_token_id = self._token_id("image_token_id")
        video_token_id = self._token_id("video_token_id")

        per_sample_tokens = []

        if qwen_inputs.get("pixel_values", None) is not None and image_token_id is not None:
            image_features = model.get_image_features(
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            )
            image_features = torch.cat(list(image_features), dim=0).to(device)
            counts = (input_ids == int(image_token_id)).sum(dim=1).tolist()
            image_splits = list(torch.split(image_features, counts, dim=0))
        else:
            image_splits = [None] * input_ids.shape[0]

        if qwen_inputs.get("pixel_values_videos", None) is not None and video_token_id is not None:
            video_features = model.get_video_features(
                pixel_values_videos=qwen_inputs["pixel_values_videos"],
                video_grid_thw=qwen_inputs.get("video_grid_thw", None),
            )
            video_features = torch.cat(list(video_features), dim=0).to(device)
            counts = (input_ids == int(video_token_id)).sum(dim=1).tolist()
            video_splits = list(torch.split(video_features, counts, dim=0))
        else:
            video_splits = [None] * input_ids.shape[0]

        for image_tokens, video_tokens in zip(image_splits, video_splits):
            chunks = []
            if image_tokens is not None:
                chunks.append(image_tokens)
            if video_tokens is not None:
                chunks.append(video_tokens)
            if chunks:
                per_sample_tokens.append(torch.cat(chunks, dim=0))
            else:
                per_sample_tokens.append(torch.zeros((0, model.config.hidden_size), device=device))

        max_len = max(max(tokens.shape[0], 1) for tokens in per_sample_tokens)
        vision = torch.zeros(
            (input_ids.shape[0], max_len, model.config.hidden_size),
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        vision_mask = torch.zeros((input_ids.shape[0], max_len), dtype=torch.bool, device=device)
        for i, tokens in enumerate(per_sample_tokens):
            if tokens.numel() == 0:
                continue
            tokens = tokens.to(device=vision.device, dtype=vision.dtype)
            vision[i, : tokens.shape[0]] = tokens
            vision_mask[i, : tokens.shape[0]] = True

        return vision, vision_mask & attention_mask.new_ones(vision_mask.shape)

    def _encode_language_layers_full(self, batch_images: List, instructions: List[str]):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        language_mask = self._language_mask_from_inputs(qwen_inputs)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        language_layers = []
        packed_mask = None
        for hidden in outputs.hidden_states[-self.num_action_dit_layers:]:
            packed, packed_mask = self._pack_masked_tokens(hidden, language_mask)
            language_layers.append(packed)

        return language_layers, packed_mask, qwen_inputs

    def _fuse_language_and_vision(
        self,
        language_layers: List[torch.Tensor],
        language_mask: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor,
    ):
        fused_layers = []
        fused_mask = torch.cat([language_mask, vision_mask], dim=1)
        for layer_hidden, projector in zip(language_layers, self.project_layers):
            vision = vision_tokens.to(device=layer_hidden.device, dtype=layer_hidden.dtype)
            tokens = torch.cat([layer_hidden, vision], dim=1)
            tokens = self.lang_vision_fusion_mlp(tokens)
            fused_layers.append(projector(tokens))
        return fused_layers, fused_mask

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        language_layers, language_mask, qwen_inputs = self._encode_language_layers_full(batch_images, instructions)
        vision_tokens, vision_mask = self._split_vision_tokens_by_sample(qwen_inputs)
        return self._fuse_language_and_vision(language_layers, language_mask, vision_tokens, vision_mask)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        if examples and "image_sequence" not in examples[0]:
            return super().forward(examples=examples, **kwargs)

        image_sequences = [example["image_sequence"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        first_frame_images = [image_sequence[0] for image_sequence in image_sequences]
        language_layers, language_mask, _ = self._encode_language_layers_full(first_frame_images, instructions)
        base_hidden = language_layers[-1]

        actions = torch.tensor(
            np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
        )

        num_refreshes = min(len(seq) for seq in image_sequences)
        repeated_diffusion_steps = (
            self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 4
        )

        losses = []
        with torch.autocast("cuda", dtype=torch.float32):
            for refresh_i in range(num_refreshes):
                start = refresh_i * self.vision_refresh_steps
                end = start + self.action_horizon
                if end > actions.shape[1]:
                    break

                frame_images = [image_sequence[refresh_i] for image_sequence in image_sequences]
                qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                    images=frame_images, instructions=instructions
                )
                vision_tokens, vision_mask = self._split_vision_tokens_by_sample(qwen_inputs)
                vl_embs_list, attention_mask = self._fuse_language_and_vision(
                    language_layers, language_mask, vision_tokens, vision_mask
                )

                actions_target = actions[:, start:end, :]
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
                attention_mask_repeated = attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

                loss = self.action_model(
                    vl_embs_list_repeated,
                    actions_target_repeated,
                    None,
                    encoder_attention_mask=attention_mask_repeated,
                )
                losses.append(loss)

        if not losses:
            raise ValueError(
                "TwoChunk forward received no valid action chunks. "
                f"actions.shape={tuple(actions.shape)}, action_horizon={self.action_horizon}, "
                f"vision_refresh_steps={self.vision_refresh_steps}, num_refreshes={num_refreshes}"
            )

        return {"action_loss": torch.stack(losses).mean()}

    def _should_refresh_language(self, instructions: List[str], batch_size: int) -> bool:
        if self._cached_language_layers is None or self._cached_language_mask is None:
            return True
        if self._cached_batch_size != batch_size:
            return True
        instruction_key = tuple(instructions)
        if self._cached_instruction_key != instruction_key:
            return True

        steps_since_refresh = self._predict_call_count * max(int(self.action_horizon), 1)
        return steps_since_refresh % self.language_refresh_steps == 0

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        batch_size = len(examples)
        if self._should_refresh_language(instructions, batch_size):
            language_layers, language_mask, _ = self._encode_language_layers_full(batch_images, instructions)
            self._cached_language_layers = [hidden.detach() for hidden in language_layers]
            self._cached_language_mask = language_mask.detach()
            self._cached_instruction_key = tuple(instructions)
            self._cached_batch_size = batch_size
            self._predict_call_count = 0

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        vision_tokens, vision_mask = self._split_vision_tokens_by_sample(qwen_inputs)
        vl_embs_list, attention_mask = self._fuse_language_and_vision(
            self._cached_language_layers,
            self._cached_language_mask,
            vision_tokens,
            vision_mask,
        )
        attention_mask = attention_mask.to(dtype=torch.bool)

        base_hidden = vl_embs_list[-1]
        state_tensor = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state_tensor, encoder_attention_mask=attention_mask
            )

        self._predict_call_count += 1
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
