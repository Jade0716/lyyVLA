"""QwenPI-style layer-wise VLM conditioning with an L1 action head.

This ablation keeps the defining QwenPI connection: each selected text-decoder
layer provides its complete token sequence to the matching action-head
attention block. It replaces flow matching with the deterministic attention
regressor used by the TwoChunk family and trains it directly with L1 action
loss. The number of selected final VLM layers is controlled by
``action_model.gated_num_blocks``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    add_discretized_state_to_instruction,
    merge_framework_config,
)
from starVLA.model.modules.action_model.GatedAttentionActionHeader import (
    GatedAttentionActionHead,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenPI_v3_L1DefaultConfig:
    name: str = "QwenPI_v3_L1"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_head_type": "gated_attention",
            "hidden_size": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 32,
            "gated_num_blocks": 16,
            "gated_num_heads": 8,
            "gated_use_rope": True,
            "zero_init_output": True,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v3_L1")
class Qwen_PI_v3_L1(baseframework):
    """QwenPI layer-wise cross-attention ablation with configurable L1 chunks."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenPI_v3_L1DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        self.total_vl_layers = int(text_cfg.num_hidden_layers)
        vl_hidden_dim = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = vl_hidden_dim
        self.config.framework.qwenvl.num_vl_layers = self.total_vl_layers

        action_cfg = self.config.framework.action_model
        self.num_action_layers = int(action_cfg.get("gated_num_blocks", self.total_vl_layers))
        if self.num_action_layers <= 0 or self.num_action_layers > self.total_vl_layers:
            raise ValueError(
                "QwenPI_v3_L1 requires gated_num_blocks in [1, num_vl_layers]; "
                f"got {self.num_action_layers} for a {self.total_vl_layers}-layer VLM."
            )
        self.action_horizon = int(action_cfg.action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(
                "QwenPI_v3_L1 requires a positive action_horizon; "
                f"got action_horizon={self.action_horizon}."
            )
        vla_data_cfg = self.config.datasets.get("vla_data", {})
        action_window_size = vla_data_cfg.get("action_window_size", None)
        if action_window_size is not None and int(action_window_size) != self.action_horizon:
            raise ValueError(
                "QwenPI_v3_L1 requires datasets.vla_data.action_window_size to "
                "match framework.action_model.action_horizon; got "
                f"action_window_size={action_window_size} and "
                f"action_horizon={self.action_horizon}."
            )

        action_hidden_dim = int(action_cfg.get("hidden_size", vl_hidden_dim))
        num_heads = int(action_cfg.get("gated_num_heads", 8))
        if action_hidden_dim % num_heads != 0:
            raise ValueError(
                f"action hidden size {action_hidden_dim} must be divisible by "
                f"gated_num_heads={num_heads}."
            )

        # Match each action block to one of the selected final VLM layers.
        # All valid sequence tokens are retained for every selected layer.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if vl_hidden_dim == action_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(vl_hidden_dim),
                        nn.Linear(vl_hidden_dim, action_hidden_dim),
                    )
                )
                for _ in range(self.num_action_layers)
            ]
        )
        self.action_model = GatedAttentionActionHead(
            input_dim=action_hidden_dim,
            hidden_dim=action_hidden_dim,
            action_dim=int(action_cfg.action_dim),
            NUM_ACTIONS_CHUNK=self.action_horizon,
            num_blocks=self.num_action_layers,
            num_heads=num_heads,
            use_rope=bool(action_cfg.get("gated_use_rope", True)),
            separate_condition_paths=False,
            zero_init_output=bool(action_cfg.get("zero_init_output", True)),
        )
        self.l1_loss = nn.L1Loss()

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> tuple[List[torch.Tensor], torch.Tensor | None]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_layers = list(outputs.hidden_states[-self.num_action_layers :])
            vl_layers = [
                projector(hidden)
                for projector, hidden in zip(self.project_layers, vl_layers)
            ]
        return vl_layers, attention_mask

    def _prepare_instructions(self, examples: List[dict]) -> List[str]:
        instructions = [example["lang"] for example in examples]
        if "state" in examples[0]:
            states = [example["state"] for example in examples]
            instructions = add_discretized_state_to_instruction(instructions, states)
        return instructions

    def _predict_from_layers(
        self,
        vl_layers: List[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if len(vl_layers) != len(self.action_model.blocks):
            raise ValueError(
                f"Got {len(vl_layers)} VLM layers for "
                f"{len(self.action_model.blocks)} action attention blocks."
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        return self.action_model.predict_action(
            vl_layers[-1],
            condition_layers=vl_layers,
            condition_attention_mask=attention_mask,
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = self._prepare_instructions(examples)
        vl_layers, attention_mask = self._encode_vl_hidden_states(
            batch_images,
            instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self._predict_from_layers(vl_layers, attention_mask)
            actions = torch.as_tensor(
                np.asarray([example["action"] for example in examples]),
                device=pred_actions.device,
                dtype=pred_actions.dtype,
            )
            if actions.shape[1] < self.action_horizon:
                raise ValueError(
                    f"Expected at least {self.action_horizon} target actions, "
                    f"got shape {tuple(actions.shape)}."
                )
            actions_target = actions[:, -self.action_horizon :, :]
            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = self._prepare_instructions(examples)

        train_obs_image_size = getattr(
            self.config.datasets.vla_data,
            "obs_image_size",
            None,
        )
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_layers, attention_mask = self._encode_vl_hidden_states(
            batch_images,
            instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self._predict_from_layers(vl_layers, attention_mask)

        return {"normalized_actions": pred_actions.float().cpu().numpy()}
