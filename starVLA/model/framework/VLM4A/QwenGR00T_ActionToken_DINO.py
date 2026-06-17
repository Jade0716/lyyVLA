from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_DINO")
class Qwen_GR00T_ActionToken_DINO(Qwen_GR00T_ActionToken):
    """
    QwenGR00T ActionToken variant that conditions DiT on action-query states
    plus DINO patch tokens from the same observation images.

    The VLM path still appends and returns the learned action query hidden
    states. DINO runs in parallel on the images, projects its patch tokens to the
    Qwen hidden size, and the action head receives:

        [action query hidden states, DINO patch tokens]

    DCT guidance remains attached only to the action-query hidden states.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        dino_cfg = self.config.framework.get("dino", {})
        self.dino_encoder = get_dino_model(
            backone_name=dino_cfg.get("dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,
            out_features=hidden_size,
        )

    def _encode_dino_hidden_states(
        self,
        batch_images: List,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        batch_size = len(batch_images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dino_features = self.dino_encoder(image_tensors)
            dino_features = dino_features.reshape(batch_size, -1, dino_features.shape[-1])
            dino_hidden = self.dino_pro(dino_features)
        return dino_hidden.to(dtype=dtype)

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        action_query_hidden, action_query_attention_mask = super()._encode_vl_hidden_states(
            batch_images=batch_images,
            instructions=instructions,
        )
        dino_hidden = self._encode_dino_hidden_states(
            batch_images=batch_images,
            dtype=action_query_hidden.dtype,
        ).to(device=action_query_hidden.device)

        action_condition = torch.cat([action_query_hidden, dino_hidden], dim=1)
        dino_attention_mask = torch.ones(
            dino_hidden.shape[0],
            dino_hidden.shape[1],
            dtype=torch.bool,
            device=dino_hidden.device,
        )
        if action_query_attention_mask is None:
            return action_condition, torch.ones(
                action_condition.shape[0],
                action_condition.shape[1],
                dtype=torch.bool,
                device=action_condition.device,
            )

        action_condition_attention_mask = torch.cat(
            [
                action_query_attention_mask.to(dtype=torch.bool),
                dino_attention_mask,
            ],
            dim=1,
        )
        return action_condition, action_condition_attention_mask

    def _compute_motion_dct_loss(
        self,
        action_condition_hidden: torch.Tensor,
        actions: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor:
        action_query_hidden = action_condition_hidden[:, : self.motion_dct_keep_freq, :]
        return super()._compute_motion_dct_loss(action_query_hidden, actions, chunk_len)
