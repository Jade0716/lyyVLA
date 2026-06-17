from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


class DinoToActionQueryAdapter(nn.Module):
    """
    Use action-query hidden states as queries to retrieve DINO patch-token context.

    Input:
        action_q:  [B, Q, H]
        dino_feat: [B, L, H]

    Output:
        fused action-query-shaped tokens: [B, Q, H]
    """

    def __init__(self, hidden_size: int, num_heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, action_q: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross_attn(
            query=action_q,
            key=dino_feat,
            value=dino_feat,
            need_weights=False,
        )
        return action_q + self.gate * out


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_DINOQueryFusion")
class Qwen_GR00T_ActionToken_DINOQueryFusion(Qwen_GR00T_ActionToken):
    """
    QwenGR00T ActionToken variant that lets action-query hidden states query
    DINO patch tokens before DiT conditioning.

    The action head receives:

        [original action-query hidden states, DINO-fused action-query states]

    This keeps the DINO contribution action-query-shaped instead of appending
    hundreds of raw DINO tokens. DCT guidance remains attached only to the
    original action-query hidden states.
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
        self.dino_action_query_adapter = DinoToActionQueryAdapter(
            hidden_size=hidden_size,
            num_heads=int(dino_cfg.get("adapter_num_heads", dino_cfg.get("qformer_num_heads", 8))),
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

        dino_fused_action_query = self.dino_action_query_adapter(
            action_q=action_query_hidden,
            dino_feat=dino_hidden,
        )
        action_condition = torch.cat([action_query_hidden, dino_fused_action_query], dim=1)

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
                action_query_attention_mask.to(dtype=torch.bool),
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
