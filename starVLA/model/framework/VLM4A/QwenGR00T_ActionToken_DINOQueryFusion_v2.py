from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


class BBoxGuidedDinoResampler(nn.Module):
    """
    Use bbox-oriented VLM token states as queries over DINO patch tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bbox_norm = nn.LayerNorm(hidden_size)
        self.bbox_to_dino_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, bbox_hidden: torch.Tensor, dino_hidden: torch.Tensor) -> torch.Tensor:
        dino_context, _ = self.bbox_to_dino_attn(
            query=self.bbox_norm(bbox_hidden),
            key=dino_hidden,
            value=dino_hidden,
            need_weights=False,
        )
        return self.output_norm(bbox_hidden + dino_context)


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_DINOQueryFusion_v2")
class Qwen_GR00T_ActionToken_DINOQueryFusion_v2(Qwen_GR00T_ActionToken):
    """
    ActionToken + DINO query-fusion variant with separated token roles.

    The VLM receives two learned token groups appended after the text/image
    prompt:

        [bbox query tokens, motion action tokens]

    The 4 bbox tokens form a visual-selection bottleneck and query DINO patch
    tokens to produce 4 guided DINO tokens. The 4 motion tokens keep the
    original ActionToken role and remain the only states used for DCT guidance.

    The action head receives:

        [motion action tokens, bbox-guided DINO visual tokens]
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        dino_cfg = self.config.framework.get("dino", {})

        self.num_bbox_tokens = int(dino_cfg.get("num_bbox_tokens", 4))
        self.max_dino_tokens = dino_cfg.get("max_dino_tokens", None)
        self.max_dino_tokens = None if self.max_dino_tokens is None else int(self.max_dino_tokens)

        self.bbox_query_token = nn.Parameter(torch.randn(1, self.num_bbox_tokens, hidden_size) * 0.02)

        self.dino_encoder = get_dino_model(
            backone_name=dino_cfg.get("dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,
            out_features=hidden_size,
        )
        self.dino_resampler = BBoxGuidedDinoResampler(
            hidden_size=hidden_size,
            num_heads=int(dino_cfg.get("adapter_num_heads", dino_cfg.get("qformer_num_heads", 8))),
            dropout=float(dino_cfg.get("adapter_dropout", 0.0)),
        )

    def _append_action_query(self, qwen_inputs: dict) -> dict:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)

        bbox_query = self.bbox_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        bbox_query = bbox_query.expand(inputs_embeds.shape[0], -1, -1)
        motion_query = self.action_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        motion_query = motion_query.expand(inputs_embeds.shape[0], -1, -1)
        query = torch.cat([bbox_query, motion_query], dim=1)
        query_len = query.shape[1]

        qwen_inputs["inputs_embeds"] = torch.cat([inputs_embeds, query], dim=1)
        qwen_inputs.pop("input_ids", None)

        if "attention_mask" in qwen_inputs and qwen_inputs["attention_mask"] is not None:
            query_mask = torch.ones(
                qwen_inputs["attention_mask"].shape[0],
                query_len,
                dtype=qwen_inputs["attention_mask"].dtype,
                device=qwen_inputs["attention_mask"].device,
            )
            qwen_inputs["attention_mask"] = torch.cat([qwen_inputs["attention_mask"], query_mask], dim=1)

        if "mm_token_type_ids" in qwen_inputs and qwen_inputs["mm_token_type_ids"] is not None:
            query_type = torch.zeros(
                qwen_inputs["mm_token_type_ids"].shape[0],
                query_len,
                dtype=qwen_inputs["mm_token_type_ids"].dtype,
                device=qwen_inputs["mm_token_type_ids"].device,
            )
            qwen_inputs["mm_token_type_ids"] = torch.cat([qwen_inputs["mm_token_type_ids"], query_type], dim=1)

        return qwen_inputs

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
            if self.max_dino_tokens is not None:
                dino_features = dino_features[:, : self.max_dino_tokens, :]
            dino_hidden = self.dino_pro(dino_features)
        return dino_hidden.to(dtype=dtype)

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = self._append_action_query(qwen_inputs)
        query_len = self.num_bbox_tokens + self.motion_dct_keep_freq

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            query_hidden = qwenvl_outputs.hidden_states[-1][:, -query_len:, :]

        bbox_hidden = query_hidden[:, : self.num_bbox_tokens, :]
        motion_hidden = query_hidden[:, self.num_bbox_tokens :, :]

        dino_hidden = self._encode_dino_hidden_states(
            batch_images=batch_images,
            dtype=query_hidden.dtype,
        ).to(device=query_hidden.device)

        dino_visual_tokens = self.dino_resampler(
            bbox_hidden=bbox_hidden,
            dino_hidden=dino_hidden,
        )

        action_condition = torch.cat([motion_hidden, dino_visual_tokens], dim=1)
        action_condition_attention_mask = torch.ones(
            action_condition.shape[0],
            action_condition.shape[1],
            dtype=torch.bool,
            device=action_condition.device,
        )
        return action_condition, action_condition_attention_mask

    def _compute_motion_dct_loss(
        self,
        action_condition_hidden: torch.Tensor,
        actions: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor:
        motion_hidden = action_condition_hidden[:, : self.motion_dct_keep_freq, :]
        return super()._compute_motion_dct_loss(motion_hidden, actions, chunk_len)
