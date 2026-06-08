from typing import List

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


class ActionDimDCTHead(nn.Module):
    """Predict one action dimension's low-frequency DCT coefficients per token."""

    def __init__(self, hidden_size=1024, keep_freq=8, action_dim=7):
        super().__init__()
        self.keep_freq = keep_freq
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, keep_freq),
        )

    def forward(self, action_dim_hiddens: torch.Tensor) -> torch.Tensor:
        if action_dim_hiddens.dim() != 3:
            raise ValueError(f"ActionDimDCTHead expects [B, action_dim, H], got {tuple(action_dim_hiddens.shape)}")
        if action_dim_hiddens.shape[1] != self.action_dim:
            raise ValueError(
                f"ActionDimDCTHead expects {self.action_dim} action tokens, "
                f"got {action_dim_hiddens.shape[1]}"
            )

        # [B, action_dim, keep_freq] -> [B, keep_freq, action_dim], matching the DCT target.
        coeffs = self.net(action_dim_hiddens)
        return coeffs.transpose(1, 2).contiguous()


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_7ActionDCT")
class Qwen_GR00T_ActionToken_TwoChunk_7ActionDCT(Qwen_GR00T_ActionToken_TwoChunk):
    """TwoChunk variant with one action token per action dimension for DCT supervision."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)
        self.action_query_token = nn.Parameter(torch.randn(1, action_dim, hidden_size) * 0.02)
        self.motion_dct_head = ActionDimDCTHead(
            hidden_size=hidden_size,
            keep_freq=self.motion_dct_keep_freq,
            action_dim=action_dim,
        )

    def _append_action_query(self, qwen_inputs: dict) -> dict:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        query = self.action_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        query = query.expand(inputs_embeds.shape[0], -1, -1)
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

    def _encode_action_token_hidden(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = self._append_action_query(qwen_inputs)
        query_len = self.action_query_token.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return qwenvl_outputs.hidden_states[-1][:, -query_len:, :]
