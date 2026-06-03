from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken")
class Qwen_GR00T_ActionToken(Qwen_GR00T):
    """
    QwenGR00T variant that conditions the GR00T DiT action head on one
    learnable action-query token.

    The Qwen VLM still encodes the full image/language context. A trainable
    token is appended to the Qwen input embeddings, then only that token's final
    hidden state is exposed to the flow-matching action head cross-attention.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.action_query_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def _append_action_query(self, qwen_inputs: dict) -> dict:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        query = self.action_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        query = query.expand(inputs_embeds.shape[0], -1, -1)

        qwen_inputs["inputs_embeds"] = torch.cat([inputs_embeds, query], dim=1)
        qwen_inputs.pop("input_ids", None)

        if "attention_mask" in qwen_inputs and qwen_inputs["attention_mask"] is not None:
            query_mask = torch.ones(
                qwen_inputs["attention_mask"].shape[0],
                1,
                dtype=qwen_inputs["attention_mask"].dtype,
                device=qwen_inputs["attention_mask"].device,
            )
            qwen_inputs["attention_mask"] = torch.cat([qwen_inputs["attention_mask"], query_mask], dim=1)

        if "mm_token_type_ids" in qwen_inputs and qwen_inputs["mm_token_type_ids"] is not None:
            query_type = torch.zeros(
                qwen_inputs["mm_token_type_ids"].shape[0],
                1,
                dtype=qwen_inputs["mm_token_type_ids"].dtype,
                device=qwen_inputs["mm_token_type_ids"].device,
            )
            qwen_inputs["mm_token_type_ids"] = torch.cat([qwen_inputs["mm_token_type_ids"], query_type], dim=1)

        return qwen_inputs

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = self._append_action_query(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1][:, -1:, :]

        action_token_attention_mask = torch.ones(
            last_hidden.shape[0],
            1,
            dtype=torch.bool,
            device=last_hidden.device,
        )
        return last_hidden, action_token_attention_mask
