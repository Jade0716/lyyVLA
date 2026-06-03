from typing import List

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPI_v3_ActionToken")
class Qwen_PI_v3_ActionToken(Qwen_PI_v3):
    """
    QwenPI_v3 variant that lets the action head attend only to one learnable
    action-query token appended to the Qwen sequence.

    The VLM still sees the full images and language prompt. After Qwen produces
    hidden states, each selected VLM layer is reduced from (B, L, D) to (B, 1, D)
    by selecting the appended query position, and only that token is
    exposed to the layer-wise Action DiT cross-attention.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.motion_token = "<MOTION>"
        # self.motion_token_id = self._ensure_motion_special_token()

        hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)
        self.action_query_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    # def _ensure_motion_special_token(self) -> int:
    #     tokenizer = self.qwen_vl_interface.processor.tokenizer
    #     model = self.qwen_vl_interface.model
    #
    #     special_tokens_dict = {
    #         "extra_special_tokens": [
    #             self.motion_token,
    #         ]
    #     }
    #     tokenizer.add_special_tokens(special_tokens_dict, replace_extra_special_tokens=False)
    #
    #     embedding_size = model.get_input_embeddings().num_embeddings
    #     if embedding_size != len(tokenizer):
    #         model.resize_token_embeddings(len(tokenizer))
    #
    #     motion_token_id = tokenizer.convert_tokens_to_ids(self.motion_token)
    #     if motion_token_id is None or motion_token_id == tokenizer.unk_token_id:
    #         raise ValueError(f"Failed to add motion special token: {self.motion_token}")
    #     return int(motion_token_id)

    def _append_action_query(self, qwen_inputs):
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

    def _select_action_condition_hidden(
        self,
        vl_embs_list: List[torch.Tensor],
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        selected = [hidden[:, -1:, :] for hidden in vl_embs_list]
        attention_mask = torch.ones(
            selected[0].shape[0],
            1,
            dtype=torch.bool,
            device=selected[0].device,
        )
        return selected, attention_mask

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        qwen_inputs = self._append_action_query(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers:])
            vl_embs_list, attention_mask = self._select_action_condition_hidden(vl_embs_list)
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list, attention_mask
