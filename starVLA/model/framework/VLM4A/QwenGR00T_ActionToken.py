from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken")
class Qwen_GR00T_ActionToken(Qwen_GR00T):
    """
    QwenGR00T variant that conditions the GR00T DiT action head on learned
    action-query tokens.

    The Qwen VLM still encodes the full image/language context. A trainable
    token block is appended to the Qwen input embeddings, then only those token
    hidden states are exposed to the flow-matching action head cross-attention.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)
        self.motion_dct_keep_freq = int(qwenvl_cfg.get("motion_dct_keep_freq", 1))
        self.motion_dct_chunk_len = int(qwenvl_cfg.get("motion_dct_chunk_len", self.action_horizon))
        self.use_motion_dct_loss = _as_bool(qwenvl_cfg.get("use_motion_dct_loss", False))
        self.motion_dct_loss_weight = float(qwenvl_cfg.get("motion_dct_loss_weight", 1.0))
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
            last_hidden = qwenvl_outputs.hidden_states[-1][:, -self.motion_dct_keep_freq :, :]

        action_token_attention_mask = torch.ones(
            last_hidden.shape[0],
            last_hidden.shape[1],
            dtype=torch.bool,
            device=last_hidden.device,
        )
        return last_hidden, action_token_attention_mask

    def _motion_dct_target(self, actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
        action_chunk = actions[:, :chunk_len, : self.motion_dct_action_dim].detach().float().cpu().numpy()
        low_dct_gt = dct(action_chunk, type=2, axis=1, norm="ortho")[:, : self.motion_dct_keep_freq, :]
        if low_dct_gt.shape[1] < self.motion_dct_keep_freq:
            pad_len = self.motion_dct_keep_freq - low_dct_gt.shape[1]
            low_dct_gt = np.pad(low_dct_gt, ((0, 0), (0, pad_len), (0, 0)))
        return torch.from_numpy(low_dct_gt).to(device=actions.device, dtype=actions.dtype)

    def _compute_motion_dct_loss(
        self,
        action_token_hidden: torch.Tensor,
        actions: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor:
        if not self.use_motion_dct_loss:
            return action_token_hidden.new_zeros(())
        low_dct_gt = self._motion_dct_target(actions, chunk_len).to(
            device=action_token_hidden.device,
            dtype=action_token_hidden.dtype,
        )
        low_dct_pred = self.motion_dct_head(action_token_hidden)
        return F.mse_loss(low_dct_pred.float(), low_dct_gt.float())

    def _prepare_actions_target(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions to have shape [B, T, D], got {tuple(actions.shape)}.")
        if actions.shape[1] != self.action_horizon:
            raise ValueError(
                "ActionToken expects dataloader action length to equal action_horizon "
                f"so DiT and DCT use the same chunk; got T={actions.shape[1]} and "
                f"action_horizon={self.action_horizon}."
            )
        return actions

    @staticmethod
    def _grad_norm_wrt_hidden(loss: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return hidden.new_zeros(())
        grad = torch.autograd.grad(
            loss,
            hidden,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return hidden.new_zeros(())
        return grad.detach().float().norm(p=2)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        action_token_hidden, action_token_attention_mask = self._encode_vl_hidden_states(
            batch_images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=action_token_hidden.device, dtype=action_token_hidden.dtype
            )
            actions_target = self._prepare_actions_target(actions)

        motion_chunk_len = self.action_horizon
        motion_dct_loss = self._compute_motion_dct_loss(
            action_token_hidden,
            actions_target,
            motion_chunk_len,
        )
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss

        repeated_diffusion_steps = 16
        if self.config and hasattr(self.config, "trainer"):
            repeated_diffusion_steps = int(self.config.trainer.get("repeated_diffusion_steps", 16))

        actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
        action_token_hidden_repeated = action_token_hidden.repeat(repeated_diffusion_steps, 1, 1)
        if action_token_attention_mask is not None:
            action_token_attention_mask = action_token_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                dtype=torch.bool
            )

        state_repeated = None
        if state is not None:
            state = torch.tensor(np.array(state), device=action_token_hidden.device, dtype=action_token_hidden.dtype)
            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

        action_dit_loss = self.action_model(
            action_token_hidden_repeated,
            actions_target_repeated,
            state_repeated,
            encoder_attention_mask=action_token_attention_mask,
        )
        total_loss = action_dit_loss + weighted_motion_dct_loss

        output = {
            "action_loss": total_loss,
            "action_dit_loss": action_dit_loss,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
        }
        if kwargs.get("log_actiontoken_grad_norm", False):
            output.update(
                {
                    "grad_norm/action_token/action_dit_loss": self._grad_norm_wrt_hidden(
                        action_dit_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/motion_dct_loss": self._grad_norm_wrt_hidden(
                        motion_dct_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/weighted_motion_dct_loss": self._grad_norm_wrt_hidden(
                        weighted_motion_dct_loss, action_token_hidden
                    ),
                }
            )
        return output
