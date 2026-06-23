from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import idct

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken, _as_bool
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_IDCTCondition")
class Qwen_GR00T_ActionToken_IDCTCondition(Qwen_GR00T_ActionToken):
    """
    ActionToken variant that feeds a low-frequency IDCT action prior to the
    GR00T DiT as extra cross-attention condition tokens.

    The DiT flow-matching objective is unchanged: it still predicts velocity
    from noise to target action. The IDCT trajectory is only a conditioning
    signal by default, detached from the DiT loss so the DCT head remains
    supervised by the explicit motion DCT loss.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)

        self.use_idct_condition = _as_bool(qwenvl_cfg.get("use_idct_condition", True))
        self.detach_idct_condition = _as_bool(qwenvl_cfg.get("detach_idct_condition", True))
        self.idct_condition_token_mode = str(qwenvl_cfg.get("idct_condition_token_mode", "per_timestep"))

        self.idct_condition_proj = nn.Sequential(
            nn.Linear(self.motion_dct_action_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._refresh_idct_basis(chunk_len=self.action_horizon)

    def _refresh_idct_basis(self, chunk_len: int) -> None:
        basis_key = f"_idct_basis_{chunk_len}_{self.motion_dct_keep_freq}"
        if hasattr(self, basis_key):
            self._active_idct_basis_name = basis_key
            return

        coeff_eye = np.zeros((self.motion_dct_keep_freq, chunk_len), dtype=np.float32)
        coeff_eye[:, : self.motion_dct_keep_freq] = np.eye(self.motion_dct_keep_freq, dtype=np.float32)
        basis = idct(coeff_eye, type=2, n=chunk_len, axis=1, norm="ortho").T
        self.register_buffer(basis_key, torch.from_numpy(basis), persistent=False)
        self._active_idct_basis_name = basis_key

    def _idct_basis(self, chunk_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self._refresh_idct_basis(chunk_len)
        return getattr(self, self._active_idct_basis_name).to(device=device, dtype=dtype)

    def _predict_low_dct(self, action_token_hidden: torch.Tensor) -> torch.Tensor:
        if self.motion_dct_head is None:
            raise RuntimeError("QwenGR00T_ActionToken_IDCTCondition requires use_motion_dct_loss=true.")
        return self.motion_dct_head(action_token_hidden)

    def _idct_condition_tokens(
        self,
        action_token_hidden: torch.Tensor,
        chunk_len: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_idct_condition:
            return None, None

        low_dct_pred = self._predict_low_dct(action_token_hidden)
        basis = self._idct_basis(
            chunk_len=chunk_len,
            device=low_dct_pred.device,
            dtype=low_dct_pred.dtype,
        )
        coarse_action = torch.einsum("bkd,tk->btd", low_dct_pred, basis)
        if self.detach_idct_condition:
            coarse_action = coarse_action.detach()

        if self.idct_condition_token_mode != "per_timestep":
            raise ValueError(
                "Only idct_condition_token_mode='per_timestep' is currently supported; "
                f"got {self.idct_condition_token_mode!r}."
            )
        idct_tokens = self.idct_condition_proj(coarse_action)
        idct_mask = torch.ones(
            idct_tokens.shape[:2],
            dtype=torch.bool,
            device=idct_tokens.device,
        )
        return idct_tokens, idct_mask

    def _append_idct_condition(
        self,
        action_token_hidden: torch.Tensor,
        action_token_attention_mask: Optional[torch.Tensor],
        chunk_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idct_tokens, idct_mask = self._idct_condition_tokens(action_token_hidden, chunk_len)
        if idct_tokens is None:
            return action_token_hidden, action_token_attention_mask

        condition_hidden = torch.cat([action_token_hidden, idct_tokens], dim=1)
        if action_token_attention_mask is None:
            return condition_hidden, None
        condition_mask = torch.cat([action_token_attention_mask.to(dtype=torch.bool), idct_mask], dim=1)
        return condition_hidden, condition_mask

    def _action_dit_loss_weight(self, train_step: Optional[int] = None) -> float:
        if not self.config or not hasattr(self.config, "trainer"):
            return 1.0
        if not _as_bool(self.config.trainer.get("action_dit_loss_warmup", False)):
            return 1.0
        if train_step is None:
            return 1.0

        start_step = int(self.config.trainer.get("action_dit_loss_start_step", 0))
        warmup_steps = int(self.config.trainer.get("action_dit_loss_warmup_steps", 0))
        if train_step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return 1.0
        progress = float(train_step - start_step) / float(warmup_steps)
        return min(1.0, max(0.0, progress))

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
        action_dit_loss_weight = self._action_dit_loss_weight(kwargs.get("train_step", None))

        condition_hidden, condition_attention_mask = self._append_idct_condition(
            action_token_hidden,
            action_token_attention_mask,
            chunk_len=self.action_horizon,
        )

        repeated_diffusion_steps = 16
        if self.config and hasattr(self.config, "trainer"):
            repeated_diffusion_steps = int(self.config.trainer.get("repeated_diffusion_steps", 16))

        actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
        condition_hidden_repeated = condition_hidden.repeat(repeated_diffusion_steps, 1, 1)
        if condition_attention_mask is not None:
            condition_attention_mask = condition_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                dtype=torch.bool
            )

        state_repeated = None
        if state is not None:
            state = torch.tensor(np.array(state), device=condition_hidden.device, dtype=condition_hidden.dtype)
            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

        action_dit_loss = self.action_model(
            condition_hidden_repeated,
            actions_target_repeated,
            state_repeated,
            encoder_attention_mask=condition_attention_mask,
        )
        total_loss = action_dit_loss_weight * action_dit_loss + weighted_motion_dct_loss

        output = {
            "action_loss": total_loss,
            "action_dit_loss": action_dit_loss,
            "action_dit_loss_weight": action_dit_loss_weight,
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

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [self._to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            from starVLA.training.trainer_utils.trainer_tools import resize_images

            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_token_hidden, action_token_attention_mask = self._encode_vl_hidden_states(
            batch_images=batch_images, instructions=instructions
        )
        if action_token_attention_mask is not None:
            action_token_attention_mask = action_token_attention_mask.to(dtype=torch.bool)

        condition_hidden, condition_attention_mask = self._append_idct_condition(
            action_token_hidden,
            action_token_attention_mask,
            chunk_len=self.action_horizon,
        )

        state = (
            torch.tensor(np.array(state), device=condition_hidden.device, dtype=condition_hidden.dtype)
            if state is not None
            else None
        )
        actions = self.action_model.predict_action(
            condition_hidden,
            state,
            encoder_attention_mask=condition_attention_mask,
        )
        return {"normalized_actions": actions.float().cpu().numpy()}

    @staticmethod
    def _to_pil_preserve(image):
        from deployment.model_server.tools.image_tools import to_pil_preserve

        return to_pil_preserve(image)
