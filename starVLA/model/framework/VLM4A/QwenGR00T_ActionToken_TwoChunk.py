import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import idct

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import (
    Qwen_GR00T_ActionToken,
    _as_bool,
)
from starVLA.model.modules.action_model.GatedAttentionActionHeader import GatedAttentionActionHead
from starVLA.model.modules.action_model.MLP_ActionHeader_TwoChunk import L1RegressionActionHead
from starVLA.model.modules.dino_model.dinov3 import get_twochunk_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk")
class Qwen_GR00T_ActionToken_TwoChunk(Qwen_GR00T_ActionToken):
    """
    Two-chunk ActionToken variant with:
      - slow chunk: Qwen hidden states from 8 learnable action tokens
      - DCT head: predict low-frequency coefficients for all action dimensions,
        including the gripper
      - fast chunk: DINO tokens fused with the raw slow action-token hidden
      - action head: regress short-chunk residual actions against (GT - IDCT prior)
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        dino_cfg = self.config.framework.get("dino", {})
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)

        self.language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 64))
        self.vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", self.action_horizon))
        self.fast_chunk_size = self.vision_refresh_steps
        self.twochunk_window_size = int(qwenvl_cfg.get("twochunk_window_size", self.language_refresh_steps))

        self.motion_dct_keep_freq = int(qwenvl_cfg.get("motion_dct_keep_freq", 8))
        self.motion_dct_chunk_len = int(qwenvl_cfg.get("motion_dct_chunk_len", self.language_refresh_steps))
        self.use_motion_dct_loss = _as_bool(qwenvl_cfg.get("use_motion_dct_loss", True))
        self.motion_dct_loss_weight = float(qwenvl_cfg.get("motion_dct_loss_weight", 1.0))
        self.detach_idct_condition = _as_bool(qwenvl_cfg.get("detach_idct_condition", True))

        # if self.motion_dct_keep_freq != 8:
        #     raise ValueError(
        #         "QwenGR00T_ActionToken_TwoChunk expects framework.qwenvl.motion_dct_keep_freq=8 "
        #         f"for the requested 8 slow tokens, got {self.motion_dct_keep_freq}."
        #     )

        self.action_model.action_horizon = self.fast_chunk_size
        if hasattr(self.action_model, "config"):
            self.action_model.config.action_horizon = self.fast_chunk_size

        # TwoChunk uses the coarse IDCT trajectory as an additive prior for the
        # full action vector. Include gripper so the residual head refines all
        # dimensions instead of predicting gripper entirely from scratch.
        self.motion_dct_action_dim = action_dim
        self.action_query_token = nn.Parameter(torch.randn(1, self.motion_dct_keep_freq, hidden_size) * 0.02)
        self.motion_dct_head = None
        if self.use_motion_dct_loss:
            self.motion_dct_head = L1RegressionActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                action_dim=self.motion_dct_action_dim,
                NUM_ACTIONS_CHUNK=self.motion_dct_keep_freq,
            )

        self.dino_encoder = get_twochunk_dino_model(
            backbone_name=dino_cfg.get("dino_backbone", "dinov3_vits16plus"),
            repo_path=dino_cfg.get("dino_repo_path", "/home/liuyuyan/dinov3"),
            weights_path=dino_cfg.get(
                "dino_weights_path",
                "/mnt/8ac36469-5f21-42a9-a6dd-21bfcb724d52/liuyuyan/DINO/"
                "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            ),
            input_size=int(dino_cfg.get("dino_input_size", 256)),
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,
            out_features=hidden_size,
        )
        self.action_model = self._build_fast_action_head(
            hidden_size=hidden_size,
            action_dim=action_dim,
        )
        self.l1_loss = nn.L1Loss()

        self._cached_action_token_hidden = None
        self._cached_coarse_action = None
        self._cached_instruction_key = None
        self._cached_batch_size = None
        self._predict_call_count = 0

        self._refresh_idct_basis(chunk_len=self.motion_dct_chunk_len)

    def _build_fast_action_head(
        self,
        hidden_size: int,
        action_dim: int,
    ) -> nn.Module:
        action_cfg = self.config.framework.get("action_model", {})
        head_type = str(action_cfg.get("action_head_type", "pooling_mlp")).lower()
        hidden_dim = int(action_cfg.get("hidden_size", hidden_size))

        if head_type in {"gated_attention", "vla_adapter", "adapter"}:
            return GatedAttentionActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                NUM_ACTIONS_CHUNK=self.fast_chunk_size,
                num_blocks=int(action_cfg.get("gated_num_blocks", 8)),
                num_heads=int(action_cfg.get("gated_num_heads", 8)),
                use_rope=_as_bool(action_cfg.get("gated_use_rope", True)),
                adapter_token_count=int(action_cfg.get("gated_adapter_token_count", self.motion_dct_keep_freq)),
            )
        if head_type not in {"pooling_mlp", "mlp", "legacy"}:
            raise ValueError(
                f"Unknown action_model.action_head_type={head_type}. "
                "Expected one of: pooling_mlp, gated_attention."
            )
        return L1RegressionActionHead(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=self.fast_chunk_size,
        )

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
            raise RuntimeError("QwenGR00T_ActionToken_TwoChunk requires use_motion_dct_loss=true.")
        return self.motion_dct_head(action_token_hidden)

    def _predict_coarse_action(self, action_token_hidden: torch.Tensor, chunk_len: int) -> torch.Tensor:
        low_dct_pred = self._predict_low_dct(action_token_hidden)
        basis = self._idct_basis(chunk_len=chunk_len, device=low_dct_pred.device, dtype=low_dct_pred.dtype)
        coarse_action = torch.einsum("bkd,tk->btd", low_dct_pred, basis)
        return coarse_action

    def _coarse_with_gripper_pad(
        self,
        coarse_action: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        if coarse_action.shape[-1] == action_dim:
            return coarse_action
        padded = coarse_action.new_zeros(coarse_action.shape[0], coarse_action.shape[1], action_dim)
        padded[:, :, : coarse_action.shape[-1]] = coarse_action
        return padded

    def _action_loss_weight(self, train_step=None) -> float:
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

    def _encode_action_token_hidden(
        self,
        batch_images: List,
        instructions: List[str],
        qwen_inputs: dict | None = None,
    ) -> torch.Tensor:
        if qwen_inputs is None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
            )
        qwen_inputs = self._append_action_query(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ActionToken only consumes the multimodal backbone hidden state.
            # Calling ForConditionalGeneration would additionally project every
            # sequence token through the large vocabulary lm_head, even though
            # those logits are discarded.
            qwenvl_outputs = self.qwen_vl_interface.model.model(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            return qwenvl_outputs.last_hidden_state[:, -self.motion_dct_keep_freq :, :]

    def _encode_dino_hidden_states(
        self,
        batch_images: List,
        dtype: torch.dtype,
        image_tensors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_tensors is None:
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        batch_size = len(batch_images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dino_features = self.dino_encoder(image_tensors)
            dino_features = dino_features.reshape(batch_size, -1, dino_features.shape[-1])
            dino_hidden = self.dino_pro(dino_features)
        return dino_hidden.to(dtype=dtype)

    def _build_action_condition(
        self,
        action_token_hidden: torch.Tensor,
        frame_images: List,
        dino_image_tensors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dino_hidden = self._encode_dino_hidden_states(
            batch_images=frame_images,
            dtype=action_token_hidden.dtype,
            image_tensors=dino_image_tensors,
        ).to(device=action_token_hidden.device)
        condition_hidden = torch.cat([action_token_hidden, dino_hidden], dim=1)
        return condition_hidden

    @staticmethod
    def _first_refresh_images(image_sequences: List[List]) -> List[List]:
        return [image_sequence[0] for image_sequence in image_sequences]

    @staticmethod
    def _to_qwen_batch_images(batch_images: List[List]) -> List[List]:
        return [[to_pil_preserve(image) for image in views] for views in batch_images]

    @staticmethod
    def _to_pil_nested(images):
        if isinstance(images, list):
            return [Qwen_GR00T_ActionToken_TwoChunk._to_pil_nested(image) for image in images]
        return to_pil_preserve(images)

    def _valid_training_refreshes(self, image_sequences: List[List], actions: torch.Tensor) -> int:
        max_refreshes = max(1, min(self.twochunk_window_size, self.motion_dct_chunk_len) // max(self.vision_refresh_steps, 1))
        num_refreshes = min(min(len(seq) for seq in image_sequences), max_refreshes)
        while num_refreshes > 0:
            start = (num_refreshes - 1) * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            if end <= min(actions.shape[1], self.motion_dct_chunk_len):
                return num_refreshes
            num_refreshes -= 1
        return 0

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ):
        if examples and "image_sequence" not in examples[0]:
            return super().forward(examples=examples, **kwargs)

        profile_timing = bool(kwargs.get("profile_model_time", False))
        timing = {}
        if profile_timing:
            self._sync_cuda_if_needed()
        timing_last = time.perf_counter()

        def mark_timing(name: str) -> None:
            nonlocal timing_last
            if not profile_timing:
                return
            self._sync_cuda_if_needed()
            now = time.perf_counter()
            timing[f"timing/twochunk/{name}"] = now - timing_last
            timing_last = now

        image_sequences = [example["image_sequence"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        actions = torch.tensor(
            np.array(actions),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
        mark_timing("actions_to_tensor")
        num_refreshes = self._valid_training_refreshes(image_sequences, actions)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk forward received no valid action chunks. "
                f"actions.shape={tuple(actions.shape)}, fast_chunk_size={self.fast_chunk_size}, "
                f"vision_refresh_steps={self.vision_refresh_steps}, motion_dct_chunk_len={self.motion_dct_chunk_len}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        qwen_first_frame_images = self._to_qwen_batch_images(first_frame_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=qwen_first_frame_images,
            instructions=instructions,
        )
        mark_timing("qwen_build_inputs")
        action_token_hidden = self._encode_action_token_hidden(
            qwen_first_frame_images,
            instructions,
            qwen_inputs=qwen_inputs,
        )
        mark_timing("qwen_forward")

        long_chunk_len = min(self.motion_dct_chunk_len, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(action_token_hidden, actions, long_chunk_len)
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss
        action_loss_weight = self._action_loss_weight(kwargs.get("train_step", None))
        mark_timing("motion_dct_loss")

        coarse_long_action = self._predict_coarse_action(action_token_hidden, long_chunk_len)
        if self.detach_idct_condition:
            coarse_long_action = coarse_long_action.detach()
        coarse_long_action = self._coarse_with_gripper_pad(coarse_long_action, action_dim=actions.shape[-1])
        mark_timing("coarse_idct_prior")

        flat_frame_images = []
        residual_action_targets = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            residual_action_targets.append(actions[:, start:end, :] - coarse_long_action[:, start:end, :])

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        mark_timing("build_residual_targets")
        dino_image_tensors = self.dino_encoder.prepare_dino_input(flat_frame_images)
        mark_timing("dino_prepare_input")
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            dino_image_tensors=dino_image_tensors,
        )
        mark_timing("dino_encode_project_concat")
        residual_action_targets = torch.cat(residual_action_targets, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )
        mark_timing("targets_to_device")

        with torch.autocast("cuda", dtype=torch.float32):
            pred_residual_actions = self.action_model.predict_action(fused_hidden)
            mark_timing("action_head_predict")
            action_loss = self.l1_loss(pred_residual_actions, residual_action_targets)
            mark_timing("action_loss")

        total_loss = action_loss_weight * action_loss + weighted_motion_dct_loss
        mark_timing("total_loss")
        output = {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "action_dit_loss_weight": action_loss_weight,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
        }
        if kwargs.get("log_actiontoken_grad_norm", False):
            output.update(
                {
                    "grad_norm/action_token/action_dit_loss": self._grad_norm_wrt_hidden(
                        action_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/motion_dct_loss": self._grad_norm_wrt_hidden(
                        motion_dct_loss, action_token_hidden
                    ),
                    "grad_norm/action_token/weighted_motion_dct_loss": self._grad_norm_wrt_hidden(
                        weighted_motion_dct_loss, action_token_hidden
                    ),
                }
            )
        output.update(timing)
        return output

    def _should_refresh_action_token(self, instructions: List[str], batch_size: int) -> bool:
        if self._cached_action_token_hidden is None:
            return True
        if self._cached_batch_size != batch_size:
            return True
        if self._cached_instruction_key != tuple(instructions):
            return True

        steps_since_refresh = self._predict_call_count * max(int(self.fast_chunk_size), 1)
        return steps_since_refresh % self.language_refresh_steps == 0

    def _reset_predict_cache(self) -> None:
        self._cached_action_token_hidden = None
        self._cached_coarse_action = None
        self._cached_instruction_key = None
        self._cached_batch_size = None
        self._predict_call_count = 0

    @staticmethod
    def _sync_cuda_if_needed() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        if type(examples) is not list:
            examples = [examples]

        if examples and "image_sequence" in examples[0]:
            return self._predict_action_window(examples, **kwargs)

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        batch_size = len(examples)
        debug_twochunk = bool(kwargs.get("debug_twochunk", False))
        reset_cache = bool(kwargs.get("reset_cache", False))
        if reset_cache:
            self._reset_predict_cache()

        slow_refresh = self._should_refresh_action_token(instructions, batch_size)
        slow_time_s = 0.0
        if slow_refresh:
            # Match VLA-Adapter: processor/tokenization is preprocessing and is
            # excluded from synchronized model inference time.
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
            )
            self._sync_cuda_if_needed()
            slow_start = time.perf_counter()
            self._cached_action_token_hidden = self._encode_action_token_hidden(
                batch_images,
                instructions,
                qwen_inputs=qwen_inputs,
            ).detach()
            self._cached_coarse_action = self._predict_coarse_action(
                self._cached_action_token_hidden,
                self.motion_dct_chunk_len,
            ).detach()
            self._cached_instruction_key = tuple(instructions)
            self._cached_batch_size = batch_size
            self._predict_call_count = 0
            self._sync_cuda_if_needed()
            slow_time_s = time.perf_counter() - slow_start

        start = self._predict_call_count * self.fast_chunk_size
        end = start + self.fast_chunk_size
        if end > self.motion_dct_chunk_len:
            self._predict_call_count = 0
            start = 0
            end = self.fast_chunk_size

        # DINO image tensor construction is preprocessing; keep it outside the
        # model-only interval just like the VLA-Adapter image processor.
        dino_image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        self._sync_cuda_if_needed()
        fast_start = time.perf_counter()
        fused_hidden = self._build_action_condition(
            self._cached_action_token_hidden,
            batch_images,
            dino_image_tensors=dino_image_tensors,
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_residual_actions = self.action_model.predict_action(fused_hidden)
        coarse_action = self._coarse_with_gripper_pad(self._cached_coarse_action, pred_residual_actions.shape[-1])
        pred_actions = pred_residual_actions + coarse_action[:, start:end, :]
        self._sync_cuda_if_needed()
        fast_time_s = time.perf_counter() - fast_start

        self._predict_call_count += 1
        if debug_twochunk:
            hidden_mode = "new_slow_action_hidden" if slow_refresh else "reuse_slow_action_hidden"
            print(
                "[TwoChunkFramework] "
                f"{hidden_mode}; reset_cache={reset_cache}; "
                f"fast_refresh_every={self.fast_chunk_size} env steps; "
                f"slow_refresh_every={self.language_refresh_steps} env steps; "
                f"predict_call_count={self._predict_call_count}; "
                f"slow_time={slow_time_s:.4f}s; "
                f"fast_time={fast_time_s:.4f}s; "
                f"fused_hidden_shape={tuple(fused_hidden.shape)}; "
                f"action_shape={tuple(pred_actions.shape)}"
            )
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
            "inference_timing": {
                "slow_refresh": slow_refresh,
                "slow_time_s": slow_time_s,
                "fast_time_s": fast_time_s,
                "model_inference_time_s": slow_time_s + fast_time_s,
                "timing_scope": "model_only_after_preprocess",
                "fast_chunk_size": self.fast_chunk_size,
                "language_refresh_steps": self.language_refresh_steps,
                "vision_refresh_steps": self.vision_refresh_steps,
            },
        }

    def _valid_prediction_refreshes(self, image_sequences: List[List], examples: List[dict]) -> int:
        max_refreshes = max(1, min(self.twochunk_window_size, self.motion_dct_chunk_len) // max(self.vision_refresh_steps, 1))
        num_refreshes = min(min(len(seq) for seq in image_sequences), max_refreshes)
        if examples and "action" in examples[0]:
            action_len = min(np.asarray(example["action"]).shape[0] for example in examples)
            while num_refreshes > 0:
                end = (num_refreshes - 1) * self.vision_refresh_steps + self.fast_chunk_size
                if end <= min(action_len, self.motion_dct_chunk_len):
                    return num_refreshes
                num_refreshes -= 1
            return 0
        return num_refreshes

    @torch.inference_mode()
    def _predict_action_window(
        self,
        examples: List[dict],
        **kwargs: str,
    ):
        image_sequences = [self._to_pil_nested(example["image_sequence"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            image_sequences = resize_images(image_sequences, target_size=train_obs_image_size)

        num_refreshes = self._valid_prediction_refreshes(image_sequences, examples)
        if num_refreshes == 0:
            raise ValueError(
                "TwoChunk predict_action received no valid action windows. "
                f"fast_chunk_size={self.fast_chunk_size}, vision_refresh_steps={self.vision_refresh_steps}, "
                f"motion_dct_chunk_len={self.motion_dct_chunk_len}"
            )

        first_frame_images = self._first_refresh_images(image_sequences)
        action_token_hidden = self._encode_action_token_hidden(first_frame_images, instructions)
        coarse_long_action = self._coarse_with_gripper_pad(
            self._predict_coarse_action(action_token_hidden, self.motion_dct_chunk_len),
            action_dim=int(self.config.framework.action_model.action_dim),
        )

        flat_frame_images = []
        for refresh_i in range(num_refreshes):
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])

        flat_action_token_hidden = action_token_hidden.repeat(num_refreshes, 1, 1)
        fused_hidden = self._build_action_condition(flat_action_token_hidden, flat_frame_images)
        pred_residual_actions = self.action_model.predict_action(fused_hidden)

        batch_size = len(examples)
        pred_residual_actions = pred_residual_actions.view(num_refreshes, batch_size, self.fast_chunk_size, -1)
        pred_residual_actions = pred_residual_actions.permute(1, 0, 2, 3)

        coarse_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunks.append(coarse_long_action[:, start:end, :])
        coarse_chunks = torch.stack(coarse_chunks, dim=1)

        pred_actions = pred_residual_actions + coarse_chunks
        pred_actions = pred_actions.reshape(batch_size, num_refreshes * self.fast_chunk_size, -1)
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
