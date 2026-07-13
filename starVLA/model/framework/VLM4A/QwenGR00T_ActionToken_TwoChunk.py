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
        vla_data_cfg = self.config.get("datasets", {}).get("vla_data", {})
        self.include_state_condition = _as_bool(vla_data_cfg.get("include_state", False))
        self.state_dim = int(self.config.framework.action_model.get("state_dim", 0))

        self.language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 64))
        self.vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", self.action_horizon))
        self.fast_chunk_size = self.vision_refresh_steps
        self.twochunk_window_size = int(qwenvl_cfg.get("twochunk_window_size", self.language_refresh_steps))

        self.motion_dct_keep_freq = int(qwenvl_cfg.get("motion_dct_keep_freq", 8))
        self.motion_dct_chunk_len = int(qwenvl_cfg.get("motion_dct_chunk_len", self.language_refresh_steps))
        self.use_motion_dct_loss = _as_bool(qwenvl_cfg.get("use_motion_dct_loss", True))
        self.motion_dct_loss_weight = float(qwenvl_cfg.get("motion_dct_loss_weight", 1.0))
        self.detach_idct_condition = _as_bool(qwenvl_cfg.get("detach_idct_condition", True))
        self.action_condition_token_count = int(qwenvl_cfg.get("action_condition_token_count", 0))
        if self.action_condition_token_count < 0:
            raise ValueError(
                "framework.qwenvl.action_condition_token_count must be >= 0, "
                f"got {self.action_condition_token_count}."
            )
        self.layerwise_vlm_condition = _as_bool(qwenvl_cfg.get("layerwise_vlm_condition", False))

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
        self.action_condition_query_token = None
        if self.action_condition_token_count > 0:
            self.action_condition_query_token = nn.Parameter(
                torch.randn(1, self.action_condition_token_count, hidden_size) * 0.02
            )
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
        self.state_condition_proj = None
        self.state_condition_type_embedding = None
        if self.include_state_condition:
            if self.state_dim <= 0:
                raise ValueError("include_state=true requires framework.action_model.state_dim > 0.")
            self.state_condition_proj = nn.Linear(self.state_dim, hidden_size, bias=False)
            self.state_condition_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.action_model = self._build_fast_action_head(
            hidden_size=hidden_size,
            action_dim=action_dim,
        )
        action_cfg = self.config.framework.get("action_model", {})
        self.coarse_condition_query = bool(getattr(self.action_model, "coarse_condition_query", False))
        self.coarse_action_side_tokens = bool(getattr(self.action_model, "coarse_action_side_tokens", False))
        self.coarse_actions_for_action_head = self.coarse_condition_query or self.coarse_action_side_tokens
        self.coarse_condition_tokens = _as_bool(action_cfg.get("coarse_condition_tokens", False))
        if self.coarse_condition_tokens:
            self.coarse_condition_proj = nn.Linear(action_dim, hidden_size, bias=False)
            self.coarse_condition_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
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
            default_condition_group_names = ["action_token", "coarse_idct", "memory", "dino"]
            if getattr(self, "action_condition_token_count", 0) > 0:
                default_condition_group_names.insert(1, "action_condition_token")
            if getattr(self, "include_state_condition", False):
                default_condition_group_names.append("state")
            return GatedAttentionActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                NUM_ACTIONS_CHUNK=self.fast_chunk_size,
                num_blocks=int(action_cfg.get("gated_num_blocks", 8)),
                num_heads=int(action_cfg.get("gated_num_heads", 8)),
                use_rope=_as_bool(action_cfg.get("gated_use_rope", True)),
                adapter_token_count=int(action_cfg.get("gated_adapter_token_count", self.motion_dct_keep_freq)),
                coarse_condition_query=_as_bool(action_cfg.get("coarse_condition_query", False)),
                separate_condition_paths=_as_bool(action_cfg.get("separate_condition_paths", False)),
                condition_group_names=action_cfg.get(
                    "condition_group_names",
                    default_condition_group_names,
                ),
                zero_init_output=_as_bool(action_cfg.get("zero_init_output", False)),
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

    def _append_action_query(self, qwen_inputs: dict) -> dict:
        model = self.qwen_vl_interface.model
        input_ids = qwen_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        dct_query = self.action_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        dct_query = dct_query.expand(inputs_embeds.shape[0], -1, -1)
        query_parts = [dct_query]
        if self.action_condition_query_token is not None:
            condition_query = self.action_condition_query_token.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            query_parts.append(condition_query.expand(inputs_embeds.shape[0], -1, -1))
        query = torch.cat(query_parts, dim=1)
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

    def _final_slow_token_hidden(self, slow_token_hidden: torch.Tensor) -> torch.Tensor:
        if slow_token_hidden.ndim == 4:
            return slow_token_hidden[-1]
        return slow_token_hidden

    def _dct_action_token_hidden(self, slow_token_hidden: torch.Tensor) -> torch.Tensor:
        slow_token_hidden = self._final_slow_token_hidden(slow_token_hidden)
        return slow_token_hidden[:, : self.motion_dct_keep_freq, :]

    def _extra_action_condition_hidden(self, slow_token_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.action_condition_token_count <= 0:
            return None
        slow_token_hidden = self._final_slow_token_hidden(slow_token_hidden)
        start = self.motion_dct_keep_freq
        end = start + self.action_condition_token_count
        return slow_token_hidden[:, start:end, :]

    def _repeat_slow_token_hidden(self, slow_token_hidden: torch.Tensor, repeats: int) -> torch.Tensor:
        if slow_token_hidden.ndim == 4:
            return slow_token_hidden.repeat(1, repeats, 1, 1)
        return slow_token_hidden.repeat(repeats, 1, 1)

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
        return self.motion_dct_head(self._dct_action_token_hidden(action_token_hidden))

    def _compute_motion_dct_loss(
        self,
        action_token_hidden: torch.Tensor,
        actions: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor:
        return super()._compute_motion_dct_loss(
            self._dct_action_token_hidden(action_token_hidden),
            actions,
            chunk_len,
        )

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
        total_slow_tokens = self.motion_dct_keep_freq + self.action_condition_token_count
        captured_token_hidden = []
        hook_handles = []
        if self.layerwise_vlm_condition:
            num_layers = len(getattr(self.action_model, "blocks", ()))
            if num_layers <= 0:
                raise ValueError("Layerwise VLM conditioning requires at least one action-head block.")
            text_model = getattr(self.qwen_vl_interface.model.model, "language_model", None)
            if text_model is None:
                text_model = self.qwen_vl_interface.model.model
            vlm_layers = getattr(text_model, "layers", None)
            if vlm_layers is None:
                raise AttributeError("Layerwise VLM conditioning requires text decoder layers to be exposed as .layers.")
            if num_layers > len(vlm_layers):
                raise ValueError(
                    f"Action head has {num_layers} blocks, but the VLM only has {len(vlm_layers)} text layers."
                )

            def capture_action_tokens(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                # Materialize only the learnable-token positions so the hook does
                # not retain the full sequence output through a view.
                captured_token_hidden.append(hidden[:, -total_slow_tokens:, :].clone())

            for layer in vlm_layers[-num_layers:]:
                hook_handles.append(layer.register_forward_hook(capture_action_tokens))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ActionToken only consumes the multimodal backbone hidden state.
            # Calling ForConditionalGeneration would additionally project every
            # sequence token through the large vocabulary lm_head, even though
            # those logits are discarded.
            try:
                qwenvl_outputs = self.qwen_vl_interface.model.model(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
            if self.layerwise_vlm_condition:
                if len(captured_token_hidden) != num_layers:
                    raise RuntimeError(
                        f"Expected {num_layers} captured VLM token layers, got {len(captured_token_hidden)}."
                    )
                captured_token_hidden[-1] = qwenvl_outputs.last_hidden_state[:, -total_slow_tokens:, :].clone()
                return torch.stack(captured_token_hidden, dim=0)
            return qwenvl_outputs.last_hidden_state[:, -total_slow_tokens:, :]

    def _clear_temporary_action_conditions(self) -> None:
        self._last_action_condition_groups = None
        self._last_action_condition_layers = None
        self._last_action_condition_group_layers = None

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
        coarse_actions: torch.Tensor | None = None,
        state: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        final_action_token_hidden = self._final_slow_token_hidden(action_token_hidden)
        dino_hidden = self._encode_dino_hidden_states(
            batch_images=frame_images,
            dtype=final_action_token_hidden.dtype,
            image_tensors=dino_image_tensors,
        ).to(device=final_action_token_hidden.device)
        condition_parts = []
        attention_spans = {}
        offset = 0

        dct_action_token_hidden = self._dct_action_token_hidden(action_token_hidden)
        condition_parts.append(dct_action_token_hidden)
        length = int(dct_action_token_hidden.shape[1])
        attention_spans["action_token"] = (offset, offset + length)
        offset += length

        extra_action_condition_hidden = self._extra_action_condition_hidden(action_token_hidden)
        if extra_action_condition_hidden is not None:
            condition_parts.append(extra_action_condition_hidden)
            length = int(extra_action_condition_hidden.shape[1])
            attention_spans["action_condition_token"] = (offset, offset + length)
            offset += length
        else:
            attention_spans["action_condition_token"] = (offset, offset)

        coarse_tokens = self._project_coarse_condition_tokens(
            coarse_actions,
            dtype=final_action_token_hidden.dtype,
        )
        if coarse_tokens is not None:
            condition_parts.append(coarse_tokens)
            length = int(coarse_tokens.shape[1])
            attention_spans["coarse_idct"] = (offset, offset + length)
            offset += length
        else:
            attention_spans["coarse_idct"] = (offset, offset)

        attention_spans["memory"] = (offset, offset)

        condition_parts.append(dino_hidden)
        length = int(dino_hidden.shape[1])
        attention_spans["dino"] = (offset, offset + length)
        offset += length

        state_tokens = self._project_state_condition_tokens(
            state,
            batch_size=final_action_token_hidden.shape[0],
            device=final_action_token_hidden.device,
            dtype=final_action_token_hidden.dtype,
        )
        if state_tokens is not None:
            condition_parts.append(state_tokens)
            length = int(state_tokens.shape[1])
            attention_spans["state"] = (offset, offset + length)
            offset += length
        else:
            attention_spans["state"] = (offset, offset)

        self._last_attention_debug_spans = attention_spans
        self._last_action_condition_groups = {
            "action_token": dct_action_token_hidden,
            "dino": dino_hidden,
        }
        if extra_action_condition_hidden is not None:
            self._last_action_condition_groups["action_condition_token"] = extra_action_condition_hidden
        if coarse_tokens is not None:
            self._last_action_condition_groups["coarse_idct"] = coarse_tokens
        if state_tokens is not None:
            self._last_action_condition_groups["state"] = state_tokens
        self._last_action_condition_group_layers = None
        self._last_action_condition_layers = None
        if action_token_hidden.ndim == 4:
            layer_groups = []
            use_group_paths = bool(getattr(self.action_model, "separate_condition_paths", False))
            layer_conditions = [] if not use_group_paths else None
            for layer_hidden in action_token_hidden:
                layer_dct_tokens = layer_hidden[:, : self.motion_dct_keep_freq, :]
                layer_parts = [layer_dct_tokens] if layer_conditions is not None else None
                groups = {
                    "action_token": layer_dct_tokens,
                    "dino": dino_hidden,
                }
                if self.action_condition_token_count > 0:
                    start = self.motion_dct_keep_freq
                    end = start + self.action_condition_token_count
                    layer_condition_tokens = layer_hidden[:, start:end, :]
                    if layer_parts is not None:
                        layer_parts.append(layer_condition_tokens)
                    groups["action_condition_token"] = layer_condition_tokens
                if coarse_tokens is not None:
                    if layer_parts is not None:
                        layer_parts.append(coarse_tokens)
                    groups["coarse_idct"] = coarse_tokens
                if layer_parts is not None:
                    layer_parts.append(dino_hidden)
                if state_tokens is not None:
                    if layer_parts is not None:
                        layer_parts.append(state_tokens)
                    groups["state"] = state_tokens
                if layer_conditions is not None:
                    layer_conditions.append(torch.cat(layer_parts, dim=1))
                layer_groups.append(groups)
            self._last_action_condition_layers = layer_conditions
            self._last_action_condition_group_layers = layer_groups
        condition_hidden = torch.cat(condition_parts, dim=1)
        return condition_hidden

    def _project_state_condition_tokens(
        self,
        state: torch.Tensor | np.ndarray | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.include_state_condition:
            return None
        if state is None:
            raise ValueError("include_state=true requires examples to contain `state`.")
        if self.state_condition_proj is None or self.state_condition_type_embedding is None:
            raise RuntimeError("State condition projection is not initialized.")
        state_tensor = torch.as_tensor(state, device=device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.view(1, 1, -1)
        elif state_tensor.ndim == 2:
            if state_tensor.shape[0] == batch_size:
                state_tensor = state_tensor[:, None, :]
            elif state_tensor.shape[0] == 1 and batch_size > 1:
                state_tensor = state_tensor.expand(batch_size, -1)[:, None, :]
            else:
                state_tensor = state_tensor.reshape(batch_size, 1, -1)
        elif state_tensor.ndim == 3:
            if state_tensor.shape[1] != 1:
                state_tensor = state_tensor[:, :1, :]
        else:
            raise ValueError(f"Expected state shape [D], [B,D], or [B,1,D], got {tuple(state_tensor.shape)}.")
        if state_tensor.shape[0] != batch_size:
            raise ValueError(f"State batch size {state_tensor.shape[0]} does not match condition batch size {batch_size}.")
        if state_tensor.shape[-1] != self.state_dim:
            raise ValueError(f"State dim {state_tensor.shape[-1]} does not match framework.action_model.state_dim={self.state_dim}.")
        tokens = self.state_condition_proj(
            state_tensor.to(
                device=self.state_condition_proj.weight.device,
                dtype=self.state_condition_proj.weight.dtype,
            )
        ).to(device=device, dtype=dtype)
        type_embedding = self.state_condition_type_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + type_embedding

    def _project_coarse_condition_tokens(
        self,
        coarse_actions: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.coarse_condition_tokens:
            return None
        if coarse_actions is None:
            raise ValueError("coarse_condition_tokens=true requires coarse_actions.")
        tokens = self.coarse_condition_proj(
            coarse_actions.to(
                device=self.coarse_condition_proj.weight.device,
                dtype=self.coarse_condition_proj.weight.dtype,
            )
        ).to(device=coarse_actions.device, dtype=dtype)
        type_embedding = self.coarse_condition_type_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + type_embedding

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

        image_sequences = [example["image_sequence"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        actions = torch.tensor(
            np.array(actions),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
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
        action_token_hidden = self._encode_action_token_hidden(
            qwen_first_frame_images,
            instructions,
            qwen_inputs=qwen_inputs,
        )

        long_chunk_len = min(self.motion_dct_chunk_len, actions.shape[1])
        motion_dct_loss = self._compute_motion_dct_loss(action_token_hidden, actions, long_chunk_len)
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss
        action_loss_weight = self._action_loss_weight(kwargs.get("train_step", None))

        coarse_long_action = self._predict_coarse_action(action_token_hidden, long_chunk_len)
        if self.detach_idct_condition:
            coarse_long_action = coarse_long_action.detach()
        coarse_long_action = self._coarse_with_gripper_pad(coarse_long_action, action_dim=actions.shape[-1])

        flat_frame_images = []
        residual_action_targets = []
        coarse_action_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunk = coarse_long_action[:, start:end, :]
            flat_frame_images.extend([image_sequence[refresh_i] for image_sequence in image_sequences])
            residual_action_targets.append(actions[:, start:end, :] - coarse_chunk)
            coarse_action_chunks.append(coarse_chunk)

        flat_action_token_hidden = self._repeat_slow_token_hidden(action_token_hidden, num_refreshes)
        flat_coarse_actions = torch.cat(coarse_action_chunks, dim=0).to(
            device=flat_action_token_hidden.device,
            dtype=flat_action_token_hidden.dtype,
        )
        dino_image_tensors = self.dino_encoder.prepare_dino_input(flat_frame_images)
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            dino_image_tensors=dino_image_tensors,
            coarse_actions=flat_coarse_actions,
        )
        residual_action_targets = torch.cat(residual_action_targets, dim=0).to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )
        flat_coarse_actions = flat_coarse_actions.to(
            device=fused_hidden.device,
            dtype=fused_hidden.dtype,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_residual_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=flat_coarse_actions if self.coarse_actions_for_action_head else None,
                condition_groups=getattr(self, "_last_action_condition_groups", None),
                condition_layers=getattr(self, "_last_action_condition_layers", None),
                condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
                attention_debug_spans=None,
            )
            self._clear_temporary_action_conditions()
            action_loss = self.l1_loss(pred_residual_actions, residual_action_targets)

        total_loss = action_loss_weight * action_loss + weighted_motion_dct_loss
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
        state = None
        if self.include_state_condition:
            states = [example.get("state") for example in examples]
            if any(item is None for item in states):
                raise ValueError("include_state=true requires every inference example to contain `state`.")
            state = torch.tensor(
                np.array(states),
                device=self.action_query_token.device,
                dtype=self.action_query_token.dtype,
            )
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0, :]
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
        action_dim = int(self.config.framework.action_model.action_dim)
        coarse_action = self._coarse_with_gripper_pad(self._cached_coarse_action, action_dim)
        coarse_chunk = coarse_action[:, start:end, :]
        fused_hidden = self._build_action_condition(
            self._cached_action_token_hidden,
            batch_images,
            dino_image_tensors=dino_image_tensors,
            coarse_actions=coarse_chunk,
            state=state,
        )
        attention_debug = bool(kwargs.get("attention_debug", False))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_residual_actions = self.action_model.predict_action(
                fused_hidden,
                coarse_actions=coarse_chunk if self.coarse_actions_for_action_head else None,
                condition_groups=getattr(self, "_last_action_condition_groups", None),
                condition_layers=getattr(self, "_last_action_condition_layers", None),
                condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
                attention_debug_spans=self._last_attention_debug_spans if attention_debug else None,
            )
        pred_actions = pred_residual_actions + coarse_chunk
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
        result = {
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
        if attention_debug:
            result["attention_debug"] = {
                "layers": getattr(self.action_model, "last_attention_debug", None),
                "spans": getattr(self, "_last_attention_debug_spans", None),
            }
        return result

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

        flat_action_token_hidden = self._repeat_slow_token_hidden(action_token_hidden, num_refreshes)
        batch_size = len(examples)
        coarse_chunks = []
        for refresh_i in range(num_refreshes):
            start = refresh_i * self.vision_refresh_steps
            end = start + self.fast_chunk_size
            coarse_chunks.append(coarse_long_action[:, start:end, :])
        coarse_chunks = torch.stack(coarse_chunks, dim=1)
        flat_coarse_actions = coarse_chunks.permute(1, 0, 2, 3).reshape(
            num_refreshes * batch_size,
            self.fast_chunk_size,
            -1,
        ).to(device=flat_action_token_hidden.device, dtype=flat_action_token_hidden.dtype)
        fused_hidden = self._build_action_condition(
            flat_action_token_hidden,
            flat_frame_images,
            coarse_actions=flat_coarse_actions,
        )
        flat_coarse_actions = flat_coarse_actions.to(device=fused_hidden.device, dtype=fused_hidden.dtype)
        pred_residual_actions = self.action_model.predict_action(
            fused_hidden,
            coarse_actions=flat_coarse_actions if self.coarse_actions_for_action_head else None,
            condition_groups=getattr(self, "_last_action_condition_groups", None),
            condition_layers=getattr(self, "_last_action_condition_layers", None),
            condition_group_layers=getattr(self, "_last_action_condition_group_layers", None),
            attention_debug_spans=None,
        )

        pred_residual_actions = pred_residual_actions.view(num_refreshes, batch_size, self.fast_chunk_size, -1)
        pred_residual_actions = pred_residual_actions.permute(1, 0, 2, 3)

        pred_actions = pred_residual_actions + coarse_chunks
        pred_actions = pred_actions.reshape(batch_size, num_refreshes * self.fast_chunk_size, -1)
        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
