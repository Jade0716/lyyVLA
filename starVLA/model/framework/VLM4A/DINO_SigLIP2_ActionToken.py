import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct
from transformers import AutoModel, AutoProcessor
from transformers.masking_utils import create_bidirectional_mask

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import _as_bool
from starVLA.model.modules.action_model.MLP_ActionHeader_TwoChunk import (
    L1RegressionActionHead,
)
from starVLA.model.modules.dino_model.dinov3 import get_twochunk_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("DINO_SigLIP2_ActionToken")
class DINO_SigLIP2_ActionToken(baseframework):
    """Single-chunk ablation: DINO + SigLIP2-V/T motion tokens.

    This baseline is intended to answer whether a high-frequency policy built
    from conventional visual-language encoders can replace the slow VLM branch:

    - append 4 learnable action query tokens to SigLIP-T text embeddings;
    - fuse those text action tokens with SigLIP-V and DINO tokens via a small
      cross-attention decoder;
    - supervise the 4 fused motion tokens with low-frequency DCT coefficients;
    - predict one action chunk from [motion tokens + DINO tokens + SigLIP-V tokens].
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = config

        action_cfg = self.config.framework.action_model
        siglip_cfg = self.config.framework.get("siglip", {})
        dino_cfg = self.config.framework.get("dino", {})
        dct_cfg = self.config.framework.get("qwenvl", {})

        self.action_dim = int(action_cfg.action_dim)
        self.action_horizon = int(action_cfg.action_horizon)
        self.hidden_size = int(action_cfg.get("hidden_size", 1024))
        self.motion_dct_keep_freq = int(dct_cfg.get("motion_dct_keep_freq", 4))
        self.motion_dct_chunk_len = int(dct_cfg.get("motion_dct_chunk_len", self.action_horizon))
        self.use_motion_dct_loss = _as_bool(dct_cfg.get("use_motion_dct_loss", True))
        self.motion_dct_loss_weight = float(dct_cfg.get("motion_dct_loss_weight", 1.0))
        include_gripper = _as_bool(dct_cfg.get("motion_dct_include_gripper", True))
        self.motion_dct_action_dim = self.action_dim if include_gripper else max(self.action_dim - 1, 1)

        self.siglip_model_path = siglip_cfg.get(
            "model_path",
            "/home/liuyuyan/siglip2/google/siglip2-base-patch16-224",
        )
        self.siglip_max_text_length = int(siglip_cfg.get("max_text_length", 64))
        self.siglip_model = AutoModel.from_pretrained(
            self.siglip_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_path)

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

        siglip_text_hidden = int(self.siglip_model.config.text_config.hidden_size)
        siglip_vision_hidden = int(self.siglip_model.config.vision_config.hidden_size)
        self.action_query_token = nn.Parameter(
            torch.randn(1, self.motion_dct_keep_freq, siglip_text_hidden) * 0.02
        )
        self.text_action_proj = nn.Linear(siglip_text_hidden, self.hidden_size)
        self.siglip_vision_proj = nn.Linear(siglip_vision_hidden, self.hidden_size)
        self.dino_pro = nn.Linear(self.dino_encoder.num_channels, self.hidden_size)

        fusion_layers = int(siglip_cfg.get("fusion_layers", 2))
        fusion_heads = int(siglip_cfg.get("fusion_heads", 8))
        fusion_ffn_dim = int(siglip_cfg.get("fusion_ffn_dim", self.hidden_size * 4))
        self.motion_fusion = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.hidden_size,
                    nhead=fusion_heads,
                    dim_feedforward=fusion_ffn_dim,
                    dropout=float(siglip_cfg.get("fusion_dropout", 0.1)),
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(fusion_layers)
            ]
        )
        self.motion_norm = nn.LayerNorm(self.hidden_size)

        self.motion_dct_head = None
        if self.use_motion_dct_loss:
            self.motion_dct_head = L1RegressionActionHead(
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                action_dim=self.motion_dct_action_dim,
                NUM_ACTIONS_CHUNK=self.motion_dct_keep_freq,
            )
        self.action_model = L1RegressionActionHead(
            input_dim=self.hidden_size,
            hidden_dim=int(action_cfg.hidden_size),
            action_dim=self.action_dim,
            NUM_ACTIONS_CHUNK=self.action_horizon,
        )
        self.l1_loss = nn.L1Loss()

    @staticmethod
    def _sync_cuda_if_needed() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _siglip_text_embedding_layer(self):
        text_model = self.siglip_model.text_model
        if hasattr(text_model, "get_input_embeddings"):
            return text_model.get_input_embeddings()
        if hasattr(text_model, "embeddings") and hasattr(text_model.embeddings, "token_embedding"):
            return text_model.embeddings.token_embedding
        raise AttributeError("Cannot find SigLIP text token embedding layer.")

    def _encode_siglip_text_action_tokens(self, instructions: List[str]) -> torch.Tensor:
        tokenizer = self.siglip_processor.tokenizer
        text_max_len = max(1, self.siglip_max_text_length - self.motion_dct_keep_freq)
        text_inputs = tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=text_max_len,
            return_tensors="pt",
        )
        device = next(self.siglip_model.parameters()).device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        embed_layer = self._siglip_text_embedding_layer()
        inputs_embeds = embed_layer(text_inputs["input_ids"])
        query = self.action_query_token.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        query = query.expand(inputs_embeds.shape[0], -1, -1)
        inputs_embeds = torch.cat([inputs_embeds, query], dim=1)

        attention_mask = text_inputs.get("attention_mask", None)
        if attention_mask is not None:
            query_mask = torch.ones(
                attention_mask.shape[0],
                self.motion_dct_keep_freq,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, query_mask], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = self.siglip_model.text_model.embeddings(
                inputs_embeds=inputs_embeds,
            )
            attention_mask = create_bidirectional_mask(
                config=self.siglip_model.text_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
            encoder_outputs = self.siglip_model.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
            text_hidden = self.siglip_model.text_model.final_layer_norm(
                encoder_outputs.last_hidden_state
            )
        text_action = text_hidden[:, -self.motion_dct_keep_freq :, :]
        return self.text_action_proj(text_action)

    def _flatten_views(self, batch_images: List[List]) -> tuple[list, list[int]]:
        flat_images = []
        view_counts = []
        for views in batch_images:
            view_counts.append(len(views))
            flat_images.extend(views)
        return flat_images, view_counts

    def _encode_siglip_vision_hidden(self, batch_images: List[List]) -> torch.Tensor:
        flat_images, view_counts = self._flatten_views(batch_images)
        image_inputs = self.siglip_processor.image_processor(
            images=flat_images,
            return_tensors="pt",
        )
        device = next(self.siglip_model.parameters()).device
        image_inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in image_inputs.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.siglip_model.vision_model(
                **image_inputs,
                return_dict=True,
            )
        hidden = self.siglip_vision_proj(outputs.last_hidden_state)
        per_sample = []
        offset = 0
        for count in view_counts:
            per_sample.append(hidden[offset : offset + count].reshape(1, -1, hidden.shape[-1]))
            offset += count
        return torch.cat(per_sample, dim=0)

    def _encode_dino_hidden(self, batch_images: List[List], dtype: torch.dtype) -> torch.Tensor:
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        batch_size = len(batch_images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dino_features = self.dino_encoder(image_tensors)
            dino_features = dino_features.reshape(batch_size, -1, dino_features.shape[-1])
            dino_hidden = self.dino_pro(dino_features)
        return dino_hidden.to(dtype=dtype)

    def _encode_condition(self, batch_images: List[List], instructions: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_action_tokens = self._encode_siglip_text_action_tokens(instructions)
        dino_hidden = self._encode_dino_hidden(batch_images, dtype=text_action_tokens.dtype).to(
            device=text_action_tokens.device
        )
        siglip_vision_hidden = self._encode_siglip_vision_hidden(batch_images).to(
            device=text_action_tokens.device,
            dtype=text_action_tokens.dtype,
        )

        memory = torch.cat([siglip_vision_hidden, dino_hidden], dim=1)
        motion_tokens = text_action_tokens
        for layer in self.motion_fusion:
            motion_tokens = layer(tgt=motion_tokens, memory=memory)
        motion_tokens = self.motion_norm(motion_tokens)
        condition_hidden = torch.cat([motion_tokens, dino_hidden, siglip_vision_hidden], dim=1)
        return motion_tokens, condition_hidden

    def _motion_dct_target(self, actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
        action_chunk = actions[:, :chunk_len, : self.motion_dct_action_dim].detach().float().cpu().numpy()
        low_dct_gt = dct(action_chunk, type=2, axis=1, norm="ortho")[:, : self.motion_dct_keep_freq, :]
        if low_dct_gt.shape[1] < self.motion_dct_keep_freq:
            pad_len = self.motion_dct_keep_freq - low_dct_gt.shape[1]
            low_dct_gt = np.pad(low_dct_gt, ((0, 0), (0, pad_len), (0, 0)))
        return torch.from_numpy(low_dct_gt).to(device=actions.device, dtype=actions.dtype)

    def _compute_motion_dct_loss(self, motion_tokens: torch.Tensor, actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
        if not self.use_motion_dct_loss:
            return motion_tokens.new_zeros(())
        low_dct_gt = self._motion_dct_target(actions, chunk_len).to(
            device=motion_tokens.device,
            dtype=motion_tokens.dtype,
        )
        low_dct_pred = self.motion_dct_head(motion_tokens)
        return F.mse_loss(low_dct_pred.float(), low_dct_gt.float())

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=self.action_query_token.device,
            dtype=self.action_query_token.dtype,
        )
        if actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"DINO_SigLIP2_ActionToken requires at least {self.action_horizon} actions, "
                f"got {actions.shape[1]}."
            )
        actions_target = actions[:, : self.action_horizon, :]

        motion_tokens, condition_hidden = self._encode_condition(batch_images, instructions)
        motion_dct_loss = self._compute_motion_dct_loss(
            motion_tokens,
            actions,
            min(self.motion_dct_chunk_len, actions.shape[1]),
        )
        weighted_motion_dct_loss = self.motion_dct_loss_weight * motion_dct_loss

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(condition_hidden)
            action_loss = self.l1_loss(pred_actions, actions_target.to(pred_actions.dtype))

        total_loss = action_loss + weighted_motion_dct_loss
        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "motion_dct_loss": motion_dct_loss,
            "weighted_motion_dct_loss": weighted_motion_dct_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        self._sync_cuda_if_needed()
        start = time.perf_counter()
        _, condition_hidden = self._encode_condition(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(condition_hidden)
        self._sync_cuda_if_needed()
        model_inference_time_s = time.perf_counter() - start

        return {
            "normalized_actions": pred_actions.detach().float().cpu().numpy(),
            "inference_timing": {
                "model_inference_time_s": model_inference_time_s,
                "timing_scope": "model_only_after_preprocess",
            },
        }
