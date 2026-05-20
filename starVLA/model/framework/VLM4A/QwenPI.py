# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPIDefaultConfig:
    """QwenPI (QwenFM) framework default parameters.

    Layer-wise cross-DiT flow-matching action prediction conditioned on
    multi-layer VLM hidden states.  All fields can be overridden by the
    corresponding key in the YAML ``framework:`` section.
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenPI"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (auto-set at runtime from model config)
            "vl_hidden_dim": 2048,
            # Number of VL transformer layers (auto-set at runtime)
            "num_vl_layers": 36,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "LayerwiseFM",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            # Repeat factor for flow-matching loss
            "repeated_diffusion_steps": 2,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # DiT architecture settings — shape fields (num_layers,
            # input_embedding_dim, cross_attention_dim, num_attention_heads)
            # are auto-populated by populate_layerwise_dit_cfg at runtime.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFM")
@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model (PI variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Layer-wise cross-DiT diffusion head fed by multi-layer VLM hidden states

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    #
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenPIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors;
        # we keep the last num_hidden_layers of them, so DiT depth must match.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # QwenPI: DiT runs at the LLM hidden size (no compression).  Tell the
        # action head exactly that — the head itself does not look at qwenvl.*.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        """Run QwenVL and return (layer-wise hidden states, attention_mask) for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list, attention_mask

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat features for each layer
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(  # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to directly regress future actions (no diffusion sampling).

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_with_attention(
        self,
        examples: List[dict] = None,
        save_dir: str = "./attention_heatmaps",
        layer_idx: int = -1,
        attention_type: str = "action_model",  # "vlm" or "action_model"
        **kwargs,
    ) -> dict:
        """
        推理并生成注意力热力图（基于模型原生attention weights）。

        Args:
            examples: List[dict], each dict with 'image', 'lang', optional 'state'
            save_dir: 保存热力图的目录
            layer_idx: 使用哪一层的attention (default: -1 = 最后一层)
            attention_type: "action_model" (default) - action model对vision的cross-attention
                          "vlm" - VLM自己的vision-to-vision self-attention

        Returns:
            dict: normalized_actions + heatmap_paths (list of paths for all views)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        if type(examples) is not list:
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve
        from starVLA.model.visual_tools.attention_visualizer import (
            create_side_by_side,
            create_multi_layer_heatmap,
        )
        import time

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # 构建输入
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        # 获取所有原始图像用于热力图可视化
        num_images = len(examples[0]["image"]) if examples and "image" in examples[0] else 0

        # 生成保存路径 (每个视角一张)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        instruction_short = instructions[0][:30].replace(" ", "_") if instructions else "inst"
        save_paths = [os.path.join(save_dir, f"attn_{instruction_short}_view{i}_{timestamp}_{layer_idx}.png") for i in range(num_images)]

        # 计算注意力热力图
        heatmap_paths = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        state_t = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None

        # Get vision token indices from input_ids (IMAGE_TOKEN_INDEX = 248056 for Qwen3.5-VL)
        input_ids = qwen_inputs.get("input_ids")
        image_grid_thw = qwen_inputs.get("image_grid_thw")
        spatial_merge_size = int(self.qwen_vl_interface.model.visual.spatial_merge_size) if hasattr(self.qwen_vl_interface, 'model') and hasattr(self.qwen_vl_interface.model, 'visual') else 2

        if input_ids is not None:
            input_ids_np = input_ids[0].cpu().numpy()  # first sample in batch】
            vision_token_indices = np.where(input_ids_np == 248056)[0]  # IMAGE_TOKEN_INDEX for Qwen3.5-VL
            if len(vision_token_indices) > 0:
                vision_token_indices = torch.from_numpy(vision_token_indices).long().to(base_hidden.device)
            else:
                vision_token_indices = None
        else:
            vision_token_indices = None

        if attention_type == "action_model":
            # Compute action model's cross-attention to vision tokens
            try:
                pred_actions, all_layers_attention = self.action_model.predict_action_with_attention(
                    vl_embs_list,
                    state_t,
                    layer_idx=layer_idx,
                    vision_token_indices=vision_token_indices,
                )


                # all_layers_attention is dict: {layer_idx: {'first_view': tensor, 'second_view': tensor}}
                num_layers = len(all_layers_attention)
                if num_layers == 0:
                    heatmap_paths = None
                else:
                    if image_grid_thw is not None:
                        grid = image_grid_thw[0].cpu().numpy()
                        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
                        merged_h, merged_w = h // spatial_merge_size, w // spatial_merge_size
                    else:
                        merged_h, merged_w = 8, 8

                    for view_idx, view_name in enumerate(['first_view', 'second_view']):
                        if view_idx >= num_images:
                            continue

                        save_p = save_paths[view_idx] if view_idx < len(save_paths) else None
                        if save_p is None:
                            continue

                        if examples[0]["image"][view_idx] is not None:
                            original_img_view = np.array(examples[0]["image"][view_idx])
                        else:
                            continue

                        # Collect heatmaps for all layers
                        heatmaps_list = []
                        layer_names_list = []
                        for layer_i in range(num_layers):
                            view_attn = all_layers_attention[layer_i][view_name]
                            if view_attn is None or view_attn.shape[0] == 0:
                                heatmaps_list.append(np.zeros((merged_h, merged_w), dtype=np.float32))
                            else:
                                vis_attn_2d = view_attn.float().mean(axis=0).reshape(merged_h, merged_w)
                                if isinstance(vis_attn_2d, torch.Tensor):
                                    vis_attn_2d = vis_attn_2d.cpu().numpy()
                                heatmaps_list.append(vis_attn_2d)
                            layer_names_list.append(f"Layer {layer_i}")

                        # Use PIL/cv2 based function for fast rendering
                        create_multi_layer_heatmap(
                            image=original_img_view,
                            heatmaps=heatmaps_list,
                            layer_names=layer_names_list,
                            colormap='jet',
                            alpha=0.5,
                            n_cols=4,
                            save_path=save_p
                        )
                        print(f"Action model cross-attention for view {view_idx} (all {num_layers} layers) saved to {save_p}")

                    heatmap_paths = [save_paths[i] if i < len(save_paths) else None for i in range(2)]
            except Exception as e:
                print(f"Cross-attention computation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to normal prediction
                with torch.autocast("cuda", dtype=torch.float32):
                    pred_actions = self.action_model.predict_action(vl_embs_list, state_t)

        # elif attention_type == "vlm":
        #     # Compute VLM's vision-to-vision self-attention
        #     heatmap_paths = []
        #     if qwenvl_outputs.attentions is not None:
        #         try:
        #             heatmap_2d, _ = compute_attention_heatmap(
        #                 self.qwen_vl_interface,
        #                 qwen_inputs,
        #                 layer_idx=layer_idx,
        #             )
        #             # VLM attention is for the whole image, save for each view
        #             for i, save_p in enumerate(save_paths):
        #                 if i < num_images and examples[0]["image"][i] is not None:
        #                     original_img_view = np.array(examples[0]["image"][i])
        #                     create_side_by_side(original_img_view, heatmap_2d, save_path=save_p)
        #                     heatmap_paths.append(save_p)
        #                 else:
        #                     heatmap_paths.append(None)
        #         except Exception as e:
        #             print(f"Attention heatmap computation failed: {e}")
        #             import traceback
        #             traceback.print_exc()
        #             heatmap_paths = [None] * num_images
        #     # VLM attention doesn't use action_model cross-attention, so call predict_action
        #     with torch.autocast("cuda", dtype=torch.float32):
        #         pred_actions = self.action_model.predict_action(vl_embs_list, state_t)

        else:
            # Default behavior: just predict actions
            with torch.autocast("cuda", dtype=torch.float32):
                pred_actions = self.action_model.predict_action(vl_embs_list, state_t)

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions, "heatmap_paths": heatmap_paths}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwen_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
