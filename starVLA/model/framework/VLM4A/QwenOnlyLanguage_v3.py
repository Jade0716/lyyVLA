# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenOnlyLanguage_v3 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.
Only uses non-vision (language) tokens to update actions.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key features
─────────────────────────────
1. **Language-only token updates**
   Only non-vision tokens are used for action prediction - vision tokens are filtered out
   before feeding the Action DiT.

2. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).

3. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``).

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOnlyLanguage_v3
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOnlyLanguage_v3DefaultConfig:
    """QwenOnlyLanguage_v3 framework default parameters."""

    name: str = "QwenOnlyLanguage_v3"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOnlyLanguage_v3")
class Qwen_OnlyLanguage_v3(baseframework):
    """
    Multimodal vision-language-action model (QwenOnlyLanguage_v3 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - **Only non-vision (language) tokens** are used for action prediction.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    Focus: predict a future action chunk conditioned on a natural-language
    instruction (with optional discretised state prefix). Vision tokens are
    explicitly excluded from action updates.
    """

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
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenOnlyLanguage_v3DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head.
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _filter_vision_tokens(
        self,
        vl_embs_list: List[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Filter out vision tokens, keeping only language tokens.

        Returns:
            vl_embs_filtered: list of filtered hidden states
            attention_mask_filtered: filtered attention mask (or None if input mask was None)
        """
        # IMAGE_TOKEN_INDEX = 248056 for Qwen3.5-VL
        input_ids_np = input_ids[0].cpu().numpy() if input_ids.dim() == 2 else input_ids.cpu().numpy()
        vision_token_mask = input_ids_np == 248056
        non_vision_indices = np.where(~vision_token_mask)[0]

        vl_embs_filtered = []
        for layer_hidden in vl_embs_list:
            if layer_hidden.shape[1] == len(non_vision_indices) + vision_token_mask.sum():
                # Full sequence, need to filter
                layer_hidden_filtered = layer_hidden[:, non_vision_indices, :]
                vl_embs_filtered.append(layer_hidden_filtered)
            else:
                # Already filtered
                vl_embs_filtered.append(layer_hidden)

        # Also filter attention_mask to match the filtered sequence length
        attention_mask_filtered = None
        if attention_mask is not None:
            attn_mask_np = attention_mask.cpu().numpy()
            if attn_mask_np.ndim == 2 and attn_mask_np.shape[1] == len(non_vision_indices) + vision_token_mask.sum():
                attention_mask_filtered = torch.from_numpy(attn_mask_np[:, non_vision_indices]).to(
                    attention_mask.device, dtype=attention_mask.dtype
                )
            else:
                attention_mask_filtered = attention_mask

        return vl_embs_filtered, attention_mask_filtered

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        """Run QwenVL, filter vision tokens, project hidden states, and return (layer-wise embeddings, attention_mask) for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        input_ids = qwen_inputs.get("input_ids")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = list(qwenvl_outputs.hidden_states)

            # Filter out vision tokens - only keep language tokens
            if input_ids is not None:
                vl_embs_list, attention_mask_filtered = self._filter_vision_tokens(
                    all_hidden[-self.num_action_dit_layers:], input_ids, attention_mask
                )
            else:
                vl_embs_list = list(all_hidden[-self.num_action_dit_layers:])
                attention_mask_filtered = attention_mask

            # Project to action DiT hidden dimension
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)

        return vl_embs_list, attention_mask_filtered

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
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples]  # List[ndarray (T, action_dim)]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Prepend discretised proprioceptive state to each instruction string.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None  # state is now encoded in the instruction tokens

        # Step 1: encode through QwenVL (filter vision tokens)
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        # Step 2: compute flow-matching loss over the action chunk
        with torch.autocast("cuda", dtype=torch.float32):
            # Align labels: keep only the last action_horizon timesteps.
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon:, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 2
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat every VLM layer embedding to match the duplicated action batch.
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
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Filter vision tokens, project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None

        # Encode proprioceptive state into the instruction string, then discard raw state.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        # Optionally resize images to the resolution used during training.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL (filter vision tokens)
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 2: run the flow-matching sampler to produce the denoised action chunk.
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    args.config_yaml = "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = Qwen_OnlyLanguage_v3(cfg)
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")