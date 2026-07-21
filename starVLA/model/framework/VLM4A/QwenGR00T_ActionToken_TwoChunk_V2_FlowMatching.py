"""TwoChunk V2 with QwenPI-v3's layer-wise flow-matching action head."""

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk_V2 import (
    Qwen_GR00T_ActionToken_TwoChunk_V2,
)
from starVLA.model.framework.share_tools import populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_V2_FlowMatching")
class Qwen_GR00T_ActionToken_TwoChunk_V2_FlowMatching(Qwen_GR00T_ActionToken_TwoChunk_V2):
    """Keep V2 conditioning and predict full actions through flow matching."""

    def _build_fast_action_head(self, hidden_size: int, action_dim: int) -> nn.Module:
        action_cfg = self.config.framework.get("action_model", {})
        head_type = str(action_cfg.get("action_head_type", "")).lower()
        if head_type in {"flow_matching", "flowmatching", "layerwise_flow_matching"}:
            # The TwoChunk base constructor builds its gated/MLP head before
            # this subclass can create the QwenPI-style layerwise flow head.
            # Keep a registered placeholder only for that initialization span;
            # __init__ replaces it immediately after super().__init__ returns.
            return nn.Identity()
        return super()._build_fast_action_head(hidden_size=hidden_size, action_dim=action_dim)

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        if not self.layerwise_vlm_condition:
            raise ValueError("V2 FlowMatching requires framework.qwenvl.layerwise_vlm_condition=true.")

        action_cfg = self.config.framework.action_model
        diffusion_cfg = action_cfg.diffusion_model_cfg
        num_flow_layers = int(action_cfg.get("flow_num_blocks", diffusion_cfg.get("num_layers", 16)))
        vl_hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        flow_hidden_dim = int(diffusion_cfg.get("action_dit_hidden_dim", action_cfg.get("hidden_size", 1024)))

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=flow_hidden_dim,
            num_dit_layers=num_flow_layers,
        )

        # V2 already encodes robot state as a cross-attention condition token.
        # Avoid constructing a second, unused state path inside the PI head.
        configured_state_dim = int(action_cfg.get("state_dim", 0))
        try:
            action_cfg.state_dim = 0
            self.action_model = get_action_model(config=self.config)
        finally:
            action_cfg.state_dim = configured_state_dim

        self.flow_condition_projectors = nn.ModuleList(
            [
                nn.Identity()
                if vl_hidden_dim == flow_hidden_dim
                else nn.Sequential(
                    nn.LayerNorm(vl_hidden_dim),
                    nn.Linear(vl_hidden_dim, flow_hidden_dim),
                )
                for _ in range(num_flow_layers)
            ]
        )
        self.num_flow_layers = num_flow_layers
        self.flow_hidden_dim = flow_hidden_dim
        self.coarse_actions_for_action_head = False

    def forward(self, examples=None, **kwargs):
        if examples and "image_sequence" not in examples[0]:
            raise ValueError("V2 FlowMatching requires TwoChunk examples with `image_sequence`.")
        return super().forward(examples=examples, **kwargs)

    def _project_flow_conditions(self) -> list[torch.Tensor]:
        condition_layers = getattr(self, "_last_action_condition_layers", None)
        if condition_layers is None:
            raise RuntimeError(
                "V2 FlowMatching requires layerwise_vlm_condition=true and per-layer action conditions."
            )
        if len(condition_layers) != len(self.flow_condition_projectors):
            raise ValueError(
                f"Got {len(condition_layers)} condition layers for "
                f"{len(self.flow_condition_projectors)} flow-matching blocks."
            )
        return [projector(hidden) for projector, hidden in zip(self.flow_condition_projectors, condition_layers)]

    def _repeat_flow_batch(
        self,
        condition_layers: list[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        repeats = int(self.config.trainer.get("repeated_diffusion_steps", 1))
        if repeats <= 1:
            return condition_layers, actions
        return (
            [hidden.repeat(repeats, 1, 1) for hidden in condition_layers],
            actions.repeat(repeats, 1, 1),
        )

    def _compute_fast_action_loss(
        self,
        fused_hidden: torch.Tensor,
        action_targets: torch.Tensor,
        coarse_actions: torch.Tensor,
    ) -> torch.Tensor:
        del fused_hidden, coarse_actions
        try:
            condition_layers = self._project_flow_conditions()
            condition_layers, action_targets = self._repeat_flow_batch(condition_layers, action_targets)
            return self.action_model(
                condition_layers,
                action_targets,
                state=None,
                encoder_attention_mask=None,
            )
        finally:
            self._clear_temporary_action_conditions()

    def _predict_fast_action(
        self,
        fused_hidden: torch.Tensor,
        coarse_actions: torch.Tensor,
        attention_debug_spans=None,
    ) -> torch.Tensor:
        del fused_hidden, coarse_actions, attention_debug_spans
        try:
            condition_layers = self._project_flow_conditions()
            return self.action_model.predict_action(
                condition_layers,
                state=None,
                encoder_attention_mask=None,
            )
        finally:
            self._clear_temporary_action_conditions()
