"""TwoChunk variants that expose IDCT coarse actions on the action-query side.

This is architecture option 2: keep the learned action query unchanged, append
projected coarse action tokens to the action-side self-attention sequence, and
compute loss only from the learned query outputs.
"""

import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import _as_bool
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk_DCTMemory import (
    Qwen_GR00T_ActionToken_TwoChunk_DCTMemory,
)
from starVLA.model.modules.action_model.GatedAttentionActionHeader import GatedAttentionActionHead
from starVLA.model.modules.action_model.MLP_ActionHeader_TwoChunk import L1RegressionActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _force_action_side_coarse_config(config) -> None:
    action_cfg = config.framework.get("action_model", {})
    action_cfg["coarse_condition_query"] = False
    action_cfg["coarse_condition_tokens"] = False
    action_cfg["coarse_action_side_tokens"] = True


class _ActionSideCoarseMixin:
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
                coarse_condition_query=False,
                coarse_action_side_tokens=_as_bool(action_cfg.get("coarse_action_side_tokens", True)),
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


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_ActionSideCoarse")
class Qwen_GR00T_ActionToken_TwoChunk_ActionSideCoarse(
    _ActionSideCoarseMixin,
    Qwen_GR00T_ActionToken_TwoChunk,
):
    """TwoChunk option 2: coarse tokens join action-side self attention."""

    def __init__(self, config=None, **kwargs) -> None:
        _force_action_side_coarse_config(config)
        super().__init__(config=config, **kwargs)


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_DCTMemory_ActionSideCoarse")
class Qwen_GR00T_ActionToken_TwoChunk_DCTMemory_ActionSideCoarse(
    _ActionSideCoarseMixin,
    Qwen_GR00T_ActionToken_TwoChunk_DCTMemory,
):
    """DCTMemory option 2: memory stays in condition, coarse joins action side."""

    def __init__(self, config=None, **kwargs) -> None:
        _force_action_side_coarse_config(config)
        super().__init__(config=config, **kwargs)
