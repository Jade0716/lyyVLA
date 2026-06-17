import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken import Qwen_GR00T_ActionToken
from starVLA.model.tools import FRAMEWORK_REGISTRY


class LinearMotionDCTHead(nn.Module):
    """
    Lightweight DCT probe for action-token hidden states.

    This intentionally keeps the auxiliary head weak: if the DCT loss improves,
    the DCT target must be close to linearly decodable from the action tokens
    instead of being fitted by a high-capacity MLP head.
    """

    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, action_dim)

    def forward(self, action_token_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(action_token_hidden))


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_v2")
class Qwen_GR00T_ActionToken_v2(Qwen_GR00T_ActionToken):
    """
    QwenGR00T ActionToken with a lightweight DCT head.

    It keeps the original ActionToken training/inference path unchanged, but
    replaces the heavy MLPResNet DCT auxiliary head with LayerNorm + Linear.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if self.use_motion_dct_loss:
            hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
            self.motion_dct_head = LinearMotionDCTHead(
                input_dim=hidden_size,
                action_dim=self.motion_dct_action_dim,
            )
