"""Legacy TwoChunk implementation for evaluating pre-DINOv3 checkpoints."""

from starVLA.model.framework.VLM4A.QwenGR00T_ActionToken_TwoChunk import (
    Qwen_GR00T_ActionToken_TwoChunk,
)
from starVLA.model.modules.action_model.MLP_ActionHeader_TwoChunk import (
    L1RegressionActionHead,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_ActionToken_TwoChunk_back")
class Qwen_GR00T_ActionToken_TwoChunk_back(Qwen_GR00T_ActionToken_TwoChunk):
    """Restore the DINOv2 + six-dimensional DCT layout used by old runs."""

    def __init__(self, config=None, **kwargs) -> None:
        # Force the original visual backbone before the parent constructs it.
        dino_cfg = config.framework.get("dino", {})
        dino_cfg["dino_backbone"] = dino_cfg.get("dino_backbone", "dinov2_vits14")
        if not str(dino_cfg["dino_backbone"]).startswith("dinov2_"):
            dino_cfg["dino_backbone"] = "dinov2_vits14"

        super().__init__(config=config, **kwargs)

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        action_dim = int(self.config.framework.action_model.action_dim)

        # Legacy DCT supervised only motion dimensions and excluded gripper.
        self.motion_dct_action_dim = max(action_dim - 1, 1)
        if self.use_motion_dct_loss:
            self.motion_dct_head = L1RegressionActionHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                action_dim=self.motion_dct_action_dim,
                NUM_ACTIONS_CHUNK=self.motion_dct_keep_freq,
            )
