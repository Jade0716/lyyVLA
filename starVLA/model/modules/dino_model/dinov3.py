"""Local DINOv3 backbone wrapper for TwoChunk visual conditioning."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


DEFAULT_DINOV3_REPO = Path("/home/liuyuyan/dinov3")
DEFAULT_DINOV3_WEIGHTS = Path(
    "/mnt/8ac36469-5f21-42a9-a6dd-21bfcb724d52/liuyuyan/DINO/"
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)


class DINOv3BackBone(nn.Module):
    """DINOv3 ViT wrapper exposing normalized patch tokens."""

    def __init__(
        self,
        backbone_name: str = "dinov3_vits16plus",
        repo_path: str | Path = DEFAULT_DINOV3_REPO,
        weights_path: str | Path = DEFAULT_DINOV3_WEIGHTS,
        input_size: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.repo_path = Path(repo_path).expanduser()
        self.weights_path = Path(weights_path).expanduser()
        self.input_size = int(input_size)

        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"DINOv3 repository not found: {self.repo_path}")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights_path}")
        if self.input_size <= 0 or self.input_size % 16 != 0:
            raise ValueError(
                f"DINOv3 input_size must be a positive multiple of 16, got {self.input_size}."
            )

        repo_string = str(self.repo_path)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        from dinov3.hub import backbones

        if not hasattr(backbones, backbone_name):
            raise ValueError(f"Unsupported DINOv3 backbone: {backbone_name}")
        builder = getattr(backbones, backbone_name)

        # Build locally and load the checkpoint directly. The official helper
        # treats local paths as file:// URLs and copies them through torch.hub.
        self.body = builder(pretrained=False)
        state_dict = torch.load(
            self.weights_path,
            map_location="cpu",
            weights_only=True,
        )
        self.body.load_state_dict(state_dict, strict=True)

        self.num_channels = int(self.body.embed_dim)
        self.patch_size = int(self.body.patch_size)
        self.dino_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.body.forward_features(tensor)
        return features["x_norm_patchtokens"]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(
                        list(executor.map(self.dino_transform, views))
                    )
                    for views in img_list
                ]
            )

        batch_size, num_views, channels, height, width = image_tensors.shape
        image_tensors = image_tensors.view(
            batch_size * num_views,
            channels,
            height,
            width,
        )
        return image_tensors.to(next(self.parameters()).device)


def get_dinov3_model(
    backbone_name: str = "dinov3_vits16plus",
    repo_path: str | Path = DEFAULT_DINOV3_REPO,
    weights_path: str | Path = DEFAULT_DINOV3_WEIGHTS,
    input_size: int = 256,
) -> DINOv3BackBone:
    return DINOv3BackBone(
        backbone_name=backbone_name,
        repo_path=repo_path,
        weights_path=weights_path,
        input_size=input_size,
    )


def get_twochunk_dino_model(
    backbone_name: str = "dinov3_vits16plus",
    repo_path: str | Path = DEFAULT_DINOV3_REPO,
    weights_path: str | Path = DEFAULT_DINOV3_WEIGHTS,
    input_size: int = 256,
) -> nn.Module:
    """Build DINOv3 for new runs while retaining old DINOv2 checkpoint support."""
    if backbone_name.startswith("dinov2_"):
        from starVLA.model.modules.dino_model.dino import get_dino_model

        return get_dino_model(backone_name=backbone_name)
    return get_dinov3_model(
        backbone_name=backbone_name,
        repo_path=repo_path,
        weights_path=weights_path,
        input_size=input_size,
    )
