"""Local DINOv3 backbone wrapper for TwoChunk visual conditioning."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.body.forward_features(tensor)
        return features["x_norm_patchtokens"]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        image_tensors = self._stack_views_fast(img_list)
        if image_tensors is None:
            flat_views = [view for views in img_list for view in views]
            with ThreadPoolExecutor() as executor:
                flat_tensors = list(executor.map(self._image_to_chw_float, flat_views))
            num_views = len(img_list[0]) if img_list else 0
            image_tensors = torch.stack(flat_tensors).view(len(img_list), num_views, *flat_tensors[0].shape)

        batch_size, num_views = image_tensors.shape[:2]
        device = next(self.parameters()).device
        if image_tensors.ndim != 5:
            raise ValueError(f"Expected 5D image tensor, got {tuple(image_tensors.shape)}")
        if image_tensors.shape[2] in (1, 3):
            _, _, channels, height, width = image_tensors.shape
            image_tensors = image_tensors.view(batch_size * num_views, channels, height, width).to(
                device=device,
                non_blocking=True,
            )
        elif image_tensors.shape[-1] in (1, 3):
            _, _, height, width, channels = image_tensors.shape
            image_tensors = image_tensors.view(batch_size * num_views, height, width, channels).to(
                device=device,
                non_blocking=True,
            )
            image_tensors = image_tensors.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Cannot infer image channel dimension from shape {tuple(image_tensors.shape)}")
        if image_tensors.dtype == torch.uint8:
            image_tensors = image_tensors.to(dtype=torch.float32).div_(255.0)
        else:
            image_tensors = image_tensors.to(dtype=torch.float32)
            if image_tensors.max() > 1.0:
                image_tensors = image_tensors / 255.0
        if height != self.input_size or width != self.input_size:
            image_tensors = F.interpolate(
                image_tensors,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (image_tensors - self.dino_mean) / self.dino_std

    @staticmethod
    def _stack_views_fast(img_list) -> torch.Tensor | None:
        if not img_list or not img_list[0]:
            return None

        first = img_list[0][0]
        if isinstance(first, np.ndarray):
            try:
                array = np.stack([np.stack(views, axis=0) for views in img_list], axis=0)
            except ValueError:
                return None
            if array.ndim != 5 or array.shape[-1] not in (1, 3):
                return None
            if array.dtype == np.uint8:
                return torch.from_numpy(np.ascontiguousarray(array))
            array = array.astype(np.float32, copy=False)
            if array.max() > 1.0:
                array = array / 255.0
            return torch.from_numpy(np.ascontiguousarray(array))

        if isinstance(first, torch.Tensor):
            try:
                tensor = torch.stack([torch.stack(views, dim=0) for views in img_list], dim=0).detach()
            except RuntimeError:
                return None
            if tensor.ndim != 5:
                return None
            if tensor.shape[2] not in (1, 3) and tensor.shape[-1] not in (1, 3):
                return None
            return tensor

        return None

    @staticmethod
    def _image_to_chw_float(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach()
            if tensor.ndim != 3:
                raise ValueError(f"Expected image tensor with 3 dims, got {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.to(dtype=torch.float32)
            elif tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32)
            else:
                raise ValueError(f"Cannot infer image channel dimension from shape {tuple(tensor.shape)}")
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in (1, 3):
            raise ValueError(f"Expected HWC image array with 1 or 3 channels, got {array.shape}")
        if not array.flags.writeable:
            array = np.array(array, copy=True)
        if array.dtype != np.uint8:
            array = array.astype(np.float32, copy=False)
            if array.max() > 1.0:
                array = array / 255.0
            return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(array))
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )


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
