from typing import Optional
import torch
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange
from utils.geometry_utils import CameraPose
from .dfot_video import DFoTVideo


class DFoTVideoControl(DFoTVideo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self._check_cfg(cfg)
        super().__init__(cfg)

    def _check_cfg(self, cfg: DictConfig):
        """
        Check if the config is valid
        """
        if cfg.backbone.name not in {"mmdit3d", "u_vit3d_control"}:
            raise ValueError(
                f"DiffusionForcingVideo3D only supports backbone 'mmdit3d' or 'u_vit3d_control', got {cfg.backbone.name}"
            )



    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def _process_conditions(
        self, conditions: Tensor, noise_levels: Optional[Tensor] = None
    ) -> Tensor:
        return conditions