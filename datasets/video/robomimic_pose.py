"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py
"""

from typing import Any, Dict, List, Optional, Literal
from fractions import Fraction
import csv
import os
import random
from os import path
from pathlib import Path
import urllib
import shutil
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig
import torch
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    download_url,
)
from tqdm import tqdm
import numpy as np
from utils.print_utils import cyan
from .base_pose_video import (
    BasePoseVideoDataset,
    BaseSimplePoseVideoDataset,
    BaseAdvancedPoseVideoDataset,
    SPLIT,
)
from .utils import read_video, rescale_and_crop

VideoPreprocessingType = Literal["npz", "mp4"]
VideoPreprocessingMp4FPS: int = 20

def _preprocess_video(
    video_path: Path,
    resolution: int,
    preprocessing_type: VideoPreprocessingType = "npz",
):
    try:
        video = read_video(str(video_path))
        video = rescale_and_crop(video, resolution)
        video_path = (
            video_path.parent.parent
            / f"{video_path.parent.name}_preprocessed_{resolution}_{preprocessing_type}"
            / video_path.name
        ).with_suffix("." + preprocessing_type)
        if preprocessing_type == "npz":
            np.savez_compressed(
                video_path,
                video=video.transpose(0, 3, 1, 2).copy(),
            )
        elif preprocessing_type == "mp4":
            write_video(
                filename=video_path,
                video_array=torch.from_numpy(video).clone(),
                fps=VideoPreprocessingMp4FPS,
            )

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    # remove original video
    # video_path.unlink()


class RobomimicBasePoseVideoDataset(BasePoseVideoDataset):
   
    _ALL_SPLITS = ["training", "validation"]

    @property
    def use_video_preprocessing(self) -> bool:
        return self.cfg.video_preprocessing is not None

    def download_dataset(self):
        pass

    def setup(self) -> None:
        if self.use_video_preprocessing:
            if not (
                self.save_dir
                / f"{self.split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            ).exists():
                for split in ["training", "validation"]:
                    self._preprocess_videos(split)
            self.metadata = self.exclude_failed_videos(self.metadata)
            self.transform = lambda x: x

    def _preprocess_videos(self, split: SPLIT) -> None:
        """
        Preprocesses videos to {self.resolution}x{self.resolution} resolution
        """
        print(
            cyan(
                f"Preprocessing {split} videos to {self.resolution}x{self.resolution}..."
            )
        )
        (
            self.save_dir
            / f"{split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
        ).mkdir(parents=True, exist_ok=True)
        (
            self.pose_save_dir
            / f"{split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
        ).mkdir(parents=True, exist_ok=True)

        video_paths = torch.load(self.metadata_dir / f"{split}.pt", weights_only=False)
        video_paths = video_paths["video_paths"]
        pose_video_paths = torch.load(self.pose_metadata_dir / f"{split}.pt", weights_only=False)['video_paths']
        video_paths += pose_video_paths
        preprocess_fn = partial(
            _preprocess_video,
            resolution=self.resolution,
            preprocessing_type=self.cfg.video_preprocessing,
        )
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(preprocess_fn, video_paths),
                    total=len(video_paths),
                    desc=f"Preprocessing {split} videos",
                )
            )

    def exclude_failed_videos(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that failed to preprocess
        """
        preprocessed_video_paths = set(
            list(
                (
                    self.save_dir
                    / f"{self.split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
                ).glob(f"**/*.{self.cfg.video_preprocessing}")
            )
        )
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_path_to_preprocessed_path(
                video_metadata["video_paths"]
            )
            in preprocessed_video_paths,
            "failed-to-preprocess videos",
        )

    def video_path_to_preprocessed_path(self, video_path: Path) -> Path:
        return (
            video_path.parent.parent
            / f"{video_path.parent.name}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            / video_path.name
        ).with_suffix("." + self.cfg.video_preprocessing)

    def load_video(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        if self.use_video_preprocessing:
            preprocessed_path = self.video_path_to_preprocessed_path(
                video_metadata["video_paths"]
            )
            match self.cfg.video_preprocessing:
                case "npz":
                    video = np.load(
                        preprocessed_path,
                    )[
                        "video"
                    ][start_frame:end_frame]
                    return torch.from_numpy(video / 255.0).float()
                case "mp4":
                    video = read_video(
                        preprocessed_path,
                        pts_unit="sec",
                        start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                        end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                    )
                    return video.permute(0, 3, 1, 2) / 255.0
        else:
            return super().load_video(video_metadata, start_frame, end_frame)


class RobomimicSimplePoseVideoDataset(
    RobomimicBasePoseVideoDataset, BaseSimplePoseVideoDataset
):
    """
    Robomimic simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimplePoseVideoDataset.__init__(self, cfg, split)
        self.setup()


class RobomimicAdvancedPoseVideoDataset(
    RobomimicBasePoseVideoDataset, BaseAdvancedPoseVideoDataset
):
    """
    Robomimic advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "test":
            split = "validation"
        BaseAdvancedPoseVideoDataset.__init__(self, cfg, split, current_epoch)

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def setup(self) -> None:
        super().setup()



