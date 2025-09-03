from typing import Literal, List, Dict, Any, Callable, Tuple, Optional
from abc import ABC, abstractmethod
import random
import bisect
from pathlib import Path
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from tqdm import tqdm
from einops import rearrange
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from datasets.video.utils import read_video, VideoTransform
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)

SPLIT = Literal["train", "eval", "test"]


class BasePoseVideoDataset(BaseVideoDataset):
    """
    Common base class for video dataset with pose videos.
    Methods here are shared between simple and advanced video datasets

    Folder structure of each dataset:
    - {save_dir} (specified in config, e.g., data/phys101)
        - /{split}
            - data files (e.g. 000001.mp4, 000001.pt)
        - /metadata
            - {split}.pt
    - {save_dir}_latent_{latent_resolution} (same structure as save_dir)
    - {save_dir}_pose (same structure as save_dir)
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
    ):
        super(BaseVideoDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.latent_resolution = cfg.resolution // cfg.latent.downsampling_factor[1]
        self.save_dir = Path(cfg.save_dir)
        self.pose_save_dir = Path(str(self.save_dir) + "_pose")
        self.latent_dir = self.save_dir.with_name(
            f"{self.save_dir.name}_latent_{self.latent_resolution}{'_' + cfg.latent.suffix if cfg.latent.suffix else ''}"
        )
        self.pose_latent_dir = self.pose_save_dir.with_name(
            f"{self.pose_save_dir.name}_latent_{self.latent_resolution}{'_' + cfg.latent.suffix if cfg.latent.suffix else ''}"
        )
        self.split_dir = self.save_dir / split
        self.pose_split_dir = self.pose_save_dir / split
        self.metadata_dir = self.save_dir / "metadata"
        self.pose_metadata_dir = self.pose_save_dir / "metadata"

        # Download dataset if not exists
        if self._should_download():
            self.download_dataset()
        if not self.metadata_dir.exists() or not any(self.metadata_dir.iterdir()):
            self.metadata_dir.mkdir(exist_ok=True, parents=True)
            self.pose_metadata_dir.mkdir(exist_ok=True, parents=True)
            for split in self._ALL_SPLITS:
                self.build_metadata(split)

        self.metadata, self.pose_metadata = self.load_metadata()
        self.augment_dataset()
        self.transform = self.build_transform()

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir
        This may vary depending on the dataset.
        Default:
        ```
        {
            "video_paths": List[str],
            "video_pts": List[str],
            "video_fps": List[float],
        }
        ```
        """
        video_paths = sorted(list((self.save_dir / split).glob("**/*.mp4")), key=str)
        pose_video_paths = sorted(list((self.pose_save_dir / split).glob("**/*.mp4")), key=str)
        assert all(pose_video_paths[i] == self.video_path_to_pose_path(video_paths[i]) for i in range(len(video_paths))) \
            and len(video_paths) == len(pose_video_paths), \
             "Video and pose video files do not match"
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        dl_pose: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(pose_video_paths),
            batch_size=16,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        video_pts: List[torch.Tensor] = (
            []
        )  # each entry is a tensor of shape (num_frames, )
        video_fps: List[float] = []
        pose_video_pts: List[torch.Tensor] = (
            []
        )
        pose_video_fps: List[float] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch, batch_pose in zip(dl, dl_pose):
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts
                ]
                batch_pose_pts, batch_pose_fps = list(zip(*batch_pose))
                batch_pose_pts = [
                    torch.as_tensor(pts, dtype=torch.long) for pts in batch_pose_pts
                ]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)
                pose_video_pts.extend(batch_pose_pts)
                pose_video_fps.extend(batch_pose_fps)

        ## filter out videos that timestamps do not match:
        valid_indices = [
            i for i in range(len(video_pts))
            if torch.equal(video_pts[i], pose_video_pts[i]) and video_fps[i] == pose_video_fps[i]
        ]
        rank_zero_print(
            cyan(f"{len(valid_indices)} out of {len(video_paths)} videos in {split} have matching timestamps and fps between video and pose video")
        )
        video_paths = [video_paths[i] for i in valid_indices]
        video_pts = [video_pts[i] for i in valid_indices]
        video_fps = [video_fps[i] for i in valid_indices]
        pose_video_paths = [pose_video_paths[i] for i in valid_indices]
        pose_video_pts = [pose_video_pts[i] for i in valid_indices]
        pose_video_fps = [pose_video_fps[i] for i in valid_indices]
        assert all(torch.equal(video_pts[i], pose_video_pts[i]) and video_fps[i] == pose_video_fps[i] for i in range(len(video_pts)))

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        pose_metadata = {
            "video_paths": pose_video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")
        torch.save(pose_metadata, self.pose_metadata_dir / f"{split}.pt")

    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata from metadata_dir
        """
        metadata = torch.load(
            self.metadata_dir / f"{self.split}.pt", weights_only=False
        )
        pose_metadata = torch.load(
            self.pose_metadata_dir / f"{self.split}.pt", weights_only=False
        )
        return [
            {key: metadata[key][i] for key in metadata.keys()}
            for i in range(len(metadata["video_paths"]))
        ], [
            {key: pose_metadata[key][i] for key in pose_metadata.keys()}
            for i in range(len(pose_metadata["video_paths"]))
        ]


    def video_path_to_pose_path(self, video_path: Path) -> Path:
        pose_save_dir = Path(str(video_path.parent.parent) + '_pose')
        assert pose_save_dir == self.pose_save_dir, f"the parent dir {pose_save_dir} is different from {self.pose_save_dir}"
        pose_video_path = Path(pose_save_dir / video_path.relative_to(self.save_dir))
        return pose_video_path

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load video from video_idx with given start_frame and end_frame (exclusive)
        if end_frame is None, load until the end of the video
        return shape: (T, C, H, W)
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path, video_pts = (
            video_metadata["video_paths"],
            video_metadata["video_pts"],
        )
        start_pts = video_pts[start_frame].item()
        end_pts = video_pts[end_frame - 1].item()
        video = read_video(video_path, start_pts, end_pts)
        return video.permute(0, 3, 1, 2) / 255.0


class BaseSimplePoseVideoDataset(BaseVideoDataset):
    """
    Base class for simple video datasets
    that load full videos with given resolution
    Also provides latent_path where latent should be saved
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        super().__init__(cfg, split)
        self.latent_dir.mkdir(exist_ok=True, parents=True)
        # filter videos to only include the ones that have not been preprocessed
        self.metadata = self.exclude_videos_with_latents(self.metadata)

    def exclude_videos_with_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))

        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            not in latent_paths,
            "videos that have already been preprocessed to latents",
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        loads video together with the path where latent should be saved
        """
        video_metadata = self.metadata[idx]
        video = self.load_video(video_metadata, 0)

        return (
            self.transform(video),
            self.video_metadata_to_latent_path(video_metadata).as_posix(),
        )


class BaseAdvancedPoseVideoDataset(BaseAdvancedVideoDataset, BasePoseVideoDataset):
    """
    Base class for video dataset
    that load video clips with given resolution and frame skip
    Videos may be of variable lengths.
    """
    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        super().__init__(cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        from copy import deepcopy
        pose_video_metadata = deepcopy(video_metadata)
        pose_video_metadata['video_paths'] = self.video_path_to_pose_path(video_metadata['video_paths'])
        return self.load_video(pose_video_metadata, start_frame, end_frame)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        video, latent, cond = None, None, None
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)

        if self.use_preprocessed_latents and self.split == "training":
            # do not load video if we are training with latents
            if self.external_cond_dim > 0:
                cond = self.load_cond(video_metadata, start_frame, end_frame)

        else:
            if self.external_cond_dim > 0:
                # load video together with condition
                video, cond = self.load_video_and_cond(
                    video_metadata, start_frame, end_frame
                )
            else:
                # load video only
                video = self.load_video(video_metadata, start_frame, end_frame)

        lens = [len(x) for x in (video, cond, latent) if x is not None]
        assert len(set(lens)) == 1, "video, cond, latent must have the same length"
        pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            if video is not None:
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if latent is not None:
                latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if cond is not None:
                cond = F.pad(cond, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0

        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
        if cond is not None:
            cond = self._process_external_cond(cond)

        output = {
            "videos": self.transform(video) if video is not None else None,
            "latents": latent,
            "conds": self.transform(cond) if cond is not None else None,
            "nonterminal": nonterminal,
        }
        return {key: value for key, value in output.items() if value is not None}

    def _process_external_cond(
            self, external_cond: torch.Tensor, frame_skip: Optional[int] = None
    ) -> torch.Tensor:
        """
        Post-processes external condition.
        """
        return external_cond[:: frame_skip or self.frame_skip]
