#!/usr/bin/env python3
"""
Fuse multiple video and pose datasets by creating symlinks into a single
destination that follows the same folder structure as process_dataset.py.

Structure produced (same as process_dataset.py):
  - <dst_dir>/training/<i>.mp4         (original videos)
  - <dst_pose_dir>/training/<i>.mp4    (pose videos)
  - <dst_dir>/validation/<i>.mp4
  - <dst_pose_dir>/validation/<i>.mp4

Indices are assigned from a single global counter: training first, then
validation continues the sequence. Files are discovered by matching identical
relative paths under provided video and pose roots (e.g., training/3144.mp4).

Example:
  python datasets/preprocess/fuse_dataset.py \
    --video-roots data/robomimic/datasets_std_0.0 data/robomimic/datasets_std_0.5 \
    --pose-roots  data/robomimic/datasets_std_0.0_pose data/robomimic/datasets_std_0.5_pose \
    --dst-dir     ./data/robomimic/datasets_std_merged
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SUBSETS = {"training", "validation"}


def make_symlink(src: Path, dst: Path, force: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            if force:
                dst.unlink()
            else:
                return
        os.symlink(src.as_posix(), dst.as_posix())
    except FileExistsError:
        pass


def scan_map(roots: Iterable[Path]) -> Dict[str, Path]:
    """Build map of relative path -> absolute path for all valid video files."""
    out: Dict[str, Path] = {}
    for root in roots:
        root = root.resolve()
        if not root.exists():
            print(f"Warning: root does not exist, skipping: {root}")
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            rel = p.relative_to(root).as_posix()
            # Keep first occurrence if duplicates across roots
            if rel not in out:
                out[rel] = p.resolve()
    return out


def gather_pairs(
    video_roots: List[Path], pose_roots: List[Path]
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[str]]:
    """Return (train_pairs, val_pairs, skipped) where pairs are (video, pose).

    Pairs are matched by identical relative path under video and pose roots.
    Only entries under subsets {training, validation} are kept; others are
    collected into skipped list (by rel path) for info.
    """
    pose_map = scan_map(pose_roots)
    train, val = [], []
    skipped: List[str] = []

    for root in video_roots:
        root = root.resolve()
        if not root.exists():
            print(f"Warning: video root does not exist, skipping: {root}")
            continue
        for v in sorted(root.rglob("*")):
            if not v.is_file() or v.suffix.lower() not in VIDEO_EXTS:
                continue
            if v.name.endswith("_pose" + v.suffix.lower()):
                # Safety: ignore accidental pose files in video roots
                continue
            rel = v.relative_to(root).as_posix()
            pose_path = pose_map.get(rel)
            if pose_path is None:
                # No pose counterpart found; skip
                continue
            subset = rel.split("/", 1)[0]
            if subset not in SUBSETS:
                skipped.append(rel)
                continue
            if subset == "training":
                train.append((v.resolve(), pose_path))
            else:
                val.append((v.resolve(), pose_path))

    return train, val, skipped


def link_pairs(
    train_pairs: List[Tuple[Path, Path]],
    val_pairs: List[Tuple[Path, Path]],
    dst_video: Path,
    dst_pose: Path,
    *,
    force: bool,
    dry_run: bool,
) -> None:
    sequence = [
        (dst_video / "training", dst_pose / "training", v, p) for (v, p) in train_pairs
    ] + [
        (dst_video / "validation", dst_pose / "validation", v, p) for (v, p) in val_pairs
    ]

    for i, (sv, sp, v, p) in enumerate(sequence):
        dv = sv / f"{i}.mp4"
        dp = sp / f"{i}.mp4"
        if dry_run:
            print(f"LINK: {dv} -> {v}")
            print(f"LINK: {dp} -> {p}")
        else:
            make_symlink(v, dv, force=force)
            make_symlink(p, dp, force=force)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fuse multiple video/pose roots into a merged dataset using symlinks.")
    ap.add_argument("--video-roots", type=Path, nargs="+", help="One or more roots with original videos")
    ap.add_argument("--pose-roots", type=Path, nargs="+", help="One or more roots with pose videos")
    ap.add_argument("--dst-dir", type=Path, help="Destination root for original videos; pose root is '<dst-dir>_pose'")
    ap.add_argument("--no-force", action="store_true", help="Do not overwrite existing links")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without creating links")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.dst_dir = Path("data/robomimic_128/datasets_noisy")
    dst_video = args.dst_dir.resolve()
    args.video_roots = [Path("data/robomimic_128/datasets_std_0.0"), Path("data/robomimic_128/datasets_std_0.1"), Path("data/robomimic_128/datasets_std_0.5")]
    args.pose_roots = [Path("data/robomimic_128/datasets_std_0.0_pose"), Path("data/robomimic_128/datasets_std_0.1_pose"), Path("data/robomimic_128/datasets_std_0.5_pose")]
    dst_pose = (dst_video.parent / (dst_video.name + "_pose")).resolve()

    train_pairs, val_pairs, skipped = gather_pairs(
        video_roots=[r.resolve() for r in args.video_roots],
        pose_roots=[r.resolve() for r in args.pose_roots],
    )

    print(
        f"Found pairs -> training: {len(train_pairs)} | validation: {len(val_pairs)}"
        + (f" | skipped (non-subset): {len(skipped)}" if skipped else "")
    )

    link_pairs(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        dst_video=dst_video,
        dst_pose=dst_pose,
        force=not args.no_force,
        dry_run=args.dry_run,
    )

    if skipped:
        print("Note: Skipped entries not under training/ or validation/ (first 20):")
        for r in skipped[:20]:
            print(f"  - {r}")

    print(
        f"Symlinks created under: {dst_video}/training, {dst_video}/validation and "
        f"{dst_pose}/training, {dst_pose}/validation"
    )


if __name__ == "__main__":
    main()
