#!/usr/bin/env python3
"""
Split robomimic-style videos into train/eval and create paired symlinks for
original and pose videos.

Source structure (example):
    <task>/<type>/<demo>/<name>.mp4
    <task>/<type>/<demo>/<name>_pose.mp4

This script:
  1) Recursively finds all original/pose mp4 pairs under --src-dir
  2) Creates a deterministic train/eval split (optionally stratified by <task>/<type>)
  3) Creates symlinks:
        ./data/robomimic_dataset/training/i.mp4       (original videos)
        ./data/robomimic_pose_dataset/training/i.mp4  (pose videos)
     and similarly for validation/test.

Indices `i` are assigned from a **single global counter**: training indices come
first [0..N_train-1], then validation continues [N_train..N_total-1].

Usage example:
  python datasets/preprocess/process_robomimic.py \
    --src-dir /net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/robomimic_dataset/robomimic/datasets \
    --dst-dir ./data/robomimic_dataset --train-ratio 0.9 --seed 42
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def rel_parts_under(src_dir: Path, file_path: Path) -> Tuple[str, str, str]:
    """Return (task, type, demo) relative to src_dir.
    Expects at least <task>/<type>/<demo>/<file> depth; falls back gracefully.
    """
    rel = file_path.relative_to(src_dir)
    parts = rel.parts
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    if len(parts) == 3:
        return parts[0], parts[1], "unknown"
    if len(parts) >= 2:
        return parts[0], parts[1], "unknown"
    return "unknown", "unknown", "unknown"


def find_video_pose_pairs(src_dir: Path, keys: List[str] = None) -> List[Tuple[Path, Path]]:
    """Find all (video, pose) mp4 pairs under src_dir.
    A pair is `<name>.mp4` and `<name>_pose.mp4` in the same folder.
    Returns a list of tuples (video_path, pose_path).
    """
    video_files = sorted(
        p for p in src_dir.rglob("*.mp4")
        if p.is_file() and not p.name.endswith("_pose.mp4") \
        and (keys is None or any(k in p.stem for k in keys))
    )
    pairs: List[Tuple[Path, Path]] = []
    for v in video_files:
        pose = v.with_name(v.stem + "_pose.mp4")
        if pose.exists():
            pairs.append((v.resolve(), pose.resolve()))
        else:
            print(f"Warning: pose file missing for {v}")
    return pairs


def stratified_split(pairs: List[Tuple[Path, Path]], src_dir: Path, train_ratio: float, seed: int, stratify: bool) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    random.seed(seed)
    if not stratify:
        shuffled = pairs[:]
        random.shuffle(shuffled)
        cut = int(round(len(shuffled) * train_ratio))
        return shuffled[:cut], shuffled[cut:]

    # Group by (task, type) to keep class balance
    buckets: defaultdict = defaultdict(list)
    for v, p in pairs:
        task, typ, _ = rel_parts_under(src_dir, v)
        buckets[(task, typ)].append((v, p))

    train, eval_ = [], []
    for _, group in buckets.items():
        g = group[:]
        random.shuffle(g)
        cut = int(round(len(g) * train_ratio))
        train.extend(g[:cut])
        eval_.extend(g[cut:])

    random.shuffle(train)
    random.shuffle(eval_)
    return train, eval_


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


def link_split(pairs_train: List[Tuple[Path, Path]], pairs_eval: List[Tuple[Path, Path]], dst_video: Path, dst_pose: Path, force: bool = True, dry_run: bool = False) -> None:
    training_video = dst_video / "training"
    validation_video = dst_video / "validation"
    training_pose = dst_pose / "training"
    validation_pose = dst_pose / "validation"

    # Global sequence: training first, then validation
    sequence = [
        (training_video, training_pose, v, p) for (v, p) in pairs_train
    ] + [
        (validation_video, validation_pose, v, p) for (v, p) in pairs_eval
    ]

    for i, (subset_video, subset_pose, v, p) in enumerate(sequence):
        dst_v = subset_video / f"{i}.mp4"
        dst_p = subset_pose / f"{i}.mp4"
        if dry_run:
            print(f"LINK: {dst_v} -> {v}")
            print(f"LINK: {dst_p} -> {p}")
        else:
            make_symlink(v, dst_v, force=force)
            make_symlink(p, dst_p, force=force)


def main():
    parser = argparse.ArgumentParser(description="Split robomimic mp4 pairs and create train/eval symlinks for originals and poses.")
    parser.add_argument("--src-dir", type=Path, help="Root directory containing <task>/<type>/<demo>/*.mp4")
    parser.add_argument("--dst-dir", type=Path, default=Path("./data/robomimic_dataset"), help="Destination for original video links (robomimic_dataset)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Proportion of pairs to place in train (0-1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--no-stratify", action="store_true", help="Do not stratify by <task>/<type>; split globally")
    parser.add_argument("--no-force", action="store_true", help="Do not overwrite existing links")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating links")
    parser.add_argument("--keys", type=str, nargs='+', default=["all"], help="Only process videos whose path contains this substring")

    args = parser.parse_args()
    args.src_dir = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/robomimic_dataset/robomimic/datasets_std_0.001_128_chunk160_len160")
    args.dst_dir = Path("./data/robomimic_128_f8-32/datasets_std_0.001_128_chunk160_len160")
    # args.keys = ['merged_2x2']
    src_dir: Path = args.src_dir.resolve()
    dst_video: Path = args.dst_dir.resolve()  # ./data/robomimic/
    dst_pose: Path = dst_video.parent / Path(dst_video.name + "_pose")  # ./data/robomimic_pose/

    if not src_dir.exists():
        raise SystemExit(f"Source directory does not exist: {src_dir}")
    if 'all' in args.keys:
        args.keys = None
    print(f"Processing src_dir: {src_dir} with keys: {args.keys}")
    pairs = find_video_pose_pairs(src_dir, keys=args.keys)
    if not pairs:
        raise SystemExit(f"No video/pose .mp4 pairs found under: {src_dir}")

    print(f"Found {len(pairs)} video/pose pairs under {src_dir}")

    train_pairs, eval_pairs = stratified_split(
        pairs=pairs,
        src_dir=src_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
    )

    print(f"Split -> training: {len(train_pairs)} | validation: {len(eval_pairs)} (train_ratio={args.train_ratio})")

    link_split(
        pairs_train=train_pairs,
        pairs_eval=eval_pairs,
        dst_video=dst_video,
        dst_pose=dst_pose,
        force=not args.no_force,
        dry_run=args.dry_run,
    )

    print(f"Symlinks created under: {dst_video}/training, {dst_video}/validation and {dst_pose}/training, {dst_pose}/validation")


if __name__ == "__main__":
    main()