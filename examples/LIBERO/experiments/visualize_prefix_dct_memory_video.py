#!/usr/bin/env python3
"""Render a LIBERO10 video with the active prefix DCT memory curve on top."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.fft import idct
from tqdm import tqdm
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402


ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#111111")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/15T/liuyuyan/libero/libero_10_no_noops_1.0.0_lerobot"),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/meta/dct_bank_cache/chunk32_recent4_summary8_prefix-summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/LIBERO/experiments/libero10_prefix_dct_memory_episode_000000.mp4"),
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means render the full episode.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--plot-height", type=int, default=360)
    parser.add_argument("--video-width", type=int, default=768)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--show-raw", action="store_true", help="Also draw raw GT actions as faint dotted curves.")
    return parser.parse_args()


def episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    return int(episode_index) // int(chunk_size)


def episode_paths(dataset_dir: Path, episode_index: int, video_key: str, cache_dir: Path) -> tuple[Path, Path, Path]:
    chunk = episode_chunk(episode_index)
    video_path = dataset_dir / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    parquet_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    cache_path = cache_dir / f"episode_{episode_index:06d}.npz"
    for path in (video_path, parquet_path, cache_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return video_path, parquet_path, cache_path


def load_actions(parquet_path: Path) -> np.ndarray:
    frame = pd.read_parquet(parquet_path)
    if "action" in frame.columns:
        values = frame["action"].to_numpy()
        actions = np.asarray([np.asarray(v, dtype=np.float32) for v in values], dtype=np.float32)
    else:
        cols = [f"action.{name}" for name in ACTION_NAMES]
        missing = [col for col in cols if col not in frame.columns]
        if missing:
            raise KeyError(f"Missing action columns in {parquet_path}: {missing}")
        actions = frame[cols].to_numpy(dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected action array [T, D], got {actions.shape}")
    return np.clip(actions[:, : len(ACTION_NAMES)], -1.0, 1.0)


def reconstruct_summary(summary_dct: np.ndarray, length: int, keep_freq: int) -> np.ndarray:
    padded = np.zeros((length, summary_dct.shape[-1]), dtype=np.float32)
    padded[:keep_freq] = summary_dct[:keep_freq].astype(np.float32)
    return idct(padded, type=2, n=length, axis=0, norm="ortho").astype(np.float32)


def render_plot(
    *,
    frame_idx: int,
    total_frames: int,
    raw_actions: np.ndarray,
    prefix_summary_dct: np.ndarray,
    prefix_summary_count: np.ndarray,
    chunk_len: int,
    summary_keep_freq: int,
    plot_width: int,
    plot_height: int,
    show_raw: bool,
) -> np.ndarray:
    usable_chunks = min(frame_idx // chunk_len, len(prefix_summary_dct))
    memory_len = usable_chunks * chunk_len
    fig = plt.figure(figsize=(plot_width / 100.0, plot_height / 100.0), dpi=100)
    ax = fig.add_subplot(111)

    y_min = float(np.nanmin(raw_actions))
    y_max = float(np.nanmax(raw_actions))
    pad = max(0.05, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(0, max(total_frames - 1, 1))
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.axvline(frame_idx, color="#111111", linewidth=1.2, alpha=0.55)

    if show_raw:
        raw_x = np.arange(len(raw_actions))
        for dim in range(raw_actions.shape[1]):
            ax.plot(raw_x, raw_actions[:, dim], color=ACTION_COLORS[dim], linewidth=0.7, alpha=0.18, linestyle=":")

    if usable_chunks > 0:
        summary_idx = usable_chunks - 1
        count = int(prefix_summary_count[summary_idx])
        recon = reconstruct_summary(
            prefix_summary_dct[summary_idx],
            length=memory_len,
            keep_freq=summary_keep_freq,
        )
        x = np.arange(memory_len)
        for dim in range(min(recon.shape[1], len(ACTION_NAMES))):
            ax.plot(x, recon[:, dim], color=ACTION_COLORS[dim], linewidth=1.5, label=ACTION_NAMES[dim])
        title = (
            f"prefix DCT memory at frame {frame_idx} | "
            f"complete chunks={usable_chunks}, summary_count={count}, memory_len={memory_len}"
        )
    else:
        title = f"prefix DCT memory at frame {frame_idx} | no complete 32-frame chunk yet"
        ax.text(
            0.5,
            0.5,
            "no prefix memory",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
            color="#777777",
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("episode frame")
    ax.set_ylabel("action value")
    if usable_chunks > 0:
        ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)
    fig.tight_layout(pad=0.5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def resize_keep_aspect(frame_rgb: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    target_height = int(round(height * (target_width / float(width))))
    return cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = dataset_dir / "meta/dct_bank_cache/chunk32_recent4_summary8_prefix-summary"
    cache_dir = cache_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    video_path, parquet_path, cache_path = episode_paths(
        dataset_dir,
        args.episode_index,
        args.video_key,
        cache_dir,
    )
    raw_actions = load_actions(parquet_path)
    cache = np.load(cache_path)
    chunk_len = int(cache["chunk_len"])
    summary_keep_freq = int(cache["summary_keep_freq"])
    prefix_summary_dct = cache["prefix_summary_dct"]
    prefix_summary_count = cache["prefix_summary_count"]

    frames = get_all_frames(
        video_path.as_posix(),
        video_backend=args.video_backend,
        video_backend_kwargs={
            "device": "cpu",
            "num_ffmpeg_threads": 1,
            "fallback_backend": "torchcodec_cpu",
        },
    )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected decoded video frames [T,H,W,3], got {frames.shape}")

    total_video_frames = int(frames.shape[0])
    total_frames = min(total_video_frames, len(raw_actions))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    plot_width = int(args.video_width)
    first_rgb = frames[0]
    video_preview = resize_keep_aspect(first_rgb, plot_width)
    out_height = args.plot_height + video_preview.shape[0]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (plot_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")

    metadata = {
        "dataset_dir": str(dataset_dir),
        "episode_index": args.episode_index,
        "video_key": args.video_key,
        "video_path": str(video_path),
        "parquet_path": str(parquet_path),
        "cache_path": str(cache_path),
        "output": str(output),
        "frames": total_frames,
        "chunk_len": chunk_len,
        "summary_keep_freq": summary_keep_freq,
        "prefix_summary_shape": list(prefix_summary_dct.shape),
    }

    try:
        for frame_idx in tqdm(range(total_frames), desc="Rendering prefix DCT memory video"):
            frame_rgb = frames[frame_idx]
            video_rgb = resize_keep_aspect(frame_rgb, plot_width)
            plot_rgb = render_plot(
                frame_idx=frame_idx,
                total_frames=len(raw_actions),
                raw_actions=raw_actions,
                prefix_summary_dct=prefix_summary_dct,
                prefix_summary_count=prefix_summary_count,
                chunk_len=chunk_len,
                summary_keep_freq=summary_keep_freq,
                plot_width=plot_width,
                plot_height=args.plot_height,
                show_raw=args.show_raw,
            )
            combined_rgb = np.concatenate([plot_rgb, video_rgb], axis=0)
            writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
