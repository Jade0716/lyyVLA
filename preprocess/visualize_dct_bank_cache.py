#!/usr/bin/env python3
"""Visualize one DCT-bank cache entry against the full episode action trace.

The plot overlays:
  1. the full ground-truth episode action,
  2. the IDCT reconstruction of the cached summary,
  3. the IDCT reconstruction of the cached recent chunk.

By default it randomly selects one cached episode and uses the second-to-last
frame as the training sample timestep.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import idct

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dct_bank_cache import (
    DEFAULT_ACTION_COLUMNS,
    extract_actions,
    find_episode_parquets,
    iter_episode_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobot dataset directory.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "DCT-bank cache dir. Default: "
            "<dataset-dir>/meta/dct_bank_cache/chunk32_recent4_summary8_sliding-start"
        ),
    )
    parser.add_argument("--episode-index", type=int, default=None, help="Episode to visualize. Default: random.")
    parser.add_argument(
        "--timestep",
        type=int,
        default=-2,
        help="Training sample timestep. Negative values are relative to episode end; default -2.",
    )
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument(
        "--chunk-keep-freq",
        "--keep-freq",
        dest="chunk_keep_freq",
        type=int,
        default=4,
        help="Fallback recent chunk DCT frequency count for old caches.",
    )
    parser.add_argument(
        "--summary-keep-freq",
        type=int,
        default=8,
        help="Fallback summary DCT frequency count for old caches.",
    )
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--action-columns", default=",".join(DEFAULT_ACTION_COLUMNS))
    parser.add_argument("--no-clip-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png path. Default: ./viz/<cache-dir>/episode_<id>_t_<t>.png",
    )
    return parser.parse_args()


def dct_to_signal(coeff: np.ndarray, length: int, keep_freq: int) -> np.ndarray:
    padded = np.zeros((length, coeff.shape[-1]), dtype=np.float32)
    padded[:keep_freq] = coeff[:keep_freq].astype(np.float32)
    return idct(padded, type=2, n=length, axis=0, norm="ortho").astype(np.float32)


def load_episode_actions(
    dataset_dir: Path,
    episode_index: int,
    action_column: str,
    action_columns: tuple[str, ...],
    clip_actions: bool,
) -> np.ndarray:
    for parquet_path in find_episode_parquets(dataset_dir):
        for current_episode_index, frame in iter_episode_frames(parquet_path):
            if int(current_episode_index) == int(episode_index):
                return extract_actions(
                    frame,
                    action_column=action_column,
                    action_columns=action_columns,
                    clip_actions=clip_actions,
                )
    raise FileNotFoundError(f"Could not find episode {episode_index} under {dataset_dir / 'data'}")


def select_episode(cache_dir: Path, episode_index: int | None, seed: int) -> tuple[int, Path]:
    if episode_index is not None:
        path = cache_dir / f"episode_{episode_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return episode_index, path

    cache_files = sorted(cache_dir.glob("episode_*.npz"))
    if not cache_files:
        raise FileNotFoundError(f"No episode_*.npz files found under {cache_dir}")
    rng = random.Random(seed)
    path = rng.choice(cache_files)
    selected = int(path.stem.split("_")[-1])
    return selected, path


def compute_memory_indices(timestep: int, chunk_len: int) -> tuple[int, int, int]:
    usable_chunks = timestep // chunk_len
    if usable_chunks <= 0:
        return 0, 0, 0
    start = timestep - usable_chunks * chunk_len
    summary_idx = usable_chunks - 2
    recent_idx = usable_chunks - 1
    return start, summary_idx, recent_idx


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = dataset_dir / "meta" / "dct_bank_cache" / "chunk32_recent4_summary8_sliding-start"
    cache_dir = cache_dir.expanduser().resolve()

    episode_index, cache_path = select_episode(cache_dir, args.episode_index, args.seed)
    cache = np.load(cache_path)
    actions = load_episode_actions(
        dataset_dir=dataset_dir,
        episode_index=episode_index,
        action_column=args.action_column,
        action_columns=tuple(args.action_columns.split(",")),
        clip_actions=not args.no_clip_actions,
    )

    episode_len, action_dim = actions.shape
    timestep = args.timestep if args.timestep >= 0 else episode_len + args.timestep
    timestep = int(np.clip(timestep, 0, episode_len - 1))

    chunk_len = int(cache["chunk_len"]) if "chunk_len" in cache else args.chunk_len
    if "chunk_keep_freq" in cache:
        chunk_keep_freq = int(cache["chunk_keep_freq"])
    else:
        chunk_keep_freq = int(cache["keep_freq"]) if "keep_freq" in cache else args.chunk_keep_freq
    if "summary_keep_freq" in cache:
        summary_keep_freq = int(cache["summary_keep_freq"])
    else:
        summary_keep_freq = int(cache["keep_freq"]) if "keep_freq" in cache else args.summary_keep_freq
    start, summary_idx, recent_idx = compute_memory_indices(
        timestep=timestep,
        chunk_len=chunk_len,
    )

    summary_signal = None
    summary_x = None
    recent_signal = None
    recent_x = None

    if recent_idx >= 0:
        if "chunk_coarse" in cache:
            recent_signal = cache["chunk_coarse"][start, recent_idx].astype(np.float32)
        else:
            recent_signal = dct_to_signal(cache["chunk_dct"][start, recent_idx], chunk_len, chunk_keep_freq)
        recent_x = np.arange(start + recent_idx * chunk_len, start + (recent_idx + 1) * chunk_len)

    if summary_idx >= 0:
        summary_count = int(cache["prefix_summary_count"][start, summary_idx])
        summary_len = summary_count * chunk_len
        summary_signal = dct_to_signal(
            cache["prefix_summary_dct"][start, summary_idx],
            summary_len,
            summary_keep_freq,
        )
        summary_x = np.arange(start, start + summary_len)

    labels = [col.strip().replace("action.", "") for col in args.action_columns.split(",")]
    if len(labels) != action_dim:
        labels = [f"action_{i}" for i in range(action_dim)]

    fig, axes = plt.subplots(action_dim, 1, figsize=(18, max(10, action_dim * 2.1)), sharex=True)
    if action_dim == 1:
        axes = [axes]

    x = np.arange(episode_len)
    for dim, ax in enumerate(axes):
        ax.plot(x, actions[:, dim], color="#1f2937", linewidth=1.2, label="full action")
        if summary_signal is not None and summary_x is not None:
            ax.plot(
                summary_x,
                summary_signal[:, dim],
                color="#2563eb",
                linewidth=1.8,
                label="summary IDCT",
            )
        if recent_signal is not None and recent_x is not None:
            ax.plot(
                recent_x,
                recent_signal[:, dim],
                color="#dc2626",
                linewidth=1.8,
                label="recent IDCT",
            )
        ax.axvline(timestep, color="#059669", linestyle="--", linewidth=1.0, label="sample timestep")
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=4, frameon=False)
    axes[-1].set_xlabel("episode frame")
    fig.suptitle(
        f"episode={episode_index}, timestep={timestep}, start={start}, "
        f"summary_idx={summary_idx}, recent_idx={recent_idx}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    output = args.output
    if output is None:
        output_dir = cache_dir / "viz"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"episode_{episode_index:06d}_t_{timestep}.png"
    else:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=180)
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
