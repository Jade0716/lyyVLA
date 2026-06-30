#!/usr/bin/env python3
"""Visualize recursive DCT summary memory against original LIBERO actions.

This script matches the previous DCTMemory training cache semantics from
``preprocess/build_dct_bank_cache.py`` and ``_attach_dct_memory``:

  - cache mode is usually ``sliding-start``;
  - memory updates every 32 actions;
  - summary is recursively compressed with IDCT(summary) + IDCT(new chunk),
    then DCT keep ``summary_keep_freq``;
  - for one training timestep, summary excludes the most recent complete
    chunk, which is stored as ``recent``.

By default it picks the last frame of episode 0 and plots only the recursive
summary span, overlaying original actions and the IDCT reconstruction of the
cached summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import idct

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "preprocess"))

from build_dct_bank_cache import DEFAULT_ACTION_COLUMNS, extract_actions, find_episode_parquets, iter_episode_frames


DEFAULT_DATASET_DIR = Path("/15T/liuyuyan/libero/libero_10_no_noops_1.0.0_lerobot")
DEFAULT_CACHE_SUFFIX = "chunk32_recent4_summary8_sliding-start"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--base-index",
        type=int,
        default=-1,
        help="Training sample base index. Negative is relative to episode end; default -1 means last frame.",
    )
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--action-columns", default=",".join(DEFAULT_ACTION_COLUMNS))
    parser.add_argument("--no-clip-actions", action="store_true", help="Do not apply the cache builder's clip[-1,1].")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


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
    raise FileNotFoundError(f"Episode {episode_index} not found under {dataset_dir / 'data'}")


def dct_to_signal(coeff: np.ndarray, length: int, keep_freq: int) -> np.ndarray:
    padded = np.zeros((length, coeff.shape[-1]), dtype=np.float32)
    padded[:keep_freq] = coeff[:keep_freq].astype(np.float32)
    return idct(padded, type=2, n=length, axis=0, norm="ortho").astype(np.float32)


def row_for_start(cache: np.lib.npyio.NpzFile, start: int) -> int:
    starts = cache["start_indices"]
    matches = np.nonzero(starts == int(start))[0]
    if len(matches) != 1:
        raise ValueError(f"Could not find unique cache row for start={start}; matches={matches.tolist()}")
    return int(matches[0])


def default_cache_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "meta" / "dct_bank_cache" / DEFAULT_CACHE_SUFFIX


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else default_cache_dir(dataset_dir)
    cache_path = cache_dir / f"episode_{args.episode_index:06d}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    actions = load_episode_actions(
        dataset_dir=dataset_dir,
        episode_index=args.episode_index,
        action_column=args.action_column,
        action_columns=tuple(args.action_columns.split(",")),
        clip_actions=not args.no_clip_actions,
    )
    cache = np.load(cache_path)

    episode_len, action_dim = actions.shape
    base_index = args.base_index if args.base_index >= 0 else episode_len + args.base_index
    base_index = int(np.clip(base_index, 0, episode_len - 1))

    chunk_len = int(cache["chunk_len"])
    summary_keep_freq = int(cache["summary_keep_freq"])
    usable_chunks = base_index // chunk_len
    if usable_chunks < 2:
        raise ValueError(
            f"base_index={base_index} has usable_chunks={usable_chunks}; "
            "need at least 2 chunks so summary exists."
        )

    memory_start = base_index - usable_chunks * chunk_len
    row = row_for_start(cache, memory_start)
    summary_idx = usable_chunks - 2
    recent_idx = usable_chunks - 1
    summary_count = int(cache["prefix_summary_count"][row, summary_idx])
    summary_len = summary_count * chunk_len
    summary_start = memory_start
    summary_end = summary_start + summary_len

    summary_coeff = cache["prefix_summary_dct"][row, summary_idx].astype(np.float32)
    summary_recon = dct_to_signal(summary_coeff, summary_len, summary_keep_freq)
    original = actions[summary_start:summary_end].astype(np.float32)
    if original.shape != summary_recon.shape:
        min_len = min(original.shape[0], summary_recon.shape[0])
        original = original[:min_len]
        summary_recon = summary_recon[:min_len]
        summary_end = summary_start + min_len

    err = summary_recon - original
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))

    labels = [col.strip().replace("action.", "") for col in args.action_columns.split(",")]
    if len(labels) != action_dim:
        labels = [f"action_{i}" for i in range(action_dim)]

    x = np.arange(summary_start, summary_end)
    fig, axes = plt.subplots(action_dim, 1, figsize=(18, max(10, action_dim * 2.2)), sharex=True)
    if action_dim == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(x, original[:, dim], color="#111827", linewidth=1.1, label="original clipped action")
        ax.plot(x, summary_recon[:, dim], color="#2563eb", linewidth=1.7, label="recursive summary IDCT")
        ax.set_ylabel(labels[dim])
        ax.set_title(f"{labels[dim]}  MAE={mae[dim]:.4f}  RMSE={rmse[dim]:.4f}", loc="left", fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("episode frame")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(
        "Recursive DCT summary vs original actions | "
        f"episode={args.episode_index}, base_index={base_index}, "
        f"memory_start={memory_start}, summary_idx={summary_idx}, recent_idx={recent_idx}, "
        f"summary_count={summary_count}, summary_span=[{summary_start},{summary_end})",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    if args.output is None:
        output_dir = REPO_ROOT / "preprocess" / "experiments" / "recursive_summary_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"episode_{args.episode_index:06d}_base_{base_index}_summary.png"
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=args.dpi)
    print(f"Saved {output}")
    print(
        "metadata: "
        f"episode_len={episode_len}, base_index={base_index}, chunk_len={chunk_len}, "
        f"usable_chunks={usable_chunks}, memory_start={memory_start}, row={row}, "
        f"summary_idx={summary_idx}, recent_idx={recent_idx}, summary_count={summary_count}, "
        f"summary_span=[{summary_start},{summary_end})"
    )
    for dim, label in enumerate(labels):
        print(f"{label}: mae={mae[dim]:.6f}, rmse={rmse[dim]:.6f}")


if __name__ == "__main__":
    main()
