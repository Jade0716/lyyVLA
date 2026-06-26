#!/usr/bin/env python3
"""Build offline DCT-bank caches for LIBERO LeRobot episodes.

The default cache is "sliding-start": for every episode start frame s, it
stores DCT summaries for chunks [s:s+32], [s+32:s+64], ... until episode end.
This matches training samples that build history backwards from the current
frame, e.g. t=90 uses chunks 26-57 and 58-89.

Example:
  conda run -n starVLA python examples/LIBERO/train_files/build_dct_bank_cache.py \
    --dataset-dir /path/to/libero_goal_no_noops_1.0.0_lerobot
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from tqdm import tqdm


DEFAULT_ACTION_COLUMNS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobot dataset directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory. Default: "
            "<dataset-dir>/meta/dct_bank_cache/chunk{N}_recent{R}_summary{S}_{mode}"
        ),
    )
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument(
        "--chunk-keep-freq",
        "--keep-freq",
        dest="chunk_keep_freq",
        type=int,
        default=4,
        help="Low-frequency DCT coefficients kept for each recent chunk.",
    )
    parser.add_argument(
        "--summary-keep-freq",
        type=int,
        default=8,
        help="Low-frequency DCT coefficients kept for recursive summary memory.",
    )
    parser.add_argument(
        "--mode",
        choices=("sliding-start", "global-chunks"),
        default="sliding-start",
        help="sliding-start stores summaries for every start frame; global-chunks stores starts 0, 32, 64, ...",
    )
    parser.add_argument(
        "--action-column",
        default="action",
        help="Parquet column containing vector actions. Used before --action-columns fallback.",
    )
    parser.add_argument(
        "--action-columns",
        default=",".join(DEFAULT_ACTION_COLUMNS),
        help="Comma-separated scalar action columns used if --action-column is absent.",
    )
    parser.add_argument(
        "--no-clip-actions",
        action="store_true",
        help="Do not clip actions to [-1, 1]. LIBERO training config currently clips action.*.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for cached arrays.",
    )
    parser.add_argument(
        "--no-store-coarse",
        action="store_true",
        help="Only store low-frequency DCT coefficients, not 32-step IDCT reconstructions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-episode cache files.",
    )
    return parser.parse_args()


def find_episode_parquets(dataset_dir: Path) -> list[Path]:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing LeRobot data directory: {data_dir}")
    return sorted(data_dir.glob("**/*.parquet"))


def infer_episode_index(path: Path, frame: pd.DataFrame) -> int:
    if "episode_index" in frame.columns:
        values = frame["episode_index"].dropna().unique()
        if len(values) == 1:
            return int(values[0])

    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not infer episode index for {path}")


def extract_actions(
    frame: pd.DataFrame,
    action_column: str,
    action_columns: Iterable[str],
    clip_actions: bool,
) -> np.ndarray:
    if action_column in frame.columns:
        values = frame[action_column].to_numpy()
        actions = np.asarray([np.asarray(v, dtype=np.float32) for v in values], dtype=np.float32)
    else:
        columns = [col.strip() for col in action_columns if col.strip()]
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise KeyError(
                f"Action column '{action_column}' not found and scalar columns missing: {missing[:8]}"
            )
        actions = frame[columns].to_numpy(dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T, D], got {actions.shape}")

    if clip_actions:
        actions = np.clip(actions, -1.0, 1.0)
    return actions


def low_dct(action_chunk: np.ndarray, chunk_len: int, keep_freq: int) -> np.ndarray:
    padded = np.zeros((chunk_len, action_chunk.shape[-1]), dtype=np.float32)
    valid = min(len(action_chunk), chunk_len)
    if valid > 0:
        padded[:valid] = action_chunk[:valid]
    return dct(padded, type=2, axis=0, norm="ortho")[:keep_freq].astype(np.float32)


def coarse_from_dct(coeff: np.ndarray, length: int, keep_freq: int | None = None) -> np.ndarray:
    if keep_freq is None:
        keep_freq = coeff.shape[0]
    padded = np.zeros((length, coeff.shape[-1]), dtype=np.float32)
    padded[:keep_freq] = coeff[:keep_freq]
    return idct(padded, type=2, n=length, axis=0, norm="ortho").astype(np.float32)


def merge_summary(
    summary_dct: np.ndarray,
    summary_count: int,
    new_chunk_dct: np.ndarray,
    chunk_len: int,
    chunk_keep_freq: int,
    summary_keep_freq: int,
) -> np.ndarray:
    prev_len = summary_count * chunk_len
    prev_coarse = coarse_from_dct(summary_dct, prev_len, summary_keep_freq)
    new_coarse = coarse_from_dct(new_chunk_dct, chunk_len, chunk_keep_freq)
    merged = np.concatenate([prev_coarse, new_coarse], axis=0)
    return dct(merged, type=2, axis=0, norm="ortho")[:summary_keep_freq].astype(np.float32)


def build_episode_cache(
    actions: np.ndarray,
    chunk_len: int,
    chunk_keep_freq: int,
    summary_keep_freq: int,
    mode: str,
    storage_dtype: np.dtype,
    store_coarse: bool,
) -> dict[str, np.ndarray]:
    episode_len, action_dim = actions.shape
    if episode_len <= 0:
        raise ValueError("Episode has no frames.")

    if mode == "sliding-start":
        start_indices = np.arange(episode_len, dtype=np.int64)
    else:
        start_indices = np.arange(0, episode_len, chunk_len, dtype=np.int64)

    max_chunks = int(math.ceil(episode_len / chunk_len))
    shape_prefix = (len(start_indices), max_chunks)
    chunk_dct = np.zeros(shape_prefix + (chunk_keep_freq, action_dim), dtype=np.float32)
    prefix_summary_dct = np.zeros(shape_prefix + (summary_keep_freq, action_dim), dtype=np.float32)
    chunk_valid_lengths = np.zeros(shape_prefix, dtype=np.int16)
    prefix_summary_count = np.zeros(shape_prefix, dtype=np.int16)
    num_chunks_per_start = np.zeros((len(start_indices),), dtype=np.int16)
    valid_mask = np.zeros(shape_prefix, dtype=np.bool_)

    chunk_coarse = None
    if store_coarse:
        chunk_coarse = np.zeros(shape_prefix + (chunk_len, action_dim), dtype=np.float32)

    for row, start in enumerate(start_indices):
        chunks_for_start = int(math.ceil((episode_len - int(start)) / chunk_len))
        num_chunks_per_start[row] = chunks_for_start

        summary = None
        summary_count = 0
        for chunk_idx in range(chunks_for_start):
            begin = int(start) + chunk_idx * chunk_len
            end = begin + chunk_len
            valid_len = max(0, min(end, episode_len) - begin)
            coeff = low_dct(actions[begin:end], chunk_len, chunk_keep_freq)

            if summary is None:
                summary = np.zeros((summary_keep_freq, action_dim), dtype=np.float32)
                copy_len = min(chunk_keep_freq, summary_keep_freq)
                summary[:copy_len] = coeff[:copy_len]
                summary_count = 1
            else:
                summary = merge_summary(
                    summary,
                    summary_count,
                    coeff,
                    chunk_len,
                    chunk_keep_freq,
                    summary_keep_freq,
                )
                summary_count += 1

            chunk_dct[row, chunk_idx] = coeff
            prefix_summary_dct[row, chunk_idx] = summary
            chunk_valid_lengths[row, chunk_idx] = valid_len
            prefix_summary_count[row, chunk_idx] = summary_count
            valid_mask[row, chunk_idx] = True
            if chunk_coarse is not None:
                chunk_coarse[row, chunk_idx] = coarse_from_dct(coeff, chunk_len, chunk_keep_freq)

    result = {
        "episode_length": np.asarray(episode_len, dtype=np.int32),
        "chunk_len": np.asarray(chunk_len, dtype=np.int32),
        "chunk_keep_freq": np.asarray(chunk_keep_freq, dtype=np.int32),
        "summary_keep_freq": np.asarray(summary_keep_freq, dtype=np.int32),
        "action_dim": np.asarray(action_dim, dtype=np.int32),
        "start_indices": start_indices,
        "num_chunks_per_start": num_chunks_per_start,
        "valid_mask": valid_mask,
        "chunk_valid_lengths": chunk_valid_lengths,
        "chunk_dct": chunk_dct.astype(storage_dtype),
        "prefix_summary_dct": prefix_summary_dct.astype(storage_dtype),
        "prefix_summary_count": prefix_summary_count,
    }
    if chunk_coarse is not None:
        result["chunk_coarse"] = chunk_coarse.astype(storage_dtype)
    return result


def iter_episode_frames(path: Path) -> Iterable[tuple[int, pd.DataFrame]]:
    frame = pd.read_parquet(path)
    if "episode_index" not in frame.columns or frame["episode_index"].nunique(dropna=True) <= 1:
        yield infer_episode_index(path, frame), frame
        return

    for episode_index, group in frame.groupby("episode_index", sort=True):
        yield int(episode_index), group.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        suffix = (
            f"chunk{args.chunk_len}_recent{args.chunk_keep_freq}_"
            f"summary{args.summary_keep_freq}_{args.mode}"
        )
        output_dir = dataset_dir / "meta" / "dct_bank_cache" / suffix
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_dtype = np.dtype(args.dtype)
    action_columns = tuple(args.action_columns.split(","))
    parquet_paths = find_episode_parquets(dataset_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    episodes_written = 0
    frames_seen = 0
    action_dim = None
    for parquet_path in tqdm(parquet_paths, desc="DCT bank cache"):
        for episode_index, frame in iter_episode_frames(parquet_path):
            output_path = output_dir / f"episode_{episode_index:06d}.npz"
            if output_path.exists() and not args.overwrite:
                continue

            actions = extract_actions(
                frame,
                action_column=args.action_column,
                action_columns=action_columns,
                clip_actions=not args.no_clip_actions,
            )
            cache = build_episode_cache(
                actions=actions,
                chunk_len=args.chunk_len,
                chunk_keep_freq=args.chunk_keep_freq,
                summary_keep_freq=args.summary_keep_freq,
                mode=args.mode,
                storage_dtype=storage_dtype,
                store_coarse=not args.no_store_coarse,
            )
            cache["episode_index"] = np.asarray(episode_index, dtype=np.int32)
            np.savez_compressed(output_path, **cache)
            episodes_written += 1
            frames_seen += len(actions)
            action_dim = actions.shape[-1]

    metadata = {
        "dataset_dir": str(dataset_dir),
        "mode": args.mode,
        "chunk_len": args.chunk_len,
        "chunk_keep_freq": args.chunk_keep_freq,
        "summary_keep_freq": args.summary_keep_freq,
        "action_column": args.action_column,
        "action_columns": list(action_columns),
        "clip_actions": not args.no_clip_actions,
        "dtype": args.dtype,
        "store_coarse": not args.no_store_coarse,
        "num_parquet_files": len(parquet_paths),
        "episodes_written_this_run": episodes_written,
        "frames_written_this_run": frames_seen,
        "action_dim": action_dim,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {episodes_written} episode caches to {output_dir}")


if __name__ == "__main__":
    main()
