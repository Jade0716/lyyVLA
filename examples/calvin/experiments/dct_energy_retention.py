#!/usr/bin/env python3
"""Measure how much CALVIN action energy is retained by low-frequency DCT terms.

The twochunk model uses a 16-step action window and predicts the low-frequency
motion trend. With an orthonormal DCT, the squared coefficient energy is equal
to the squared signal energy, so the retained-energy ratio is:

    sum(dct_coeff[:K] ** 2) / sum(dct_coeff[:16] ** 2)

This script reports that ratio for K=4 and K=8 by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from tqdm import tqdm


DEFAULT_ACTION_KEYS = ("action.delta_joints", "action.gripper_close")
FALLBACK_ACTION_KEYS = ("action", "relative_action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/16T/liuyuyan/calvin-abc-d-lerobot"))
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--keep-dims", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--action-keys", type=str, nargs="+", default=list(DEFAULT_ACTION_KEYS))
    parser.add_argument(
        "--fallback-action-keys",
        type=str,
        nargs="+",
        default=list(FALLBACK_ACTION_KEYS),
        help="Single parquet columns to try when --action-keys are unavailable.",
    )
    parser.add_argument(
        "--drop-gripper",
        action="store_true",
        help="Drop the last action dimension, useful when inspecting only continuous arm action.",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--out", type=Path, default=Path("tmp/calvin/dct_energy_retention.json"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def stack_column(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        raise ValueError("empty parquet column")
    first = np.asarray(values[0])
    if first.ndim == 0:
        return values.astype(np.float32).reshape(-1, 1)
    return np.stack(values).astype(np.float32)


def read_action_array(parquet_path: Path, action_keys: list[str], fallback_action_keys: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()

    if all(key in columns for key in action_keys):
        df = pd.read_parquet(parquet_path, columns=action_keys)
        parts = [stack_column(df[key]) for key in action_keys]
        return np.concatenate(parts, axis=-1), action_keys

    for key in fallback_action_keys:
        if key in columns:
            df = pd.read_parquet(parquet_path, columns=[key])
            return stack_column(df[key]), [key]

    raise KeyError(
        f"None of action keys were found in {parquet_path}. "
        f"requested={action_keys}, fallback={fallback_action_keys}, available={columns}"
    )


def iter_chunks(actions: np.ndarray, chunk_len: int, stride: int, start_offset: int):
    last_start = len(actions) - chunk_len
    if last_start < start_offset:
        return
    for start in range(start_offset, last_start + 1, stride):
        yield actions[start : start + chunk_len]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be positive")
    if args.chunk_stride <= 0:
        raise ValueError("--chunk-stride must be positive")
    if any(k <= 0 or k > args.chunk_len for k in args.keep_dims):
        raise ValueError(f"--keep-dims must be in [1, {args.chunk_len}], got {args.keep_dims}")

    episodes = load_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    if args.num_episodes is not None:
        episodes = episodes[: args.num_episodes]

    skipped = Counter()
    action_key_counts = Counter()
    energy_ratios = {k: [] for k in args.keep_dims}
    mean_dim_energy_ratios = {k: [] for k in args.keep_dims}
    reconstruction_mse = {k: [] for k in args.keep_dims}
    per_dim_energy_num = {k: None for k in args.keep_dims}
    per_dim_energy_den = None
    chunks_seen = 0
    action_dim = None

    progress = tqdm(episodes, desc="Scanning CALVIN action chunks")
    for ep in progress:
        if args.max_chunks is not None and chunks_seen >= args.max_chunks:
            break

        episode_index = int(ep["episode_index"])
        parquet_path = episode_path(args.dataset_root, episode_index)
        if not parquet_path.exists():
            skipped["missing_parquet"] += 1
            continue

        try:
            actions, used_keys = read_action_array(parquet_path, args.action_keys, args.fallback_action_keys)
        except Exception as exc:
            skipped[type(exc).__name__] += 1
            continue

        if args.drop_gripper:
            actions = actions[:, :-1]
        if action_dim is None:
            action_dim = int(actions.shape[-1])
            per_dim_energy_den = np.zeros(action_dim, dtype=np.float64)
            for k in args.keep_dims:
                per_dim_energy_num[k] = np.zeros(action_dim, dtype=np.float64)

        action_key_counts[",".join(used_keys)] += 1
        for chunk in iter_chunks(actions, args.chunk_len, args.chunk_stride, args.start_offset):
            if args.max_chunks is not None and chunks_seen >= args.max_chunks:
                break

            coeff = dct(chunk.astype(np.float32), type=2, axis=0, norm="ortho")
            energy_by_dim = np.square(coeff, dtype=np.float64).sum(axis=0)
            total_energy = float(energy_by_dim.sum())
            if total_energy <= args.eps:
                skipped["zero_energy_chunk"] += 1
                continue

            per_dim_energy_den += energy_by_dim
            for k in args.keep_dims:
                kept = np.square(coeff[:k], dtype=np.float64).sum(axis=0)
                per_dim_energy_num[k] += kept
                energy_ratios[k].append(float(kept.sum() / (total_energy + args.eps)))
                mean_dim_energy_ratios[k].append(float(np.mean(kept / (energy_by_dim + args.eps))))

                truncated = np.zeros_like(coeff)
                truncated[:k] = coeff[:k]
                recon = idct(truncated, type=2, axis=0, norm="ortho")
                reconstruction_mse[k].append(float(np.mean((recon - chunk) ** 2)))

            chunks_seen += 1
        progress.set_postfix(chunks=chunks_seen)

    if chunks_seen == 0:
        raise RuntimeError(f"No valid chunks were found. skipped={dict(skipped)}")

    results = {
        "dataset_root": str(args.dataset_root),
        "chunk_len": args.chunk_len,
        "chunk_stride": args.chunk_stride,
        "start_offset": args.start_offset,
        "num_episodes_scanned": len(episodes),
        "num_chunks": chunks_seen,
        "action_dim": action_dim,
        "action_key_counts": dict(action_key_counts),
        "drop_gripper": args.drop_gripper,
        "skipped": dict(skipped),
        "metrics": {},
    }

    for k in args.keep_dims:
        dim_ratio = per_dim_energy_num[k] / (per_dim_energy_den + args.eps)
        results["metrics"][str(k)] = {
            "global_energy_ratio": summarize(energy_ratios[k]),
            "mean_dim_energy_ratio_per_chunk": summarize(mean_dim_energy_ratios[k]),
            "dataset_level_per_dim_energy_ratio": [float(x) for x in dim_ratio],
            "dataset_level_mean_dim_energy_ratio": float(dim_ratio.mean()),
            "reconstruction_mse": summarize(reconstruction_mse[k]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
