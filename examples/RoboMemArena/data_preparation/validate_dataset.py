#!/usr/bin/env python3
"""Validate a RoboMemArena LeRobot v2 conversion.

Validation covers metadata consistency, Parquet schemas, temporal indices,
action/state ranges, MP4 properties, and optional pixel-level comparison
against the source HDF5 files recorded in the conversion provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import pandas as pd


VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
REQUIRED_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate converted RoboMemArena LeRobot data.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--expected-tasks",
        default="1-26",
        help="Expected RoboMemArena task IDs, e.g. '1-26' or '1-5,10'.",
    )
    parser.add_argument(
        "--strict-task-coverage",
        action="store_true",
        help="Treat missing expected tasks as errors instead of warnings.",
    )
    parser.add_argument(
        "--video-samples",
        type=int,
        default=10,
        help="Number of evenly spaced episodes for MP4 and source-image checks; 0 disables.",
    )
    parser.add_argument(
        "--source-frame-samples",
        type=int,
        default=3,
        help="Frames per sampled video to compare with source HDF5.",
    )
    parser.add_argument(
        "--max-source-mae",
        type=float,
        default=12.0,
        help="Maximum mean absolute pixel error allowed after lossy video encoding.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser.parse_args()


def parse_task_spec(spec: str) -> set[int]:
    tasks: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            tasks.update(range(int(start_text), int(end_text) + 1))
        else:
            tasks.add(int(item))
    return tasks


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"ERROR: {message}", file=sys.stderr)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr)

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.error(message)


def ffprobe_video(path: Path, ffprobe_bin: str) -> dict[str, object]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, got {streams}")
    return streams[0]


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, min(count, length), dtype=int).tolist()))


def validate_parquet(
    *,
    path: Path,
    episode_index: int,
    expected_length: int,
    expected_task_index: int,
    expected_start_index: int,
    fps: float,
    report: Report,
) -> tuple[int, dict[str, np.ndarray]]:
    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        report.error(f"Failed to read {path}: {exc}")
        return expected_start_index, {}

    report.check(tuple(dataframe.columns) == REQUIRED_COLUMNS, f"Unexpected columns in {path}")
    report.check(len(dataframe) == expected_length, f"Length mismatch in {path}")
    if len(dataframe) == 0:
        return expected_start_index, {}

    arrays: dict[str, np.ndarray] = {}
    for column, width in (("observation.state", 8), ("action", 7)):
        try:
            array = np.stack(dataframe[column].to_numpy()).astype(np.float32)
            arrays[column] = array
            report.check(array.shape == (expected_length, width), f"{column} shape mismatch in {path}")
            report.check(np.isfinite(array).all(), f"Non-finite {column} values in {path}")
        except Exception as exc:
            report.error(f"Cannot stack {column} in {path}: {exc}")

    frame_index = dataframe["frame_index"].to_numpy()
    episode_values = dataframe["episode_index"].to_numpy()
    global_index = dataframe["index"].to_numpy()
    task_values = dataframe["task_index"].to_numpy()
    timestamp = dataframe["timestamp"].to_numpy(dtype=np.float64)
    report.check(np.array_equal(frame_index, np.arange(expected_length)), f"Bad frame_index in {path}")
    report.check(np.all(episode_values == episode_index), f"Bad episode_index in {path}")
    report.check(
        np.array_equal(global_index, np.arange(expected_start_index, expected_start_index + expected_length)),
        f"Bad global index in {path}",
    )
    report.check(np.all(task_values == expected_task_index), f"Bad task_index in {path}")
    report.check(
        np.allclose(timestamp, np.arange(expected_length) / fps, atol=2e-5),
        f"Bad timestamps in {path}",
    )
    return expected_start_index + expected_length, arrays


def validate_source_frames(
    *,
    video_path: Path,
    provenance: dict[str, object],
    source_key: str,
    frame_count: int,
    frame_samples: int,
    max_source_mae: float,
    report: Report,
) -> None:
    source_path = Path(str(provenance["source_path"]))
    demo_key = str(provenance["demo_key"])
    if not source_path.exists():
        report.warn(f"Source file unavailable for pixel comparison: {source_path}")
        return
    indices = evenly_spaced_indices(frame_count, frame_samples)
    with h5py.File(source_path, "r") as handle:
        source_dataset = handle[f"data/{demo_key}/obs/{source_key}"]
        report.check(len(source_dataset) == frame_count, f"Source frame count mismatch: {source_path}")
        for index in indices:
            source_rgb = np.asarray(source_dataset[index], dtype=np.uint8)
            decoded_rgb = read_video_frame(video_path, index)
            direct_mae = float(np.abs(decoded_rgb.astype(np.int16) - source_rgb.astype(np.int16)).mean())
            flipped_mae = float(
                np.abs(decoded_rgb.astype(np.int16) - source_rgb[::-1].astype(np.int16)).mean()
            )
            report.check(
                direct_mae <= max_source_mae,
                f"Pixel MAE {direct_mae:.2f} exceeds {max_source_mae} for {video_path} frame {index}",
            )
            report.check(
                direct_mae < flipped_mae,
                f"Video appears vertically flipped relative to source: {video_path} frame {index}",
            )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    meta_dir = dataset_dir / "meta"
    report = Report()
    required_meta = (
        "info.json",
        "modality.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "robomemarena_provenance.jsonl",
        "conversion_summary.json",
    )
    for filename in required_meta:
        report.check((meta_dir / filename).is_file(), f"Missing metadata file: {meta_dir / filename}")
    if report.errors:
        raise SystemExit(1)

    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    modality = json.loads((meta_dir / "modality.json").read_text(encoding="utf-8"))
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    episodes = read_jsonl(meta_dir / "episodes.jsonl")
    provenance = read_jsonl(meta_dir / "robomemarena_provenance.jsonl")
    summary = json.loads((meta_dir / "conversion_summary.json").read_text(encoding="utf-8"))

    report.check(info.get("codebase_version") == "v2.1", "Expected LeRobot codebase_version v2.1")
    report.check(info.get("total_episodes") == len(episodes), "total_episodes mismatch")
    report.check(len(episodes) == len(provenance), "Episode/provenance count mismatch")
    report.check(info.get("total_tasks") == len(tasks), "total_tasks mismatch")
    report.check(info.get("total_videos") == len(episodes) * 2, "total_videos mismatch")
    report.check(summary.get("image_transform") == "none", "Image transform must be 'none'")
    report.check(
        set(modality) == {"state", "action", "video", "annotation"},
        "Unexpected modality.json top-level keys",
    )
    report.check(len(modality.get("state", {})) == 8, "Expected 8 state fields")
    report.check(len(modality.get("action", {})) == 7, "Expected 7 action fields")

    task_prompt_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    report.check(len(task_prompt_by_index) == len(tasks), "Duplicate task_index in tasks.jsonl")
    task_index_by_prompt = {prompt: task_index for task_index, prompt in task_prompt_by_index.items()}
    report.check(len(task_index_by_prompt) == len(tasks), "Duplicate task prompt in tasks.jsonl")
    for task_index, prompt in task_prompt_by_index.items():
        report.check(bool(prompt.strip()), f"Empty task prompt for task_index={task_index}")

    converted_task_ids = {int(row["task_id"]) for row in provenance}
    expected_tasks = parse_task_spec(args.expected_tasks)
    missing_tasks = sorted(expected_tasks - converted_task_ids)
    if missing_tasks:
        message = f"Missing expected RoboMemArena tasks: {missing_tasks}"
        if args.strict_task_coverage:
            report.error(message)
        else:
            report.warn(message)

    episode_counts = Counter(int(row["task_id"]) for row in provenance)
    frame_counts: Counter[int] = Counter()
    global_index = 0
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)
    state_min = np.full(8, np.inf, dtype=np.float64)
    state_max = np.full(8, -np.inf, dtype=np.float64)
    chunk_size = int(info["chunks_size"])
    fps = float(info["fps"])

    for position, (episode, source) in enumerate(zip(episodes, provenance, strict=True)):
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        task_id = int(source["task_id"])
        episode_tasks = episode.get("tasks")
        report.check(isinstance(episode_tasks, list) and len(episode_tasks) == 1, "Episode must have exactly one task")
        episode_prompt = str(episode_tasks[0]) if isinstance(episode_tasks, list) and episode_tasks else ""
        expected_task_index = task_index_by_prompt.get(episode_prompt)
        report.check(episode_index == position, f"Non-contiguous episode index at position {position}")
        report.check(int(source["episode_index"]) == episode_index, "Provenance episode index mismatch")
        report.check(int(source["length"]) == length, "Provenance length mismatch")
        report.check(source.get("prompt") == episode_prompt, "Provenance prompt mismatch")
        report.check(expected_task_index is not None, "Episode task missing from tasks.jsonl")
        if "task_index" in episode:
            report.check(int(episode["task_index"]) == expected_task_index, "Episode task_index mismatch")
        if "source_task_id" in episode:
            report.check(int(episode["source_task_id"]) == task_id, "Episode source_task_id mismatch")
        chunk_index = episode_index // chunk_size
        parquet_path = (
            dataset_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        )
        report.check(parquet_path.is_file(), f"Missing parquet: {parquet_path}")
        if parquet_path.is_file() and expected_task_index is not None:
            global_index, arrays = validate_parquet(
                path=parquet_path,
                episode_index=episode_index,
                expected_length=length,
                expected_task_index=expected_task_index,
                expected_start_index=global_index,
                fps=fps,
                report=report,
            )
            if "action" in arrays:
                action_min = np.minimum(action_min, arrays["action"].min(axis=0))
                action_max = np.maximum(action_max, arrays["action"].max(axis=0))
            if "observation.state" in arrays:
                state_min = np.minimum(state_min, arrays["observation.state"].min(axis=0))
                state_max = np.maximum(state_max, arrays["observation.state"].max(axis=0))
        frame_counts[task_id] += length

    report.check(global_index == int(info["total_frames"]), "total_frames/global index mismatch")

    sampled_episode_indices = evenly_spaced_indices(len(episodes), args.video_samples)
    provenance_by_episode = {int(row["episode_index"]): row for row in provenance}
    for episode_index in sampled_episode_indices:
        length = int(episodes[episode_index]["length"])
        chunk_index = episode_index // chunk_size
        for video_key, source_key in zip(
            VIDEO_KEYS, ("agentview_rgb", "eye_in_hand_rgb"), strict=True
        ):
            video_path = (
                dataset_dir
                / f"videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            )
            report.check(video_path.is_file(), f"Missing video: {video_path}")
            if not video_path.is_file():
                continue
            try:
                stream = ffprobe_video(video_path, args.ffprobe_bin)
                report.check(int(stream["width"]) == 256 and int(stream["height"]) == 256, f"Bad video size: {video_path}")
                report.check(stream.get("pix_fmt") == "yuv420p", f"Bad pixel format: {video_path}")
                if stream.get("nb_frames") not in (None, "N/A"):
                    report.check(int(stream["nb_frames"]) == length, f"Bad frame count: {video_path}")
                numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
                probed_fps = float(numerator) / float(denominator)
                report.check(math.isclose(probed_fps, fps), f"Bad FPS {probed_fps}: {video_path}")
                validate_source_frames(
                    video_path=video_path,
                    provenance=provenance_by_episode[episode_index],
                    source_key=source_key,
                    frame_count=length,
                    frame_samples=args.source_frame_samples,
                    max_source_mae=args.max_source_mae,
                    report=report,
                )
            except Exception as exc:
                report.error(f"Video validation failed for {video_path}: {exc}")

    print("\nDataset summary")
    print(f"  path: {dataset_dir}")
    print(f"  episodes: {len(episodes)}")
    print(f"  frames: {info['total_frames']}")
    print(f"  tasks: {sorted(converted_task_ids)}")
    print(f"  episodes/task: {dict(sorted(episode_counts.items()))}")
    print(f"  frames/task: {dict(sorted(frame_counts.items()))}")
    print(f"  action min: {action_min.tolist()}")
    print(f"  action max: {action_max.tolist()}")
    print(f"  state min: {state_min.tolist()}")
    print(f"  state max: {state_max.tolist()}")
    print(f"  warnings: {len(report.warnings)}")
    print(f"  errors: {len(report.errors)}")
    if report.errors:
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
