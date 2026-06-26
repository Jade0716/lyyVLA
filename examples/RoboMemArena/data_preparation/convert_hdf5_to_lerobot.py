#!/usr/bin/env python3
"""Convert RoboMemArena full-trajectory HDF5 files to LeRobot v2 format.

The converter intentionally preserves the benchmark controls and images:

* action: ``actions`` as float32, without clipping or remapping;
* state: ``ee_states`` (6) + ``gripper_states`` (2);
* images: RGB frames as stored in HDF5, without vertical flipping;
* episode: one HDF5 demo is one LeRobot episode.

The generated layout is directly consumable by lyyVLA's built-in
``gr00t_lerobot`` loader and does not require the external ``lerobot`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import pandas as pd


FULL_TRAJECTORY_RE = re.compile(r"_seed(?P<seed>\d+)_task(?P<task_id>\d+)\.hdf5$")
EXPECTED_TASKS = tuple(range(1, 27))
VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
# The June 2026 Task 5 refresh targets the middle drawer. Some local mirrors
# retain 20 older bottom-drawer trajectories in the same folder; those are not
# demonstrations of the current Task 5 BDDL and must not enter training.
STALE_FILENAME_PATTERNS = {5: ("bottom_drawer",)}
STATE_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "pad",
    "gripper",
)
ACTION_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
)


@dataclass(frozen=True)
class SourceDemo:
    path: Path
    demo_key: str
    task_id: int
    seed: int


class ConversionProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.started_at = time.monotonic()
        self.interactive = sys.stderr.isatty()
        self.last_width = 0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def update(self, *, completed: int | None = None, status: str = "") -> None:
        if completed is not None:
            self.completed = completed
        if not self.interactive:
            return

        elapsed = time.monotonic() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.completed) / rate if rate > 0 else 0.0
        terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        bar_width = max(10, min(40, terminal_width - 58))
        fraction = self.completed / self.total if self.total else 1.0
        filled = min(bar_width, int(bar_width * fraction))
        bar = "#" * filled + "-" * (bar_width - filled)
        timing = f"elapsed {self._format_duration(elapsed)}"
        if self.completed:
            timing += f", eta {self._format_duration(remaining)}"
        prefix = (
            f"\r[{bar}] {self.completed:>{len(str(self.total))}}/{self.total} "
            f"{fraction:6.1%} | {timing} | "
        )
        status_width = max(0, terminal_width - len(prefix))
        line = prefix + status[:status_width]
        print(
            line + " " * max(0, self.last_width - len(line)),
            end="",
            file=sys.stderr,
            flush=True,
        )
        self.last_width = len(line)

    def finish(self) -> None:
        if self.interactive:
            print(file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboMemArena full_trajectory HDF5 files to LeRobot v2."
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help=(
            "Dataset root to scan recursively. Pass once per downloaded category, "
            "or pass /16T/liuyuyan to discover all RoboMemArena categories."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Destination LeRobot dataset directory.")
    parser.add_argument(
        "--bddl-dir",
        default=None,
        help="RoboMemArena bddl directory. Required when --prompt-source=bddl.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=("bddl", "hdf5"),
        default="bddl",
        help="Use evaluation-aligned BDDL stem prompts (default) or HDF5 language attributes.",
    )
    parser.add_argument(
        "--tasks",
        default="1-26",
        help="Comma/range task selection, e.g. '1-3,10,15-26'.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional deterministic cap after sorting by task, seed, path, and demo.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", default="veryfast")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--hash-sources",
        action="store_true",
        help="Compute full SHA-256 hashes for source HDF5 files (slow on this large dataset).",
    )
    parser.add_argument(
        "--strict-task-coverage",
        action="store_true",
        help="Fail if any selected task has no complete trajectory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output-dir if it already exists.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a live episode conversion progress bar when stderr is a terminal.",
    )
    return parser.parse_args()


def parse_task_spec(spec: str) -> tuple[int, ...]:
    tasks: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending task range: {item}")
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(item))
    if not tasks or min(tasks) < 1 or max(tasks) > 26:
        raise ValueError(f"Tasks must be within 1..26, got {sorted(tasks)}")
    return tuple(sorted(tasks))


def discover_sources(source_roots: Iterable[Path], selected_tasks: set[int]) -> list[SourceDemo]:
    demos: list[SourceDemo] = []
    seen_files: set[Path] = set()
    stale_files: list[Path] = []
    for root in source_roots:
        if not root.exists():
            raise FileNotFoundError(f"Source root does not exist: {root}")
        for path in root.glob("**/full_trajectory/*.hdf5"):
            if "._____temp" in path.parts:
                continue
            path = path.resolve()
            if path in seen_files:
                continue
            seen_files.add(path)
            match = FULL_TRAJECTORY_RE.search(path.name)
            if match is None:
                print(f"WARNING: skipping unrecognized full-trajectory filename: {path}", file=sys.stderr)
                continue
            task_id = int(match.group("task_id"))
            seed = int(match.group("seed"))
            if task_id not in selected_tasks:
                continue
            if any(pattern in path.name for pattern in STALE_FILENAME_PATTERNS.get(task_id, ())):
                stale_files.append(path)
                continue
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise ValueError(f"Missing /data group: {path}")
                for demo_key in sorted(handle["data"].keys()):
                    if not demo_key.startswith("demo_"):
                        continue
                    demos.append(SourceDemo(path=path, demo_key=demo_key, task_id=task_id, seed=seed))
    demos.sort(key=lambda item: (item.task_id, item.seed, str(item.path), item.demo_key))
    if stale_files:
        print(
            f"WARNING: excluded {len(stale_files)} stale trajectories that conflict with "
            f"the current task definitions; first example: {stale_files[0]}",
            file=sys.stderr,
        )
    return demos


def load_bddl_prompts(bddl_dir: Path, selected_tasks: Iterable[int]) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for task_id in selected_tasks:
        matches = sorted(bddl_dir.glob(f"{task_id}_*.bddl"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one BDDL for task {task_id} under {bddl_dir}, found {matches}"
            )
        stem = matches[0].stem
        prompts[task_id] = stem.split("_", 1)[1].replace("_", " ")
    return prompts


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _require_array(group: h5py.Group, key: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Missing dataset {group.name}/{key}")
    array = np.asarray(group[key])
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(
            f"Unexpected shape for {group.name}/{key}: {array.shape}; expected [T, {shape_tail}]"
        )
    return array


def load_demo(source: SourceDemo) -> dict[str, object]:
    with h5py.File(source.path, "r") as handle:
        demo = handle[f"data/{source.demo_key}"]
        obs = demo["obs"]
        actions = _require_array(demo, "actions", (7,)).astype(np.float32)
        ee_states = _require_array(obs, "ee_states", (6,)).astype(np.float32)
        gripper_states = _require_array(obs, "gripper_states", (2,)).astype(np.float32)
        primary = _require_array(obs, "agentview_rgb", (256, 256, 3))
        wrist = _require_array(obs, "eye_in_hand_rgb", (256, 256, 3))
        hdf5_instruction = demo.attrs.get("language_instruction", "")
        if isinstance(hdf5_instruction, bytes):
            hdf5_instruction = hdf5_instruction.decode("utf-8")
        hdf5_instruction = str(hdf5_instruction).strip()
        declared_samples = int(demo.attrs.get("num_samples", len(actions)))

    lengths = {
        "actions": len(actions),
        "ee_states": len(ee_states),
        "gripper_states": len(gripper_states),
        "primary": len(primary),
        "wrist": len(wrist),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Time dimension mismatch in {source.path}: {lengths}")
    if declared_samples != len(actions):
        raise ValueError(
            f"num_samples mismatch in {source.path}: attr={declared_samples}, actual={len(actions)}"
        )
    if len(actions) == 0:
        raise ValueError(f"Empty trajectory: {source.path}")
    if primary.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise TypeError(
            f"Images must be uint8 in {source.path}, got {primary.dtype} and {wrist.dtype}"
        )
    for name, array in (
        ("actions", actions),
        ("ee_states", ee_states),
        ("gripper_states", gripper_states),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite {name} in {source.path}")

    state = np.concatenate((ee_states, gripper_states), axis=-1).astype(np.float32)
    return {
        "actions": actions,
        "state": state,
        "primary": primary,
        "wrist": wrist,
        "hdf5_instruction": hdf5_instruction,
    }


def encode_video(
    frames: np.ndarray,
    output_path: Path,
    *,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
    ffmpeg_bin: str,
) -> None:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB [T,H,W,3], got {frames.shape} {frames.dtype}")
    height, width = frames.shape[1:3]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not created")
        for start in range(0, len(frames), 64):
            process.stdin.write(np.ascontiguousarray(frames[start : start + 64]).tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        temp_path.replace(output_path)
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temp_path.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_modality() -> dict[str, object]:
    state = {
        name: {
            "start": index,
            "end": index + 1,
            "absolute": True,
            "dtype": "float32",
            "original_key": "observation.state",
        }
        for index, name in enumerate(STATE_NAMES)
    }
    for name in ("roll", "pitch", "yaw"):
        state[name]["rotation_type"] = "axis_angle"

    action = {
        name: {
            "start": index,
            "end": index + 1,
            "absolute": False,
            "dtype": "float32",
            "original_key": "action",
        }
        for index, name in enumerate(ACTION_NAMES)
    }
    for name in ("roll", "pitch", "yaw"):
        action[name]["rotation_type"] = "axis_angle"

    return {
        "state": state,
        "action": action,
        "video": {
            "primary_image": {"original_key": VIDEO_KEYS[0]},
            "wrist_image": {"original_key": VIDEO_KEYS[1]},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def build_info(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_chunks: int,
    chunk_size: int,
    fps: int,
    codec: str,
) -> dict[str, object]:
    features: dict[str, object] = {}
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {"motors": list(STATE_NAMES)},
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": list(ACTION_NAMES)},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }


def main() -> None:
    args = parse_args()
    selected_tasks = parse_task_spec(args.tasks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_roots = [Path(item).expanduser().resolve() for item in args.source_root]
    bddl_dir = Path(args.bddl_dir).expanduser().resolve() if args.bddl_dir else None

    if args.prompt_source == "bddl" and bddl_dir is None:
        raise ValueError("--bddl-dir is required when --prompt-source=bddl")
    if args.fps <= 0 or args.chunk_size <= 0:
        raise ValueError("--fps and --chunk-size must be positive")
    if shutil.which(args.ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg_bin}")

    sources = discover_sources(source_roots, set(selected_tasks))
    available_tasks = sorted({source.task_id for source in sources})
    missing_tasks = sorted(set(selected_tasks) - set(available_tasks))
    print(f"Discovered {len(sources)} full-trajectory demos for tasks {available_tasks}")
    if missing_tasks:
        message = f"Selected tasks without complete trajectories: {missing_tasks}"
        if args.strict_task_coverage:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        sources = sources[: args.max_episodes]
    if not sources:
        raise RuntimeError("No full-trajectory HDF5 demos found")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)

    bddl_prompts = (
        load_bddl_prompts(bddl_dir, selected_tasks)
        if args.prompt_source == "bddl" and bddl_dir is not None
        else {}
    )
    selected_present_tasks = sorted({source.task_id for source in sources})
    task_index_by_id = {task_id: task_id - 1 for task_id in selected_present_tasks}
    tasks_rows: list[dict[str, object]] = []
    episodes_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    total_frames = 0
    source_hashes: dict[Path, str] = {}
    progress = ConversionProgress(len(sources)) if args.progress else None

    for episode_index, source in enumerate(sources):
        status_prefix = f"task={source.task_id} seed={source.seed}"
        if progress is not None:
            progress.update(status=f"{status_prefix} loading")
        payload = load_demo(source)
        actions = payload["actions"]
        state = payload["state"]
        primary = payload["primary"]
        wrist = payload["wrist"]
        assert isinstance(actions, np.ndarray)
        assert isinstance(state, np.ndarray)
        assert isinstance(primary, np.ndarray)
        assert isinstance(wrist, np.ndarray)
        hdf5_instruction = str(payload["hdf5_instruction"])
        prompt = bddl_prompts[source.task_id] if args.prompt_source == "bddl" else hdf5_instruction
        if not prompt:
            raise ValueError(f"Empty prompt for {source.path}:{source.demo_key}")

        task_index = task_index_by_id[source.task_id]
        chunk_index = episode_index // args.chunk_size
        length = len(actions)
        global_indices = np.arange(total_frames, total_frames + length, dtype=np.int64)
        frame_indices = np.arange(length, dtype=np.int64)
        dataframe = pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(actions),
                "timestamp": (frame_indices / float(args.fps)).astype(np.float32),
                "frame_index": frame_indices,
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": global_indices,
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        )
        parquet_path = (
            output_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress.update(status=f"{status_prefix} parquet ({length} frames)")
        dataframe.to_parquet(parquet_path, index=False)

        for video_number, (video_key, frames) in enumerate(
            zip(VIDEO_KEYS, (primary, wrist), strict=True), start=1
        ):
            video_path = (
                output_dir
                / f"videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            )
            if progress is not None:
                progress.update(
                    status=f"{status_prefix} video {video_number}/{len(VIDEO_KEYS)}"
                )
            encode_video(
                frames,
                video_path,
                fps=args.fps,
                codec=args.video_codec,
                crf=args.video_crf,
                preset=args.video_preset,
                ffmpeg_bin=args.ffmpeg_bin,
            )

        if args.hash_sources and source.path not in source_hashes:
            if progress is not None:
                progress.update(status=f"{status_prefix} hashing source")
            source_hashes[source.path] = sha256_file(source.path)
        episodes_rows.append(
            {"episode_index": episode_index, "tasks": [prompt], "length": length}
        )
        provenance_rows.append(
            {
                "episode_index": episode_index,
                "task_id": source.task_id,
                "seed": source.seed,
                "source_path": str(source.path),
                "source_size_bytes": source.path.stat().st_size,
                "source_mtime_ns": source.path.stat().st_mtime_ns,
                "source_sha256": source_hashes.get(source.path),
                "demo_key": source.demo_key,
                "length": length,
                "prompt": prompt,
                "prompt_source": args.prompt_source,
                "hdf5_language_instruction": hdf5_instruction,
                "image_transform": "none",
                "state_mapping": "concat(obs/ee_states[6], obs/gripper_states[2])",
                "action_mapping": "data/demo/actions[7], unchanged except float32 cast",
            }
        )
        total_frames += length
        if progress is not None:
            progress.update(
                completed=episode_index + 1,
                status=f"{status_prefix} done | {total_frames} frames total",
            )
        if progress is None or not progress.interactive:
            print(
                f"[{episode_index + 1}/{len(sources)}] task={source.task_id} "
                f"seed={source.seed} frames={length} total_frames={total_frames}"
            )

    if progress is not None:
        progress.finish()

    for task_id in selected_present_tasks:
        task_index = task_index_by_id[task_id]
        source_for_task = next(source for source in sources if source.task_id == task_id)
        if args.prompt_source == "bddl":
            prompt = bddl_prompts[task_id]
        else:
            prompt = str(load_demo(source_for_task)["hdf5_instruction"])
        tasks_rows.append({"task_index": task_index, "task": prompt})

    total_chunks = (len(sources) + args.chunk_size - 1) // args.chunk_size
    info = build_info(
        total_episodes=len(sources),
        total_frames=total_frames,
        total_tasks=len(tasks_rows),
        total_chunks=total_chunks,
        chunk_size=args.chunk_size,
        fps=args.fps,
        codec=args.video_codec,
    )
    (output_dir / "meta/info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "meta/modality.json").write_text(
        json.dumps(build_modality(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(output_dir / "meta/tasks.jsonl", tasks_rows)
    write_jsonl(output_dir / "meta/episodes.jsonl", episodes_rows)
    write_jsonl(output_dir / "meta/robomemarena_provenance.jsonl", provenance_rows)
    (output_dir / "meta/conversion_summary.json").write_text(
        json.dumps(
            {
                "format": "lerobot-v2",
                "selected_tasks": list(selected_tasks),
                "converted_tasks": selected_present_tasks,
                "missing_tasks": missing_tasks,
                "total_episodes": len(sources),
                "total_frames": total_frames,
                "fps": args.fps,
                "video_codec": args.video_codec,
                "video_crf": args.video_crf,
                "video_preset": args.video_preset,
                "prompt_source": args.prompt_source,
                "image_transform": "none",
                "source_roots": [str(path) for path in source_roots],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed LeRobot v2 dataset: {output_dir}")


if __name__ == "__main__":
    main()
