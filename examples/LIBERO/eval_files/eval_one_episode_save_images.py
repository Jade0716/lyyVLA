"""Run one LIBERO episode through the websocket model client and save every frame.

Example:
    python examples/LIBERO/eval_files/eval_one_episode_save_images.py \
        --args.host 127.0.0.1 \
        --args.port 6694 \
        --args.task-suite-name libero_10 \
        --args.task-id 0 \
        --args.init-state-id 0 \
        --args.output-dir ./tmp/libero_one_episode

The output contains ``agentview/*.png`` and ``wrist/*.png``. Images are rotated
180 degrees, exactly as they are before being sent to the policy server.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import pathlib

import imageio.v2 as imageio
import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from examples.LIBERO.eval_files.model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    task_suite_name: str = "libero_10"
    task_id: int = 0
    init_state_id: int = 0
    output_dir: str = "./tmp/libero_one_episode"
    num_steps_wait: int = 10
    seed: int = 7
    unnorm_key: str | None = None
    inference_warmup_steps: int = 0


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / den


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    value = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


def _policy_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agentview, wrist


def _save_frame(obs: dict, output_dir: pathlib.Path, frame_id: int) -> None:
    agentview, wrist = _policy_images(obs)
    imageio.imwrite(output_dir / "agentview" / f"{frame_id:06d}.png", agentview)
    imageio.imwrite(output_dir / "wrist" / f"{frame_id:06d}.png", wrist)


def _make_env(task, seed: int) -> OffScreenRenderEnv:
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def run(args: Args) -> None:
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite {args.task_suite_name!r}; choose from {sorted(MAX_STEPS)}")

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise IndexError(f"task_id={args.task_id} is outside [0, {suite.n_tasks})")

    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_id < len(initial_states):
        raise IndexError(f"init_state_id={args.init_state_id} is outside [0, {len(initial_states)})")

    output_dir = pathlib.Path(args.output_dir)
    (output_dir / "agentview").mkdir(parents=True, exist_ok=True)
    (output_dir / "wrist").mkdir(parents=True, exist_ok=True)

    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        inference_warmup_steps=args.inference_warmup_steps,
    )
    client.reset(task_description=task.language)
    env = _make_env(task, args.seed)

    actions: list[np.ndarray] = []
    success = False
    frame_id = 0
    policy_step = 0
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state_id])
        _save_frame(obs, output_dir, frame_id)
        frame_id += 1

        for env_step in range(MAX_STEPS[args.task_suite_name] + args.num_steps_wait):
            if env_step < args.num_steps_wait:
                action = np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float32)
            else:
                agentview, wrist = _policy_images(obs)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                example = {"image": [agentview, wrist], "lang": task.language}
                if client.include_state:
                    example["state"] = np.expand_dims(state, axis=0)

                raw_action = client.step(example=example, step=policy_step)["raw_action"]
                world = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
                rotation = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(raw_action["open_gripper"])
                if world.size != 3 or rotation.size != 3:
                    raise ValueError(
                        f"Invalid action shapes: world_vector={world.shape}, rotation_delta={rotation.shape}"
                    )
                action = np.concatenate((world, rotation, gripper))
                actions.append(action)
                policy_step += 1

            obs, _, success, _ = env.step(action.tolist())
            _save_frame(obs, output_dir, frame_id)
            frame_id += 1
            if success:
                break
    finally:
        env.close()

    np.save(output_dir / "actions.npy", np.asarray(actions, dtype=np.float32))
    metadata = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task.language,
        "init_state_id": args.init_state_id,
        "seed": args.seed,
        "success": bool(success),
        "num_saved_frames_per_camera": frame_id,
        "num_policy_steps": policy_step,
        "image_orientation": "rotated_180_degrees_to_match_policy_input",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("Episode finished: success=%s, policy_steps=%d", success, policy_step)
    logging.info("Saved %d frames per camera to %s", frame_id, output_dir.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    tyro.cli(run)
