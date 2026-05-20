#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
"""
Evaluation script for LIBERO with attention heatmap generation.
Maintains the same client-server architecture as eval_libero.py but adds GradCAM visualization.

Usage:
    1. Start server: bash examples/LIBERO/eval_files/run_policy_server.sh
    2. Run eval: bash examples/LIBERO/eval_files/eval_libero_attention.sh
"""

import dataclasses
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import time

import numpy as np
import tqdm
import tyro

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Heatmap generation parameters
    generate_heatmaps: bool = False
    heatmap_dir: str = "./attention_heatmaps"
    heatmap_layer: int = -1  # -1 means last layer, 0 means first layer, etc.
    heatmap_interval: int = 10  # Generate heatmap every N steps

    # Utils
    video_out_path: str = "experiments/libero/logs"
    seed: int = 7
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test_with_attention"


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eval_libero_with_attention(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    # Create output directories
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.generate_heatmaps:
        pathlib.Path(args.heatmap_dir).mkdir(parents=True, exist_ok=True)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Initialize client (same as original eval_libero.py)
    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Trials", leave=False):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            full_actions = []
            heatmap_count = 0

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get observation
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                replay_images.append(img)

                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ])

                example = {
                    "image": [img, wrist_img],
                    "lang": task_description,
                }

                # Generate heatmap at interval
                heatmap_paths = None
                if args.generate_heatmaps and step % args.heatmap_interval == 0:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    instruction_short = task_description[:30].replace(" ", "_")
                    save_paths = [
                        os.path.join(
                            args.heatmap_dir,
                            f"task{task_id}_trial{episode_idx}_step{step}_view{i}_{instruction_short}_{timestamp}.png"
                        ) for i in range(2)
                    ]
                    try:
                        result = client_model.step_with_attention(
                            example=example,
                            step=step,
                            save_dir=args.heatmap_dir,
                            layer_idx=args.heatmap_layer,
                        )
                        heatmap_paths = result.get("heatmap_paths")
                        heatmap_count += 1
                    except Exception as e:
                        logging.warning(f"Heatmap generation failed: {e}")
                        # Fallback to normal inference
                        client_model.reset(task_description)
                        result = client_model.step(example=example, step=step)
                else:
                    # Normal inference
                    result = client_model.step(example=example, step=step)

                raw_action = result["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                obs, reward, done, info = env.step(delta_action.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            import imageio
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) /
                f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)

            # Log results
            logging.info(f"Success: {done}")
            logging.info(f"Heatmaps generated: {heatmap_count}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")
    if args.generate_heatmaps:
        logging.info(f"Heatmaps saved to: {args.heatmap_dir}")


if __name__ == "__main__":
    tyro.cli(eval_libero_with_attention)