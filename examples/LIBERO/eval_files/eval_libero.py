import dataclasses
import json
import logging
import math
import os
import pathlib
import re

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.LIBERO.eval_files.model2libero_twochunk_interface import TwoChunkModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    eval_log_dir: str = "/tmp/libero/eval_logs"  # Path to save final eval result JSON files

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"
    twochunk: bool = False
    twochunk_debug: bool = False
    twochunk_debug_every_step: bool = False
    twochunk_short_chunks_per_long_window: int = 8
    twochunk_attention_debug: bool = False
    inference_warmup_steps: int = 10


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "libero_eval"


def _model_result_name(pretrained_path: str, job_name: str) -> str:
    if not pretrained_path:
        return _sanitize_filename(job_name)

    ckpt_path = pathlib.Path(pretrained_path)
    ckpt_stem = ckpt_path.stem
    if ckpt_stem.endswith("_pytorch_model"):
        ckpt_stem = ckpt_stem[: -len("_pytorch_model")]

    parts = ckpt_path.parts
    if "checkpoints" in parts:
        ckpt_idx = parts.index("checkpoints")
        if ckpt_idx > 0:
            return _sanitize_filename(f"{parts[ckpt_idx - 1]}_{ckpt_stem}")

    return _sanitize_filename(f"{ckpt_path.parent.name}_{ckpt_stem}")


def _inference_stats_with_per_action(client_model: ModelClient) -> dict:
    stats = client_model.get_inference_stats() if hasattr(client_model, "get_inference_stats") else {}
    if not stats:
        return {}

    action_chunk_size = int(stats.get("action_chunk_size", 0) or 0)
    avg_chunk_time_s = float(stats.get("avg_model_inference_time_s", 0.0) or 0.0)
    avg_action_time_s = avg_chunk_time_s / action_chunk_size if action_chunk_size > 0 else 0.0
    model_inference_hz = 1.0 / avg_action_time_s if avg_action_time_s > 0.0 else 0.0
    return {
        **stats,
        "avg_model_inference_time_per_action_s": avg_action_time_s,
        "avg_predict_action_time_per_action_s": avg_action_time_s,
        "model_inference_hz": model_inference_hz,
        "predict_action_hz": model_inference_hz,
    }




def _log_twochunk_attention_debug(inference_stats: dict) -> None:
    summary = inference_stats.get("twochunk_attention_debug") if inference_stats else None
    if not summary:
        return
    groups = summary.get("groups", ["self", "action_token", "coarse_idct", "memory", "dino"])
    logging.info("TwoChunk attention debug averaged over %s model calls", summary.get("count"))
    header = "layer " + " ".join(f"{group:>13}" for group in groups) + "        sum"
    logging.info(header)
    for layer in summary.get("layers", []):
        row = f"{int(layer.get('layer', -1)):>5} " + " ".join(
            f"{100.0 * float(layer.get(group, 0.0)):12.2f}%" for group in groups
        ) + f" {100.0 * float(layer.get('sum', 0.0)):9.2f}%"
        logging.info(row)
    overall = summary.get("overall", {})
    row = "  avg " + " ".join(
        f"{100.0 * float(overall.get(group, 0.0)):12.2f}%" for group in groups
    ) + f" {100.0 * float(overall.get('sum', 0.0)):9.2f}%"
    logging.info(row)

def _save_eval_results(args: Args, result_data: dict) -> pathlib.Path:
    eval_log_dir = pathlib.Path(args.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_log_dir / f"{_model_result_name(args.pretrained_path, args.job_name)}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    return result_path


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_cls = TwoChunkModelClient if args.twochunk else ModelClient
    client_model = client_cls(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        inference_warmup_steps=args.inference_warmup_steps,
        **(
            {
                "twochunk_debug": args.twochunk_debug,
                "twochunk_debug_every_step": args.twochunk_debug_every_step,
                "twochunk_short_chunks_per_long_window": args.twochunk_short_chunks_per_long_window,
                "twochunk_attention_debug": args.twochunk_attention_debug,
            }
            if args.twochunk
            else {}
        ),
    )

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_results = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }
                if client_model.include_state:
                    # Raw LIBERO proprioception in the same 8-D order used by
                    # modality.json: xyz, axis-angle, two gripper qpos values.
                    # The policy server applies checkpoint q01-q99
                    # normalization and clips the result to [-1, 1].
                    example_dict["state"] = observation["observation.state"]

                response = client_model.step(example=example_dict, step=step)

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            policy_steps = int(len(full_actions))

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            episode_results.append(
                {
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "env_steps": int(args.num_steps_wait + policy_steps),
                    "policy_steps": policy_steps,
                    "video_path": str(video_path),
                }
            )
            if hasattr(client_model, "print_episode_attention_debug"):
                client_model.print_episode_attention_debug(
                    prefix=f"[LIBERO attention] task_id={task_id} episode={episode_idx} success={bool(done)}"
                )
            if hasattr(client_model, "reset_episode_attention_debug"):
                client_model.reset_episode_attention_debug()

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")
        task_results.append(
            {
                "task_id": int(task_id),
                "task_description": task_description,
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": task_success_rate,
                "episode_results": episode_results,
            }
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    inference_stats = _inference_stats_with_per_action(client_model)
    result_data = {
        "pretrained_path": args.pretrained_path,
        "task_suite_name": args.task_suite_name,
        "num_tasks_in_suite": int(num_tasks_in_suite),
        "num_eval_tasks": int(n_eval_tasks),
        "num_trials_per_task": int(args.num_trials_per_task),
        "seed": int(args.seed),
        "video_out_path": args.video_out_path,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": final_success_rate,
        "inference_stats": inference_stats,
        "task_results": task_results,
    }
    result_path = _save_eval_results(args, result_data)

    logging.info(f"Total success rate: {final_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Eval results saved at {result_path}")
    if inference_stats:
        _log_twochunk_attention_debug(inference_stats)
        logging.info(
            "Average model inference time: "
            f"{inference_stats['avg_model_inference_time_s']:.4f}s/chunk, "
            f"{inference_stats['avg_model_inference_time_per_action_s']:.4f}s/action "
            f"({inference_stats.get('model_inference_hz', 0.0):.2f} Hz), "
            f"(chunk_size={inference_stats['action_chunk_size']}, "
            f"chunk_calls={inference_stats['model_inference_time_count']})"
        )
        if "avg_32step_inference_time_s" in inference_stats:
            logging.info(
                "TwoChunk 32-step model inference time: "
                f"{inference_stats['avg_32step_inference_time_s']:.4f}s "
                f"(long_refreshes={inference_stats.get('twochunk_long_refreshes_per_32_steps', 1.0)}, "
                f"long_avg={inference_stats['avg_long_chunk_inference_time_s']:.4f}s + "
                f"{inference_stats['twochunk_short_chunks_per_32_steps']} * "
                f"short_avg={inference_stats['avg_short_chunk_inference_time_s']:.4f}s)"
            )
        if "avg_long_window_inference_time_s" in inference_stats:
            logging.info(
                "TwoChunk long-window model inference time: "
                f"{inference_stats['avg_long_window_inference_time_s']:.4f}s "
                f"(window_steps={inference_stats.get('twochunk_long_window_steps', 32)}, "
                f"long_avg={inference_stats['avg_long_chunk_inference_time_s']:.4f}s + "
                f"{inference_stats.get('twochunk_short_chunks_per_long_window', 0)} * "
                f"short_avg={inference_stats['avg_short_chunk_inference_time_s']:.4f}s)"
            )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
