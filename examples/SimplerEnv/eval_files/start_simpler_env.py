import os
import faulthandler
import json
from pathlib import Path

import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

# from IPython import embed; embed()
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    args = get_args()
    if args.additional_env_build_kwargs is None:
        args.additional_env_build_kwargs = {}
    args.additional_env_build_kwargs.setdefault("renderer_kwargs", {"offscreen_only": True})

    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if os.getenv("DEBUG", False):
        start_debugpy_once()
    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,  # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation. SimplerEnv uses basename(args.ckpt_path) as
    # the first save-directory component, so optionally swap in a run-scoped
    # save name after the model client has already consumed the real ckpt path.
    real_ckpt_path = args.ckpt_path
    if args.eval_save_name is not None:
        args.real_ckpt_path = real_ckpt_path
        args.ckpt_path = args.eval_save_name
    success_arr = maniskill2_evaluator(model, args)
    args.ckpt_path = real_ckpt_path
    average_success = float(np.mean(success_arr)) if success_arr else float("nan")
    inference_time_stats = model.get_inference_time_stats()
    average_inference_time_sec = inference_time_stats["average_inference_time_sec"]
    print(args)
    print(" " * 10, "Average success", average_success)
    print(" " * 10, "Average inference time (s)", average_inference_time_sec)

    if args.summary_path is not None:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "average_success": average_success,
            "success_arr": [bool(success) for success in success_arr],
            "num_episodes": len(success_arr),
            **inference_time_stats,
            "args": to_jsonable(vars(args)),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved eval summary to {summary_path}")
