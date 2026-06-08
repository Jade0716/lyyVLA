"""
CALVIN eval client for QwenGR00T ActionToken TwoChunk frameworks.

The policy server stays unchanged. This client keeps the normal CALVIN action
chunk cache, but sends ``reset_cache=True`` on the first chunk of each subtask
so the framework refreshes its slow action-token cache for the new rollout.
"""

import dataclasses
import time
from typing import Optional

import numpy as np
import tyro

from deployment.model_server.tools import image_tools
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.calvin.eval_files.eval_calvin import (
    Args,
    evaluate_policy_ddp,
    make_env,
)


@dataclasses.dataclass
class TwoChunkArgs(Args):
    use_ddim: bool = True
    num_ddim_steps: int = 10
    twochunk_debug: bool = False
    twochunk_debug_every_step: bool = False


class CalvinTwoChunkModelClient:
    def __init__(
        self,
        host: str,
        port: int,
        unnorm_key: Optional[str] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        twochunk_debug: bool = True,
        twochunk_debug_every_step: bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.twochunk_debug = twochunk_debug
        self.twochunk_debug_every_step = twochunk_debug_every_step
        self.raw_actions: Optional[np.ndarray] = None
        self.predict_action_time_total_s = 0.0
        self.predict_action_time_count = 0
        print(
            f"*** CALVIN TwoChunk client: unnorm_key={unnorm_key}, "
            f"action_chunk_size={self.action_chunk_size}, server_meta={meta} ***"
        )

    def reset(self) -> None:
        self.raw_actions = None

    def get_inference_stats(self) -> dict:
        avg_time_s = (
            self.predict_action_time_total_s / self.predict_action_time_count
            if self.predict_action_time_count > 0
            else 0.0
        )
        return {
            "avg_model_inference_time_s": avg_time_s,
            "total_model_inference_time_s": self.predict_action_time_total_s,
            "model_inference_time_count": self.predict_action_time_count,
            "avg_predict_action_chunk_time_s": avg_time_s,
            "total_predict_action_chunk_time_s": self.predict_action_time_total_s,
            "predict_action_chunk_count": self.predict_action_time_count,
            "action_chunk_size": self.action_chunk_size,
        }

    def step(self, example: dict, step: int = 0) -> dict:
        refresh_chunk = step % self.action_chunk_size == 0 or self.raw_actions is None
        if refresh_chunk:
            reset_slow_cache = self.raw_actions is None
            if self.twochunk_debug:
                print(
                    "[TwoChunkClient] "
                    f"env_step={step} request fast_chunk "
                    f"(fast_refresh_every={self.action_chunk_size} env steps, "
                    f"reset_slow_cache={reset_slow_cache})"
                )
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "reset_cache": reset_slow_cache,
                "debug_twochunk": self.twochunk_debug,
            }
            inference_start = time.perf_counter()
            response = self.client.predict_action(vla_input)
            inference_time_s = time.perf_counter() - inference_start
            self.predict_action_time_total_s += inference_time_s
            self.predict_action_time_count += 1
            self.raw_actions = np.asarray(response["data"]["actions"])[0]
            if self.twochunk_debug:
                print(
                    "[TwoChunkClient] "
                    f"env_step={step} received fast_chunk shape={self.raw_actions.shape} "
                    f"request_time={inference_time_s:.4f}s "
                    f"chunk_request_count={self.predict_action_time_count}"
                )
        elif self.twochunk_debug_every_step:
            print(
                "[TwoChunkClient] "
                f"env_step={step} reuse cached fast_chunk "
                f"action_index={step % self.action_chunk_size}/{self.action_chunk_size}"
            )

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        return {"raw_action": raw_action}


class CalvinTwoChunkPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        pretrained_path: str = "",
        unnorm_key: str = "",
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        twochunk_debug: bool = True,
        twochunk_debug_every_step: bool = False,
    ) -> None:
        self.client = CalvinTwoChunkModelClient(
            host=host,
            port=port,
            unnorm_key=(unnorm_key or None),
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            twochunk_debug=twochunk_debug,
            twochunk_debug_every_step=twochunk_debug_every_step,
        )
        self.resize_size = resize_size
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0
        self.client.reset()

    def get_inference_stats(self) -> dict:
        return self.client.get_inference_stats()

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        rgb_static = obs["rgb_obs"]["rgb_static"]
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        action = np.concatenate(
            [
                np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1),
                np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1),
                np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
        self.step_count += 1
        return action


def main(args: TwoChunkArgs):
    policy = CalvinTwoChunkPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        use_ddim=args.use_ddim,
        num_ddim_steps=args.num_ddim_steps,
        twochunk_debug=args.twochunk_debug,
        twochunk_debug_every_step=args.twochunk_debug_every_step,
    )
    env = make_env(args.dataset_path)
    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
    )


if __name__ == "__main__":
    tyro.cli(main)
