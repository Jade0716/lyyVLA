"""
CALVIN eval client for QwenGR00T ActionToken TwoChunk/DCTMemory frameworks.

The policy server stays unchanged. This client requests one fast action chunk
per ``vision_refresh_steps`` and sets ``reset_cache=True`` at the first chunk of
each subtask so the framework clears both slow action-token and online memory
state for the new rollout.
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
        self.server_metadata = meta
        framework_name = str(meta.get("framework_name", ""))
        if not meta.get("twochunk", False) and "TwoChunk" not in framework_name:
            raise ValueError(
                "CalvinTwoChunkModelClient requires a TwoChunk server/checkpoint, "
                f"got metadata={meta}."
            )

        self.action_chunk_size = int(meta["action_chunk_size"])
        self.vision_refresh_steps = int(meta.get("vision_refresh_steps", self.action_chunk_size))
        self.language_refresh_steps = int(meta["language_refresh_steps"])
        if self.action_chunk_size != self.vision_refresh_steps:
            raise ValueError(
                "Server action_chunk_size must equal vision_refresh_steps, got "
                f"{self.action_chunk_size} and {self.vision_refresh_steps}."
            )

        self.include_state = bool(meta.get("include_state", False))
        self.dct_memory_enabled = bool(meta.get("dct_memory", False)) or "DCTMemory" in framework_name
        self.dct_memory_chunk_len = int(meta.get("dct_memory_chunk_len", self.language_refresh_steps))
        self.dct_memory_recent_mode = str(meta.get("dct_memory_recent_mode", "")).lower()
        self.dct_memory_cache_mode = str(meta.get("dct_memory_cache_mode", ""))
        self.online_memory_strategy = str(meta.get("online_memory_strategy", ""))
        if self.dct_memory_enabled:
            if self.dct_memory_recent_mode not in {"none", "raw", "raw_actions"}:
                raise ValueError(
                    "TwoChunk DCTMemory client expects summary-only or raw-actions recent memory, "
                    f"got dct_memory_recent_mode={self.dct_memory_recent_mode!r}; metadata={meta}."
                )
            if self.dct_memory_cache_mode and self.dct_memory_cache_mode != "prefix-summary":
                raise ValueError(
                    "TwoChunk DCTMemory client expects prefix-summary cache mode to match training, "
                    f"got dct_memory_cache_mode={self.dct_memory_cache_mode!r}; metadata={meta}."
                )
            if self.dct_memory_chunk_len <= 0:
                raise ValueError(f"Invalid dct_memory_chunk_len={self.dct_memory_chunk_len}.")
            if self.dct_memory_chunk_len % self.vision_refresh_steps != 0:
                raise ValueError(
                    "dct_memory_chunk_len must be divisible by vision_refresh_steps for aligned "
                    f"online memory updates, got {self.dct_memory_chunk_len} and {self.vision_refresh_steps}."
                )
            self.twochunk_short_chunks_per_long_window = self.dct_memory_chunk_len // self.vision_refresh_steps
        else:
            if self.language_refresh_steps % self.vision_refresh_steps != 0:
                raise ValueError(
                    "language_refresh_steps must be divisible by vision_refresh_steps, got "
                    f"{self.language_refresh_steps} and {self.vision_refresh_steps}."
                )
            self.twochunk_short_chunks_per_long_window = self.language_refresh_steps // self.vision_refresh_steps

        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.twochunk_debug = twochunk_debug
        self.twochunk_debug_every_step = twochunk_debug_every_step
        self.raw_actions: Optional[np.ndarray] = None
        self.predict_action_time_total_s = 0.0
        self.predict_action_time_count = 0
        self.long_chunk_time_total_s = 0.0
        self.long_chunk_time_count = 0
        self.short_chunk_time_total_s = 0.0
        self.short_chunk_time_count = 0
        print(
            f"*** CALVIN TwoChunk client: unnorm_key={unnorm_key}, "
            f"vision_refresh_steps={self.vision_refresh_steps}, "
            f"language_refresh_steps={self.language_refresh_steps}, "
            f"include_state={self.include_state}, "
            f"dct_memory={self.dct_memory_enabled}, "
            f"memory_strategy={self.online_memory_strategy or 'n/a'}, "
            f"server_meta={meta} ***"
        )

    def reset(self) -> None:
        self.raw_actions = None

    def get_inference_totals(self) -> dict:
        return {
            "model_inference_calls": self.predict_action_time_count,
            "model_inference_time_s": self.predict_action_time_total_s,
        }

    def get_inference_stats(self) -> dict:
        avg_time_s = (
            self.predict_action_time_total_s / self.predict_action_time_count
            if self.predict_action_time_count > 0
            else 0.0
        )
        long_avg_s = (
            self.long_chunk_time_total_s / self.long_chunk_time_count
            if self.long_chunk_time_count > 0
            else 0.0
        )
        short_avg_s = (
            self.short_chunk_time_total_s / self.short_chunk_time_count
            if self.short_chunk_time_count > 0
            else 0.0
        )
        long_window_steps = self.dct_memory_chunk_len if self.dct_memory_enabled else self.language_refresh_steps
        short_chunks_per_32_steps = 32.0 / float(self.vision_refresh_steps) if self.vision_refresh_steps > 0 else 0.0
        long_refreshes_per_32_steps = 32.0 / float(long_window_steps) if long_window_steps > 0 else 0.0
        avg_32step_s = long_refreshes_per_32_steps * long_avg_s + short_chunks_per_32_steps * short_avg_s
        return {
            "avg_model_inference_time_s": avg_time_s,
            "total_model_inference_time_s": self.predict_action_time_total_s,
            "model_inference_calls": self.predict_action_time_count,
            "action_chunk_size": self.action_chunk_size,
            "vision_refresh_steps": self.vision_refresh_steps,
            "language_refresh_steps": self.language_refresh_steps,
            "include_state": self.include_state,
            "vlm_cache_strategy": self.server_metadata.get("vlm_cache_strategy"),
            "dct_memory": self.dct_memory_enabled,
            "dct_memory_chunk_len": self.dct_memory_chunk_len if self.dct_memory_enabled else None,
            "dct_memory_recent_mode": self.dct_memory_recent_mode if self.dct_memory_enabled else None,
            "dct_memory_cache_mode": self.dct_memory_cache_mode if self.dct_memory_enabled else None,
            "online_memory_strategy": self.online_memory_strategy if self.dct_memory_enabled else None,
            "avg_long_chunk_inference_time_s": long_avg_s,
            "total_long_chunk_inference_time_s": self.long_chunk_time_total_s,
            "long_chunk_inference_time_count": self.long_chunk_time_count,
            "avg_short_chunk_inference_time_s": short_avg_s,
            "total_short_chunk_inference_time_s": self.short_chunk_time_total_s,
            "short_chunk_inference_time_count": self.short_chunk_time_count,
            "twochunk_short_chunks_per_32_steps": short_chunks_per_32_steps,
            "twochunk_short_chunks_per_long_window": self.twochunk_short_chunks_per_long_window,
            "twochunk_long_window_steps": long_window_steps,
            "twochunk_long_refreshes_per_32_steps": long_refreshes_per_32_steps,
            "avg_32step_inference_time_s": avg_32step_s,
        }

    def step(self, example: dict, step: int = 0) -> dict:
        refresh_chunk = step % self.vision_refresh_steps == 0 or self.raw_actions is None
        if refresh_chunk:
            reset_slow_cache = self.raw_actions is None
            if self.twochunk_debug:
                print(
                    "[TwoChunkClient] "
                    f"env_step={step} request fast_chunk "
                    f"(fast_refresh_every={self.vision_refresh_steps} env steps, "
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
            if response.get("status") != "ok":
                raise RuntimeError(f"TwoChunk server inference failed: {response}")
            timing = response["data"].get("inference_timing", {})
            model_time_s = float(timing.get("model_inference_time_s", 0.0) or 0.0)
            if model_time_s > 0.0:
                self.predict_action_time_total_s += model_time_s
                self.predict_action_time_count += 1
            self.raw_actions = np.asarray(response["data"]["actions"])[0]
            if self.raw_actions.shape[0] != self.vision_refresh_steps:
                raise ValueError(
                    "TwoChunk server returned an unexpected fast chunk length: "
                    f"shape={self.raw_actions.shape}, expected={self.vision_refresh_steps}."
                )
            fast_time_s = float(timing.get("fast_time_s", 0.0) or 0.0)
            if fast_time_s > 0.0:
                self.short_chunk_time_total_s += fast_time_s
                self.short_chunk_time_count += 1
            if bool(timing.get("slow_refresh", False)):
                slow_time_s = float(timing.get("slow_time_s", 0.0) or 0.0)
                if slow_time_s > 0.0:
                    self.long_chunk_time_total_s += slow_time_s
                    self.long_chunk_time_count += 1
            if self.twochunk_debug:
                print(
                    "[TwoChunkClient] "
                    f"env_step={step} received fast_chunk shape={self.raw_actions.shape} "
                    f"request_time={inference_time_s:.4f}s "
                    f"model_time={float(timing.get('model_inference_time_s', 0.0)):.4f}s "
                    f"dct_memory_chunk_len={self.dct_memory_chunk_len if self.dct_memory_enabled else 'n/a'} "
                    f"memory_strategy={self.online_memory_strategy or 'n/a'} "
                    f"chunk_request_count={self.predict_action_time_count}"
                )
        elif self.twochunk_debug_every_step:
            print(
                "[TwoChunkClient] "
                f"env_step={step} reuse cached fast_chunk "
                f"action_index={step % self.vision_refresh_steps}/{self.vision_refresh_steps}"
            )

        raw_actions = self.raw_actions[step % self.vision_refresh_steps][None]
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

    def get_inference_totals(self) -> dict:
        return self.client.get_inference_totals()

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        rgb_static = obs["rgb_obs"]["rgb_static"]
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
        # Keep CALVIN's native camera resolutions here: static is 200x200 and
        # gripper is 84x84. The framework/server applies the checkpoint's
        # training-time obs_image_size recursively, which keeps eval aligned
        # with the two-view training path that handles mixed resolutions.
        image = image_tools.convert_to_uint8(rgb_static)
        wrist_image = image_tools.convert_to_uint8(rgb_gripper)

        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }
        if self.client.include_state:
            robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32).reshape(-1)
            # Send raw 15-D CALVIN robot_obs. The server extracts the 8-D
            # training state order and applies checkpoint normalization.
            example["state"] = robot_obs[None, :]
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
