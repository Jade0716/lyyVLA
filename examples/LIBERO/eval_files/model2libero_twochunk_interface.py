"""LIBERO client adapter for QwenGR00T ActionToken TwoChunk inference."""

from typing import Optional

import numpy as np

from examples.LIBERO.eval_files.model2libero_interface import ModelClient


class TwoChunkModelClient(ModelClient):
    """Request one fast chunk every DINO refresh interval.

    The first request of every episode sets ``reset_cache=True``. The server
    then computes the slow VLM action-token cache bank once, reuses it for
    intermediate DINO refreshes, and refreshes it at ``language_refresh_steps``.
    """

    def __init__(
        self,
        *args,
        twochunk_debug: bool = False,
        twochunk_debug_every_step: bool = False,
        twochunk_short_chunks_per_long_window: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        meta = self._server_metadata
        framework_name = str(meta.get("framework_name", ""))
        if not meta.get("twochunk", False) and "TwoChunk" not in framework_name:
            raise ValueError(
                "TwoChunkModelClient requires a TwoChunk server/checkpoint, "
                f"got metadata={meta}."
            )

        self.vision_refresh_steps = int(meta.get("vision_refresh_steps", self.action_chunk_size))
        self.language_refresh_steps = int(meta["language_refresh_steps"])
        if self.action_chunk_size != self.vision_refresh_steps:
            raise ValueError(
                "Server action_chunk_size must equal vision_refresh_steps, got "
                f"{self.action_chunk_size} and {self.vision_refresh_steps}."
            )
        self.twochunk_debug = twochunk_debug
        self.twochunk_debug_every_step = twochunk_debug_every_step
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
        elif self.language_refresh_steps > 0 and self.vision_refresh_steps > 0:
            if self.language_refresh_steps % self.vision_refresh_steps != 0:
                raise ValueError(
                    "language_refresh_steps must be divisible by vision_refresh_steps, got "
                    f"{self.language_refresh_steps} and {self.vision_refresh_steps}."
                )
            self.twochunk_short_chunks_per_long_window = (
                self.language_refresh_steps // self.vision_refresh_steps
            )
        else:
            self.twochunk_short_chunks_per_long_window = int(twochunk_short_chunks_per_long_window)
        self._reset_server_cache = True
        self.long_chunk_time_total_s = 0.0
        self.long_chunk_time_count = 0
        self.short_chunk_time_total_s = 0.0
        self.short_chunk_time_count = 0

    def reset(self, task_description: str) -> None:
        super().reset(task_description)
        self._reset_server_cache = True

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        task_description = example.get("lang")
        if task_description != self.task_description:
            self.reset(task_description)

        refresh_fast_chunk = step % self.vision_refresh_steps == 0 or self.raw_actions is None
        if refresh_fast_chunk:
            reset_cache = self._reset_server_cache
            request = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "reset_cache": reset_cache,
                "debug_twochunk": self.twochunk_debug,
            }
            response = self.client.predict_action(request)
            if response.get("status") != "ok":
                raise RuntimeError(f"TwoChunk server inference failed: {response}")

            self.raw_actions = np.asarray(response["data"]["actions"])[0]
            if self.raw_actions.shape[0] != self.vision_refresh_steps:
                raise ValueError(
                    "TwoChunk server returned an unexpected fast chunk length: "
                    f"shape={self.raw_actions.shape}, expected={self.vision_refresh_steps}."
                )
            timing = response["data"].get("inference_timing", {})
            measured = self._record_server_inference_time(response)
            if measured:
                fast_time_s = float(timing.get("fast_time_s", 0.0) or 0.0)
                if fast_time_s > 0.0:
                    self.short_chunk_time_total_s += fast_time_s
                    self.short_chunk_time_count += 1
                if bool(timing.get("slow_refresh", False)):
                    slow_time_s = float(timing.get("slow_time_s", 0.0) or 0.0)
                    if slow_time_s > 0.0:
                        self.long_chunk_time_total_s += slow_time_s
                        self.long_chunk_time_count += 1
            self._reset_server_cache = False

            if self.twochunk_debug:
                print(
                    "[LIBEROTwoChunkClient] "
                    f"env_step={step}; reset_cache={reset_cache}; "
                    f"dino_refresh={self.vision_refresh_steps}; "
                    f"vlm_refresh={self.language_refresh_steps}; "
                    f"dct_memory_chunk_len={self.dct_memory_chunk_len if self.dct_memory_enabled else 'n/a'}; "
                    f"memory_strategy={self.online_memory_strategy or 'n/a'}; "
                    f"model_time={float(timing.get('model_inference_time_s', 0.0)):.4f}s"
                )
        elif self.twochunk_debug_every_step:
            print(
                "[LIBEROTwoChunkClient] "
                f"env_step={step}; cached_action_index={step % self.vision_refresh_steps}"
            )

        action = self.raw_actions[step % self.vision_refresh_steps]
        return {
            "raw_action": {
                "world_vector": np.asarray(action[:3]),
                "rotation_delta": np.asarray(action[3:6]),
                "open_gripper": np.asarray(action[6:7]),
            }
        }

    def get_inference_stats(self) -> dict:
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
        short_chunks_per_32_steps = (
            32.0 / float(self.vision_refresh_steps) if self.vision_refresh_steps > 0 else 0.0
        )
        long_refreshes_per_32_steps = (
            32.0 / float(long_window_steps) if long_window_steps > 0 else 0.0
        )
        avg_long_window_s = long_avg_s + self.twochunk_short_chunks_per_long_window * short_avg_s
        avg_32step_s = long_refreshes_per_32_steps * long_avg_s + short_chunks_per_32_steps * short_avg_s
        return {
            **super().get_inference_stats(),
            "vision_refresh_steps": self.vision_refresh_steps,
            "language_refresh_steps": self.language_refresh_steps,
            "vlm_cache_strategy": self._server_metadata.get("vlm_cache_strategy"),
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
            "avg_long_window_inference_time_s": avg_long_window_s,
            "avg_32step_inference_time_formula": (
                "twochunk_long_refreshes_per_32_steps * avg_long_chunk_inference_time_s + "
                "twochunk_short_chunks_per_32_steps * avg_short_chunk_inference_time_s"
            ),
            "avg_long_window_inference_time_formula": (
                "avg_long_chunk_inference_time_s + "
                "twochunk_short_chunks_per_long_window * avg_short_chunk_inference_time_s"
            ),
        }
