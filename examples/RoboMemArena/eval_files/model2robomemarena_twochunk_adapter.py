"""RoboMemArena policy adapter for StarVLA TwoChunk/DCTMemory servers.

This file implements RoboMemArena's official ``BasePolicyAdapter`` contract.
The server returns un-normalized env-space actions, and whether DCTMemory is
active is decided by the checkpoint's saved YAML and server metadata.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from policy_adapter import BasePolicyAdapter


class StarVLATwoChunkRoboMemArenaAdapter(BasePolicyAdapter):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6697,
        unnorm_key: Optional[str] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        twochunk_debug: bool = False,
        inference_warmup_steps: int = 10,
    ) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        framework_name = str(self.metadata.get("framework_name", ""))
        if not self.metadata.get("twochunk", False) and "TwoChunk" not in framework_name:
            raise ValueError(
                "StarVLATwoChunkRoboMemArenaAdapter requires a TwoChunk server/checkpoint, "
                f"got metadata={self.metadata}."
            )

        self.action_chunk_size = int(self.metadata["action_chunk_size"])
        self.vision_refresh_steps = int(self.metadata.get("vision_refresh_steps", self.action_chunk_size))
        self.language_refresh_steps = int(self.metadata.get("language_refresh_steps", self.vision_refresh_steps))
        if self.action_chunk_size != self.vision_refresh_steps:
            raise ValueError(
                "Server action_chunk_size must equal vision_refresh_steps, got "
                f"{self.action_chunk_size} and {self.vision_refresh_steps}."
            )

        self.dct_memory_enabled = bool(self.metadata.get("dct_memory", False)) or "DCTMemory" in framework_name
        self.dct_memory_chunk_len = int(self.metadata.get("dct_memory_chunk_len", self.language_refresh_steps))
        self.dct_memory_recent_mode = str(self.metadata.get("dct_memory_recent_mode", "")).lower()
        self.dct_memory_cache_mode = str(self.metadata.get("dct_memory_cache_mode", ""))
        self.online_memory_strategy = str(self.metadata.get("online_memory_strategy", ""))
        if self.dct_memory_enabled:
            if self.dct_memory_recent_mode not in {"none", "raw", "raw_actions"}:
                raise ValueError(
                    "TwoChunk DCTMemory eval expects summary-only or raw-actions recent memory, "
                    f"got dct_memory_recent_mode={self.dct_memory_recent_mode!r}; metadata={self.metadata}."
                )
            if self.dct_memory_cache_mode and self.dct_memory_cache_mode != "prefix-summary":
                raise ValueError(
                    "TwoChunk DCTMemory eval expects prefix-summary cache mode, "
                    f"got dct_memory_cache_mode={self.dct_memory_cache_mode!r}; metadata={self.metadata}."
                )
            if self.dct_memory_chunk_len % self.vision_refresh_steps != 0:
                raise ValueError(
                    "dct_memory_chunk_len must be divisible by vision_refresh_steps, got "
                    f"{self.dct_memory_chunk_len} and {self.vision_refresh_steps}."
                )
        elif self.language_refresh_steps % self.vision_refresh_steps != 0:
            raise ValueError(
                "language_refresh_steps must be divisible by vision_refresh_steps, got "
                f"{self.language_refresh_steps} and {self.vision_refresh_steps}."
            )

        self.include_state = bool(self.metadata.get("include_state", False))
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = int(num_ddim_steps)
        self.twochunk_debug = bool(twochunk_debug)
        self.inference_warmup_steps = int(inference_warmup_steps)
        self._reset_server_cache = True
        self._request_count = 0
        self._timed_count = 0
        self._model_time_total_s = 0.0
        self._wall_time_total_s = 0.0

        print(
            "*** RoboMemArena StarVLA adapter: "
            f"unnorm_key={unnorm_key}, include_state={self.include_state}, "
            f"vision_refresh_steps={self.vision_refresh_steps}, "
            f"language_refresh_steps={self.language_refresh_steps}, "
            f"dct_memory={self.dct_memory_enabled}, "
            f"memory_strategy={self.online_memory_strategy or 'n/a'}, "
            f"metadata={self.metadata} ***"
        )

    def reset(self) -> None:
        self._reset_server_cache = True

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def _build_example(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        image = obs.get("observation/image")
        wrist = obs.get("observation/wrist_image")
        if image is None:
            image = obs.get("base_0_rgb")
        if wrist is None:
            wrist = obs.get("right_wrist_0_rgb")
        if image is None or wrist is None:
            raise KeyError(
                "RoboMemArena adapter expected processed images under "
                "'observation/image' and 'observation/wrist_image'."
            )

        example: dict[str, Any] = {
            "image": [np.asarray(image), np.asarray(wrist)],
            "lang": str(prompt),
        }
        if self.include_state:
            state = obs.get("observation/state")
            if state is None:
                raise KeyError("Checkpoint requires state, but 'observation/state' is missing from adapter obs.")
            example["state"] = np.asarray(state, dtype=np.float32)[None, :]
        return example

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        example = self._build_example(obs, prompt)
        request = {
            "examples": [example],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "reset_cache": self._reset_server_cache,
            "debug_twochunk": self.twochunk_debug,
        }
        wall_start = time.perf_counter()
        response = self.client.predict_action(request)
        wall_time_s = time.perf_counter() - wall_start
        if response.get("status") != "ok":
            raise RuntimeError(f"TwoChunk server inference failed: {response}")

        actions = np.asarray(response["data"]["actions"], dtype=np.float32)[0]
        if actions.ndim != 2:
            raise ValueError(f"Server returned invalid action shape {actions.shape}; expected [T, D].")
        if actions.shape[0] != self.vision_refresh_steps:
            raise ValueError(
                "TwoChunk server returned an unexpected fast chunk length: "
                f"shape={actions.shape}, expected={self.vision_refresh_steps}. "
                "Set RoboMemArena --replan-steps to this same value."
            )

        self._request_count += 1
        timing = response["data"].get("inference_timing", {})
        model_time_s = float(timing.get("model_inference_time_s", 0.0) or 0.0)
        if self._request_count > self.inference_warmup_steps:
            self._timed_count += 1
            self._wall_time_total_s += wall_time_s
            self._model_time_total_s += model_time_s

        if self.twochunk_debug:
            print(
                "[RoboMemArenaStarVLAAdapter] "
                f"reset_cache={self._reset_server_cache}; action_shape={actions.shape}; "
                f"model_time={model_time_s:.4f}s; wall_time={wall_time_s:.4f}s; "
                f"dct_memory={self.dct_memory_enabled}; "
                f"online_chunks={timing.get('online_dct_memory_chunks', 'n/a')}"
            )
        self._reset_server_cache = False
        return actions

    def get_inference_stats(self) -> dict[str, Any]:
        avg_model = self._model_time_total_s / self._timed_count if self._timed_count else 0.0
        avg_wall = self._wall_time_total_s / self._timed_count if self._timed_count else 0.0
        return {
            "request_count": self._request_count,
            "timed_request_count": self._timed_count,
            "inference_warmup_steps": self.inference_warmup_steps,
            "avg_model_inference_time_s": avg_model,
            "avg_wall_inference_time_s": avg_wall,
            "vision_refresh_steps": self.vision_refresh_steps,
            "language_refresh_steps": self.language_refresh_steps,
            "dct_memory": self.dct_memory_enabled,
            "dct_memory_chunk_len": self.dct_memory_chunk_len if self.dct_memory_enabled else None,
            "dct_memory_recent_mode": self.dct_memory_recent_mode if self.dct_memory_enabled else None,
            "dct_memory_cache_mode": self.dct_memory_cache_mode if self.dct_memory_enabled else None,
            "online_memory_strategy": self.online_memory_strategy if self.dct_memory_enabled else None,
        }


def build_adapter(**kwargs: Any) -> StarVLATwoChunkRoboMemArenaAdapter:
    return StarVLATwoChunkRoboMemArenaAdapter(**kwargs)
