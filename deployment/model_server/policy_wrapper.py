# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def _config_flag(value: Any) -> bool:
    """Parse bool-like values from saved YAML configs."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        framework_cfg = model_cfg["framework"]
        framework_name = framework_cfg.get("name", "")
        self._framework_name = str(framework_name)
        action_model_cfg = framework_cfg["action_model"]
        qwenvl_cfg = framework_cfg.get("qwenvl", {})
        dct_memory_cfg = framework_cfg.get("dct_memory", {})
        vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
        self._include_state = _config_flag(vla_data_cfg.get("include_state", False))
        self._dct_memory_enabled = "DCTMemory" in self._framework_name
        self._dct_memory_cfg = dct_memory_cfg if isinstance(dct_memory_cfg, dict) else {}
        self._dct_memory_cache_mode = str(vla_data_cfg.get("dct_memory_cache_mode", ""))
        self._language_refresh_steps = None
        self._vision_refresh_steps = None

        if "TwoChunk" in framework_name and "vision_refresh_steps" in qwenvl_cfg:
            self._vision_refresh_steps = int(qwenvl_cfg["vision_refresh_steps"])
            self._language_refresh_steps = int(
                qwenvl_cfg.get("language_refresh_steps", self._vision_refresh_steps)
            )
            self._action_chunk_size = self._vision_refresh_steps
        elif "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        elif "num_actions_chunk" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["num_actions_chunk"])
        else:
            raise ValueError(
                "PolicyServerWrapper: no action_horizon, future_action_window_size, "
                f"or num_actions_chunk found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "framework_name": self._framework_name,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "include_state": self._include_state,
        }
        if self._vision_refresh_steps is not None:
            base.update(
                {
                    "twochunk": True,
                    "vision_refresh_steps": self._vision_refresh_steps,
                    "language_refresh_steps": self._language_refresh_steps,
                    "vlm_cache_strategy": "action_token_hidden_cache_bank",
                }
            )
        if self._dct_memory_enabled:
            chunk_len = int(self._dct_memory_cfg.get("chunk_len", 32))
            recent_mode = str(self._dct_memory_cfg.get("recent_mode", "dct")).lower()
            summary_keep_freq = int(self._dct_memory_cfg.get("summary_keep_freq", 8))
            chunk_keep_freq = int(self._dct_memory_cfg.get("chunk_keep_freq", 4))
            recent_disabled = recent_mode in {"none", "disabled", "off", "false", "0"}
            recent_tokens = 0 if recent_disabled else (chunk_len if recent_mode in {"raw", "raw_actions"} else chunk_keep_freq)
            if recent_disabled:
                online_memory_strategy = "prefix_summary_only"
            elif recent_mode in {"raw", "raw_actions"}:
                online_memory_strategy = "prefix_summary_raw_recent"
            else:
                online_memory_strategy = "legacy_dct_recent"
            base.update(
                {
                    "dct_memory": True,
                    "dct_memory_chunk_len": chunk_len,
                    "dct_memory_recent_mode": recent_mode,
                    "dct_memory_recent_tokens": recent_tokens,
                    "dct_memory_summary_keep_freq": summary_keep_freq,
                    "dct_memory_summary_tokens": summary_keep_freq,
                    "dct_memory_chunk_keep_freq": chunk_keep_freq,
                    "dct_memory_cache_mode": self._dct_memory_cache_mode,
                    "online_memory_strategy": online_memory_strategy,
                }
            )
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        model_examples = []
        for example in examples:
            model_example = dict(example)
            if self._include_state:
                if "state" not in model_example or model_example["state"] is None:
                    raise ValueError(
                        "Checkpoint config has datasets.vla_data.include_state=true, "
                        "but the inference example does not contain `state`."
                    )
                model_example["state"] = proc.apply_state(model_example["state"])
            else:
                model_example.pop("state", None)
            model_examples.append(model_example)

        out = self._framework.predict_action(examples=model_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        result: Dict[str, Any] = {"actions": unnorm}
        if "inference_timing" in out:
            result["inference_timing"] = out["inference_timing"]
        return result
