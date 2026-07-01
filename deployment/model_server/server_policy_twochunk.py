"""Dedicated websocket server for TwoChunk evaluation."""

import argparse
import logging
import socket

from transformers.utils import logging as transformers_logging

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.share_tools import read_mode_config


def _validate_twochunk_checkpoint(ckpt_path: str) -> None:
    model_cfg, _ = read_mode_config(ckpt_path)
    framework_cfg = model_cfg.get("framework", {})
    framework_name = str(framework_cfg.get("name", ""))
    if "TwoChunk" not in framework_name:
        raise ValueError(
            "server_policy_twochunk.py requires a TwoChunk checkpoint, "
            f"but framework.name={framework_name!r}."
        )

    qwenvl_cfg = framework_cfg.get("qwenvl", {})
    vision_refresh_steps = int(qwenvl_cfg.get("vision_refresh_steps", 0))
    language_refresh_steps = int(qwenvl_cfg.get("language_refresh_steps", 0))
    if vision_refresh_steps <= 0 or language_refresh_steps <= 0:
        raise ValueError(
            "TwoChunk checkpoint must define positive framework.qwenvl."
            "vision_refresh_steps and language_refresh_steps."
        )
    if language_refresh_steps % vision_refresh_steps != 0:
        raise ValueError(
            "language_refresh_steps must be divisible by vision_refresh_steps for "
            f"deterministic cache scheduling, got {language_refresh_steps} and "
            f"{vision_refresh_steps}."
        )

    if "DCTMemory" in framework_name:
        dct_memory_cfg = framework_cfg.get("dct_memory", {})
        recent_mode = str(dct_memory_cfg.get("recent_mode", "dct")).lower()
        chunk_len = int(dct_memory_cfg.get("chunk_len", 32))
        if recent_mode not in {"none", "raw", "raw_actions"}:
            raise ValueError(
                "TwoChunk DCTMemory eval expects framework.dct_memory.recent_mode="
                f"none or raw_actions to match supported training modes, got {recent_mode!r}."
            )
        if chunk_len <= 0:
            raise ValueError(f"framework.dct_memory.chunk_len must be positive, got {chunk_len}.")
        if chunk_len % vision_refresh_steps != 0:
            raise ValueError(
                "DCTMemory chunk_len should be divisible by vision_refresh_steps so "
                "online summary updates align with short chunks, got "
                f"chunk_len={chunk_len}, vision_refresh_steps={vision_refresh_steps}."
            )
        vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
        cache_mode = str(vla_data_cfg.get("dct_memory_cache_mode", ""))
        if cache_mode and cache_mode != "prefix-summary":
            raise ValueError(
                "TwoChunk DCTMemory eval expects datasets.vla_data.dct_memory_cache_mode="
                f"prefix-summary to match current training, got {cache_mode!r}."
            )


def main(args: argparse.Namespace) -> None:
    _validate_twochunk_checkpoint(args.ckpt_path)
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )
    metadata = {
        **wrapper.metadata,
        "server_mode": "libero_twochunk",
        "cache_reset": "client_first_request_per_episode",
    }

    hostname = socket.gethostname()
    logging.info(
        "Starting LIBERO TwoChunk server on %s:%d (%s); metadata=%s",
        args.host,
        args.port,
        hostname,
        metadata,
    )
    WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    ).serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    transformers_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    main(build_argparser().parse_args())
