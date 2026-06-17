# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Two-stage VLA trainer for QwenGR00T ActionToken TwoChunk.

Stage behavior is controlled by the training YAML:
  - trainer.stage: 1 -> DCT-only training.
  - trainer.stage: 2 -> joint DCT + action training.

In stage 1 this script saves DCT/action-token weights only, excluding
``action_model.*`` so stage 2 can load them with strict=False and initialize
the action head freshly.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    logger,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
    sync_twochunk_data_config,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _is_stage1(cfg) -> bool:
    return int(cfg.trainer.get("stage", 2)) == 1


def _filter_dct_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if not k.startswith("action_model.")}


class TwoStageVLATrainer(VLATrainer):
    def _train_step(self, batch_vla, batch_vlm=None):
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    train_step=self.completed_steps,
                    max_train_steps=self.config.trainer.max_train_steps,
                )
                total_loss = output_dict["action_loss"]

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        log_dict = {
            "action_loss": total_loss.item(),
        }
        for loss_key in (
            "action_dit_loss",
            "action_dit_loss_weight",
            "motion_dct_loss",
            "weighted_motion_dct_loss",
            "dct_loss",
        ):
            if loss_key in output_dict:
                loss_value = output_dict[loss_key]
                log_dict[loss_key] = loss_value.item() if hasattr(loss_value, "item") else loss_value
        return log_dict

    def _save_dct_only_state(self, path: str):
        save_format = getattr(self.config.trainer, "save_format", "pt")
        state_dict = self.accelerator.get_state_dict(self.model)
        state_dict = _filter_dct_state_dict(state_dict)

        if save_format == "safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, path + "_dct_model.safetensors")
        elif save_format == "pt":
            torch.save(state_dict, path + "_dct_pytorch_model.pt")
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

    def _save_checkpoint(self):
        if not _is_stage1(self.config):
            return super()._save_checkpoint()

        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            self._save_dct_only_state(checkpoint_path)

            summary_data = {"steps": self.completed_steps, "stage": 1, "checkpoint_type": "dct_only"}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Stage 1 DCT-only checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _finalize_training(self):
        if not _is_stage1(self.config):
            return super()._finalize_training()

        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            self._save_dct_only_state(os.path.join(final_checkpoint, "stage1"))
            logger.info(f"Stage 1 complete. DCT-only model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("Two-stage VLA Training :: Warming Up")

    cfg = sync_twochunk_data_config(cfg)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = TwoStageVLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    cfg.trainer.warmup = False

    main(cfg)
