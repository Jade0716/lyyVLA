"""Run RoboMemArena evaluation with the StarVLA websocket adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any


SUBSET_TASKS = {
    "all": list(range(1, 27)),
    "sequence": [1, 2, 3, 22],
    "counting": [6, 7, 8, 9, 10, 15, 16],
    "transferring": [18, 19, 25, 26],
    "occlusion": [4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24],
}


def _parse_task_ids(raw: str | None, subset: str) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(SUBSET_TASKS[subset])
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    for task_id in out:
        if task_id < 1 or task_id > 26:
            raise ValueError(f"Invalid RoboMemArena task id {task_id}; expected 1..26.")
    return out


def _json_loads_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object.")
    return data


def _load_openpi_prompt_fn(robomem_root: Path):
    prompt_path = robomem_root / "evaluation_benchmark" / "openpi_minimal_runtime" / "task_prompts.py"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"OpenPI prompt table not found: {prompt_path}")
    spec = importlib.util.spec_from_file_location("robomemarena_openpi_task_prompts", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OpenPI prompt table from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_prompt = getattr(module, "get_prompt", None)
    if not callable(get_prompt):
        raise AttributeError(f"{prompt_path} does not define callable get_prompt()")
    return get_prompt


class DatasetPromptTable:
    def __init__(self, lerobot_root: Path) -> None:
        self.lerobot_root = lerobot_root
        self.by_task_seed: dict[tuple[int, int], str] = {}
        self.by_task: dict[int, list[str]] = {}
        for path in sorted(lerobot_root.glob("*/meta/robomemarena_provenance.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    task_id = int(row["task_id"])
                    seed = int(row["seed"])
                    prompt = str(row["prompt"])
                    self.by_task_seed[(task_id, seed)] = prompt
                    prompts = self.by_task.setdefault(task_id, [])
                    if prompt not in prompts:
                        prompts.append(prompt)
        if not self.by_task_seed:
            raise FileNotFoundError(f"No robomemarena provenance files found under {lerobot_root}")

    def get(self, task_id: int, seed: int) -> str:
        prompt = self.by_task_seed.get((int(task_id), int(seed)))
        if prompt is not None:
            return prompt
        prompts = self.by_task.get(int(task_id), [])
        if prompts:
            logging.warning(
                "No exact dataset prompt for task=%s seed=%s; using first task prompt: %s",
                task_id,
                seed,
                prompts[0],
            )
            return prompts[0]
        raise KeyError(f"No dataset prompt found for task={task_id} seed={seed} in {self.lerobot_root}")


def _patch_prompt_fn(ec, tasks26, run_all, prompt_fn) -> None:
    ec.get_prompt = prompt_fn
    tasks26.ec.get_prompt = prompt_fn
    run_all.ec.get_prompt = prompt_fn


def _combine_single_episode_results(task_id: int, parts: list[dict[str, Any]], video_dir: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for ep, part in enumerate(parts):
        item = dict(part["episodes"][0])
        item["ep"] = ep
        episodes.append(item)

    n = max(1, len(episodes))
    avg_score = sum(float(ep.get("score_pct", 0.0)) for ep in episodes) / n
    tsr = 100.0 * sum(bool(ep.get("tsr_success", False)) for ep in episodes) / n
    stage = 100.0 * sum(bool(ep.get("stage_success", ep.get("tsr_success", False))) for ep in episodes) / n
    goal_values = [ep.get("goal_success") for ep in episodes if ep.get("goal_success") is not None]
    goal = None if not goal_values else 100.0 * sum(bool(x) for x in goal_values) / len(goal_values)
    return {
        "task_id": task_id,
        "task_key": f"task{task_id}",
        "prompt": "dataset_prompt_per_seed",
        "bddl_path": parts[0].get("bddl_path", "") if parts else "",
        "video_dir": str(video_dir),
        "average_score_pct": float(avg_score),
        "tsr_success_rate_pct": float(tsr),
        "stage_success_rate_pct": float(stage),
        "goal_success_rate_pct": None if goal is None else float(goal),
        "episodes": episodes,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robomemarena-root", default="/home/liuyuyan/RoboMemArena")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6697)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--use-ddim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--twochunk-debug", action="store_true")
    parser.add_argument("--inference-warmup-steps", type=int, default=10)
    parser.add_argument("--adapter-spec", default="")
    parser.add_argument("--adapter-kwargs", default="")
    parser.add_argument("--subset", choices=sorted(SUBSET_TASKS), default="all")
    parser.add_argument("--task-ids", default=None, help="Comma/range list, e.g. '1,2,6-10'. Overrides --subset.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument("--extra-pour-monitor-steps", type=int, default=30)
    parser.add_argument("--fail-on-extra-pour", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--prompt-source", choices=("openpi", "dataset", "bddl"), default="openpi")
    parser.add_argument("--lerobot-root", default="/15T/liuyuyan/robomemarena_lerobot")
    parser.add_argument("--out-root", required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    robomem_root = Path(args.robomemarena_root).resolve()
    scripts_dir = robomem_root / "evaluation_benchmark" / "scripts"
    libero_fork = robomem_root / "evaluation_benchmark" / "libero_fork"
    for path in (str(repo_root), str(scripts_dir), str(libero_fork)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import eval_common as ec
    import eval_task1_only as task1_eval
    import eval_tasks2_26 as tasks26
    import run_all_tasks1_26 as run_all
    from policy_adapter import load_policy_adapter

    openpi_get_prompt = _load_openpi_prompt_fn(robomem_root)
    dataset_prompts = None
    if args.prompt_source == "openpi":
        _patch_prompt_fn(ec, tasks26, run_all, openpi_get_prompt)
        logging.info(
            "Using OpenPI prompt table: %s",
            robomem_root / "evaluation_benchmark" / "openpi_minimal_runtime" / "task_prompts.py",
        )
    elif args.prompt_source == "dataset":
        dataset_prompts = DatasetPromptTable(Path(args.lerobot_root).resolve())
        logging.info("Using per-seed dataset prompts from: %s", dataset_prompts.lerobot_root)
    else:
        logging.info("Using default RoboMemArena BDDL-stem prompts from scripts/eval_common.py")

    adapter_spec = args.adapter_spec or str(Path(__file__).with_name("model2robomemarena_twochunk_adapter.py"))
    adapter_kwargs = {
        "host": args.host,
        "port": args.port,
        "unnorm_key": args.unnorm_key,
        "use_ddim": args.use_ddim,
        "num_ddim_steps": args.num_ddim_steps,
        "twochunk_debug": args.twochunk_debug,
        "inference_warmup_steps": args.inference_warmup_steps,
    }
    adapter_kwargs.update(_json_loads_object(args.adapter_kwargs))
    adapter = load_policy_adapter(adapter_spec, **adapter_kwargs)

    task_ids = _parse_task_ids(args.task_ids, args.subset)
    out_root = Path(args.out_root)
    video_root = out_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    tasks26._patch_env_resolution()
    results: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            _, task_key = ec._resolve_task_id(task_id)
            bddl_path = ec._resolve_bddl_path(task_id)
            prompt = ec.get_prompt(task_key, bddl_path.stem)
            video_dir = video_root / f"task{task_id}"
            logging.info(
                "task=%s seed_start=%s trials=%s prompt=%s replan_steps=%s",
                task_id,
                args.seed,
                args.num_trials_per_task,
                prompt,
                args.replan_steps,
            )
            if dataset_prompts is None:
                results.append(
                    run_all._run_task(
                        task_id=task_id,
                        adapter=adapter,
                        num_trials_per_task=args.num_trials_per_task,
                        resize_size=args.resize_size,
                        replan_steps=args.replan_steps,
                        num_steps_wait=args.num_steps_wait,
                        max_steps=args.max_steps,
                        post_goal_steps=args.post_goal_steps,
                        fail_on_extra_pour=args.fail_on_extra_pour,
                        extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                        video_dir=video_dir,
                        seed=args.seed,
                    )
                )
                continue

            task_parts: list[dict[str, Any]] = []
            for ep in range(args.num_trials_per_task):
                current_seed = args.seed + ep
                prompt = dataset_prompts.get(task_id, current_seed)
                _patch_prompt_fn(ec, tasks26, run_all, lambda _task_key, _stem, p=prompt: p)
                logging.info(
                    "task=%s ep=%s seed=%s dataset_prompt=%s",
                    task_id,
                    ep,
                    current_seed,
                    prompt,
                )
                task_parts.append(
                    run_all._run_task(
                        task_id=task_id,
                        adapter=adapter,
                        num_trials_per_task=1,
                        resize_size=args.resize_size,
                        replan_steps=args.replan_steps,
                        num_steps_wait=args.num_steps_wait,
                        max_steps=args.max_steps,
                        post_goal_steps=args.post_goal_steps,
                        fail_on_extra_pour=args.fail_on_extra_pour,
                        extra_pour_monitor_steps=args.extra_pour_monitor_steps,
                        video_dir=video_dir,
                        seed=current_seed,
                    )
                )
            results.append(_combine_single_episode_results(task_id, task_parts, video_dir))
        run_all._write_outputs(out_root, results, args.seed)

        stats_fn = getattr(adapter, "get_inference_stats", None)
        if callable(stats_fn):
            stats_path = out_root / "adapter_inference_stats.json"
            stats_path.write_text(json.dumps(stats_fn(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
