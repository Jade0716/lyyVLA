#!/usr/bin/env python3
"""Train a small MLP to classify CALVIN task verbs from action DCT features."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np 
import pandas as pd
from scipy.fft import dct
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


CLASSES = ("pick", "place", "push", "pull", "open", "close")

EXACT_RULES = {name: (rf"\b{name}\w*\b",) for name in CLASSES}
SYNONYM_RULES = {
    "pick": (
        r"\bpick\w*\b",
        r"\blift\w*\b",
        r"\btake\w*\b",
        r"\bremove\w*\b",
        r"\bunstack\w*\b",
        r"\bcollapse\w*\b",
        r"(?=.*\bgrasp\b)(?=.*\bblock\b)",
    ),
    "place": (
        r"\bplace\w*\b",
        r"\bput\w*\b",
        r"\bstore\w*\b",
        r"^\s*stack\b",
    ),
    "push": (
        r"\bpush\w*\b",
        r"\bpress\w*\b",
        r"\bsweep\w*\b",
        r"\bslide\b(?=.*\b(?:block|object)\b)",
        r"\bgo slide\b(?=.*\bblock\b)",
    ),
    "pull": (r"\bpull\w*\b",),
    "open": (r"\bopen\w*\b",),
    "close": (r"\bclose\w*\b",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/16T/liuyuyan/calvin-abc-d-lerobot"))
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--dct-dim", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--action-key", type=str, default="relative_action")
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--align-mode", choices=("gripper_close", "none"), default="gripper_close")
    parser.add_argument("--gripper-width-index", type=int, default=6)
    parser.add_argument("--gripper-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-min-run", type=int, default=3)
    parser.add_argument("--pad-mode", choices=("last", "zero"), default="last")
    parser.add_argument("--label-mode", choices=("exact", "synonym"), default="exact")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/calvin/dct_task_mlp"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def match_label(text: str, rules: dict[str, tuple[str, ...]]) -> str | None:
    text = text.lower()
    matched = [label for label, patterns in rules.items() if any(re.search(p, text) for p in patterns)]
    return matched[0] if len(matched) == 1 else None


def episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def dct_feature(actions: np.ndarray, dct_dim: int) -> np.ndarray:
    coeff = dct(actions.astype(np.float32), type=2, axis=0, norm="ortho")
    if coeff.shape[0] < dct_dim:
        pad = np.zeros((dct_dim - coeff.shape[0], coeff.shape[1]), dtype=np.float32)
        coeff = np.concatenate([coeff, pad], axis=0)
    return coeff[:dct_dim].reshape(-1).astype(np.float32)


def first_gripper_close_frame(gripper_width: np.ndarray, threshold: float, min_run: int) -> int | None:
    closed = gripper_width < threshold
    if min_run <= 1:
        hits = np.flatnonzero(closed)
    elif len(closed) >= min_run:
        run_counts = np.convolve(closed.astype(np.int32), np.ones(min_run, dtype=np.int32), mode="valid")
        hits = np.flatnonzero(run_counts >= min_run)
    else:
        hits = np.array([], dtype=np.int64)
    return None if len(hits) == 0 else int(hits[0])


def fixed_action_chunk(actions: np.ndarray, start_frame: int, chunk_len: int, pad_mode: str) -> np.ndarray:
    chunk = actions[start_frame : start_frame + chunk_len]
    if len(chunk) == 0:
        raise ValueError(f"Empty action chunk: start_frame={start_frame}, trajectory_len={len(actions)}")
    if len(chunk) == chunk_len:
        return chunk

    if pad_mode == "last":
        pad = np.repeat(chunk[-1:], chunk_len - len(chunk), axis=0)
    elif pad_mode == "zero":
        pad = np.zeros((chunk_len - len(chunk), actions.shape[1]), dtype=actions.dtype)
    else:
        raise ValueError(f"Unknown pad_mode={pad_mode!r}")
    return np.concatenate([chunk, pad], axis=0)


def build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    episodes = load_jsonl(args.dataset_root / "meta" / "episodes.jsonl")[: args.num_episodes]
    rules = EXACT_RULES if args.label_mode == "exact" else SYNONYM_RULES

    features: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict] = []
    skipped = Counter()
    start_frames: list[int] = []

    for ep in tqdm(episodes, desc="Loading trajectories"):
        episode_index = int(ep["episode_index"])
        instruction = " ".join(ep["tasks"])
        label = match_label(instruction, rules)
        if label is None:
            skipped["unmatched_or_ambiguous"] += 1
            continue

        parquet_path = episode_path(args.dataset_root, episode_index)
        if not parquet_path.exists():
            skipped["missing_parquet"] += 1
            continue

        columns = [args.action_key]
        if args.align_mode == "gripper_close":
            columns.append(args.state_key)
        df = pd.read_parquet(parquet_path, columns=columns)
        actions = np.stack(df[args.action_key].to_numpy()).astype(np.float32)

        if args.align_mode == "gripper_close":
            states = np.stack(df[args.state_key].to_numpy())
            gripper_width = states[:, args.gripper_width_index].astype(np.float32)
            start_frame = first_gripper_close_frame(
                gripper_width=gripper_width,
                threshold=args.gripper_threshold,
                min_run=args.gripper_min_run,
            )
            if start_frame is None:
                skipped["no_gripper_close"] += 1
                continue
            gripper_width_at_start = float(gripper_width[start_frame])
        else:
            start_frame = 0
            gripper_width_at_start = None

        action_chunk = fixed_action_chunk(
            actions=actions,
            start_frame=start_frame,
            chunk_len=args.chunk_len,
            pad_mode=args.pad_mode,
        )
        features.append(dct_feature(action_chunk, args.dct_dim))
        labels.append(CLASSES.index(label))
        start_frames.append(start_frame)
        rows.append(
            {
                "episode_index": episode_index,
                "instruction": instruction,
                "label": label,
                "length": int(len(actions)),
                "start_frame": int(start_frame),
                "chunk_len": int(args.chunk_len),
                "gripper_width_at_start": gripper_width_at_start,
            }
        )

    start_frame_percentiles = {}
    if start_frames:
        start_frame_percentiles = {
            str(p): float(np.percentile(start_frames, p)) for p in (0, 5, 10, 25, 50, 75, 90, 95, 100)
        }

    meta = {
        "num_requested": len(episodes),
        "num_labeled": len(labels),
        "skipped": dict(skipped),
        "class_counts": dict(Counter(CLASSES[y] for y in labels)),
        "label_mode": args.label_mode,
        "action_key": args.action_key,
        "align_mode": args.align_mode,
        "chunk_len": args.chunk_len,
        "pad_mode": args.pad_mode,
        "state_key": args.state_key if args.align_mode == "gripper_close" else None,
        "gripper_width_index": args.gripper_width_index if args.align_mode == "gripper_close" else None,
        "gripper_threshold": args.gripper_threshold if args.align_mode == "gripper_close" else None,
        "gripper_min_run": args.gripper_min_run if args.align_mode == "gripper_close" else None,
        "start_frame_percentiles": start_frame_percentiles,
        "dct_dim": args.dct_dim,
    }
    if not features:
        raise RuntimeError(f"No labeled samples were built. Skipped counts: {dict(skipped)}")
    return np.stack(features), np.asarray(labels, dtype=np.int64), rows, meta


def stratified_split(y: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(len(cls_indices) * test_ratio)))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


class DctMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    return device


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[DctMLP, list[dict[str, float]], torch.device]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = DctMLP(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, num_classes=len(CLASSES)).to(device)

    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_test_t = torch.from_numpy(x_test.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )

    class_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float32)
    class_weights = len(y_train) / (len(CLASSES) * np.maximum(class_counts, 1.0))
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training MLP"):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            train_pred = predict(model, x_train_t, device)
            test_pred = predict(model, x_test_t, device)
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(np.mean(losses)),
                    "train_acc": accuracy(train_pred, y_train),
                    "test_acc": accuracy(test_pred, y_test),
                }
            )
    return model, history, device


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(x.to(device))
    return logits.argmax(dim=1).cpu().numpy()


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def confusion_matrix(pred: np.ndarray, target: np.ndarray) -> list[list[int]]:
    mat = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for y, p in zip(target, pred):
        mat[int(y), int(p)] += 1
    return mat.tolist()


def per_class_accuracy(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for idx, name in enumerate(CLASSES):
        mask = target == idx
        scores[name] = None if not mask.any() else float((pred[mask] == target[mask]).mean())
    return scores


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y, rows, meta = build_dataset(args)
    train_idx, test_idx = stratified_split(y, args.test_ratio, args.seed)
    x_train, x_test, mean, std = standardize(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    model, history, device = train_mlp(x_train, y_train, x_test, y_test, args)
    train_pred = predict(model, torch.from_numpy(x_train.astype(np.float32)), device)
    test_pred = predict(model, torch.from_numpy(x_test.astype(np.float32)), device)

    metrics = {
        **meta,
        "feature_dim": int(x.shape[1]),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "model": "torch_mlp",
        "hidden_dim": int(args.hidden_dim),
        "device": str(device),
        "train_accuracy": accuracy(train_pred, y_train),
        "test_accuracy": accuracy(test_pred, y_test),
        "test_per_class_accuracy": per_class_accuracy(test_pred, y_test),
        "confusion_matrix_labels": list(CLASSES),
        "confusion_matrix": confusion_matrix(test_pred, y_test),
        "history": history,
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_dir / "labeled_episodes.csv", index=False)
    np.savez_compressed(
        args.out_dir / "features.npz",
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        mean=mean,
        std=std,
    )
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": CLASSES,
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "args": vars(args),
        },
        args.out_dir / "torch_mlp.pt",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
