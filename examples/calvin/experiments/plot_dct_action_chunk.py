#!/usr/bin/env python3
"""Plot CALVIN action chunks after low-frequency DCT reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dct, idct


ACTION_DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
CURVE_COLORS = (
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one CALVIN action chunk and its low-frequency DCT reconstructions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/16T/liuyuyan/calvin-abc-d-lerobot"),
        help="LeRobot-format CALVIN dataset root.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=4,
        help="Episode id to read from meta/data.",
    )
    parser.add_argument(
        "--chunk-start",
        "--start-frame",
        dest="chunk_start",
        type=int,
        default=20,
        help="Start frame of the action chunk inside the selected episode.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=8,
        help="Number of action steps in the chunk.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="relative_action",
        help="Parquet column containing 7D actions.",
    )
    parser.add_argument(
        "--dct-keep-dims",
        "--keep-dims",
        dest="dct_keep_dims",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="DCT coefficient counts to keep before inverse reconstruction.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Defaults to dct_action_chunk{chunk_len}_plots.",
    )
    parser.add_argument(
        "--hide-original",
        "--no-original",
        dest="show_original",
        action="store_false",
        help="Only plot DCT reconstructions, without the original action curve.",
    )
    parser.set_defaults(show_original=True)
    return parser.parse_args()


def episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def load_action_chunk(args: argparse.Namespace) -> np.ndarray:
    parquet_path = episode_path(args.dataset_root, args.episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path, columns=[args.action_key])
    actions = np.stack(df[args.action_key].to_numpy()).astype(np.float32)
    end_frame = args.chunk_start + args.chunk_len
    if args.chunk_start < 0 or end_frame > len(actions):
        raise ValueError(
            f"Requested frames [{args.chunk_start}, {end_frame}) from trajectory "
            f"with length {len(actions)}"
        )
    return actions[args.chunk_start:end_frame]


def dct_reconstructions(chunk: np.ndarray, keep_dims: list[int]) -> dict[int, np.ndarray]:
    coeff = dct(chunk, type=2, axis=0, norm="ortho")
    reconstructions = {}
    for keep_dim in keep_dims:
        truncated = np.zeros_like(coeff)
        truncated[:keep_dim] = coeff[:keep_dim]
        reconstructions[keep_dim] = idct(truncated, type=2, axis=0, norm="ortho")
    return reconstructions


def color_for_index(index: int) -> str:
    return CURVE_COLORS[index % len(CURVE_COLORS)]


def plot_with_matplotlib(
    x: np.ndarray,
    original: np.ndarray,
    reconstructions: dict[int, np.ndarray],
    dim: int,
    title: str,
    out_path: Path,
    include_original: bool,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 4.5), dpi=160)
    if include_original:
        plt.plot(x, original[:, dim], marker="o", linewidth=2.4, color="#111827", label="original")
    for i, (keep_dim, recon) in enumerate(reconstructions.items()):
        plt.plot(
            x,
            recon[:, dim],
            marker="o",
            linewidth=2.0,
            color=color_for_index(i),
            label=f"DCT first {keep_dim}",
        )
    plt.title(title)
    plt.xlabel("chunk step")
    plt.ylabel("action value")
    plt.xticks(x)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_all_with_matplotlib(
    x: np.ndarray,
    original: np.ndarray,
    reconstructions: dict[int, np.ndarray],
    title: str,
    out_path: Path,
    include_original: bool,
) -> None:
    import matplotlib.pyplot as plt

    n_dims = len(ACTION_DIM_NAMES)
    n_cols = 2
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows), dpi=160, squeeze=False)
    axes_flat = axes.reshape(-1)

    for dim, name in enumerate(ACTION_DIM_NAMES):
        ax = axes_flat[dim]
        if include_original:
            ax.plot(x, original[:, dim], marker="o", linewidth=2.2, color="#111827", label="original")
        for i, (keep_dim, recon) in enumerate(reconstructions.items()):
            ax.plot(
                x,
                recon[:, dim],
                marker="o",
                linewidth=1.8,
                color=color_for_index(i),
                label=f"DCT first {keep_dim}",
            )
        ax.set_title(f"dim {dim}: {name}")
        ax.set_xlabel("chunk step")
        ax.set_ylabel("action value")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n_dims:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def nice_range(values: np.ndarray) -> tuple[float, float]:
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if np.isclose(ymin, ymax):
        pad = max(abs(ymin) * 0.1, 1.0)
    else:
        pad = (ymax - ymin) * 0.12
    return ymin - pad, ymax + pad


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    x_positions: list[float],
    ymin: float,
    ymax: float,
    plot_top: int,
    plot_bottom: int,
    color: str,
    width: int,
) -> None:
    scale = plot_bottom - plot_top
    points = [
        (x_positions[i], plot_bottom - ((float(v) - ymin) / (ymax - ymin)) * scale)
        for i, v in enumerate(values)
    ]
    draw.line(points, fill=color, width=width, joint="curve")
    r = 4
    for px, py in points:
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def plot_with_pil(
    x: np.ndarray,
    original: np.ndarray,
    reconstructions: dict[int, np.ndarray],
    dim: int,
    title: str,
    out_path: Path,
    include_original: bool,
) -> None:
    width, height = 1100, 680
    left, right, top, bottom = 105, 45, 80, 110
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=18) if hasattr(ImageFont.load_default(), "size") else font

    curves = [recon[:, dim] for recon in reconstructions.values()]
    if include_original:
        curves.append(original[:, dim])
    ymin, ymax = nice_range(np.concatenate(curves))
    x_positions = [
        plot_left + (plot_right - plot_left) * i / max(1, len(x) - 1)
        for i in range(len(x))
    ]

    for frac in np.linspace(0, 1, 6):
        y = plot_bottom - frac * (plot_bottom - plot_top)
        value = ymin + frac * (ymax - ymin)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        label = f"{value:.3g}"
        draw.text((plot_left - text_width(draw, label, font) - 10, y - 7), label, fill="#374151", font=font)

    for i, step in enumerate(x):
        px = x_positions[i]
        draw.line((px, plot_top, px, plot_bottom), fill="#f3f4f6", width=1)
        draw.text((px - 4, plot_bottom + 14), str(int(step)), fill="#374151", font=font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
    draw.text(((width - text_width(draw, title, title_font)) / 2, 26), title, fill="#111827", font=title_font)
    draw.text(((width - text_width(draw, "chunk step", font)) / 2, height - 44), "chunk step", fill="#111827", font=font)
    draw.text((14, (height - text_width(draw, "action value", font)) / 2), "action value", fill="#111827", font=font)

    legend_items: list[tuple[str, str]] = []
    if include_original:
        draw_polyline(draw, original[:, dim], x_positions, ymin, ymax, plot_top, plot_bottom, "#111827", 4)
        legend_items.append(("original", "#111827"))
    for i, (keep_dim, recon) in enumerate(reconstructions.items()):
        color = color_for_index(i)
        draw_polyline(draw, recon[:, dim], x_positions, ymin, ymax, plot_top, plot_bottom, color, 3)
        legend_items.append((f"DCT first {keep_dim}", color))

    legend_x, legend_y = plot_left, height - 78
    for label, color in legend_items:
        draw.line((legend_x, legend_y + 7, legend_x + 34, legend_y + 7), fill=color, width=4)
        draw.text((legend_x + 42, legend_y), label, fill="#111827", font=font)
        legend_x += 42 + text_width(draw, label, font) + 34

    image.save(out_path)


def plot_all_with_pil(
    x: np.ndarray,
    original: np.ndarray,
    reconstructions: dict[int, np.ndarray],
    title: str,
    out_path: Path,
    include_original: bool,
) -> None:
    panel_width, panel_height = 900, 440
    n_cols = 2
    n_rows = int(np.ceil(len(ACTION_DIM_NAMES) / n_cols))
    title_height = 70
    legend_height = 70
    width = panel_width * n_cols
    height = title_height + panel_height * n_rows + legend_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=18) if hasattr(ImageFont.load_default(), "size") else font

    draw.text(((width - text_width(draw, title, title_font)) / 2, 24), title, fill="#111827", font=title_font)

    for dim, name in enumerate(ACTION_DIM_NAMES):
        col = dim % n_cols
        row = dim // n_cols
        panel_x = col * panel_width
        panel_y = title_height + row * panel_height

        left, right, top, bottom = 88, 36, 48, 70
        plot_left = panel_x + left
        plot_right = panel_x + panel_width - right
        plot_top = panel_y + top
        plot_bottom = panel_y + panel_height - bottom

        curves = [recon[:, dim] for recon in reconstructions.values()]
        if include_original:
            curves.append(original[:, dim])
        ymin, ymax = nice_range(np.concatenate(curves))
        x_positions = [
            plot_left + (plot_right - plot_left) * i / max(1, len(x) - 1)
            for i in range(len(x))
        ]

        for frac in np.linspace(0, 1, 5):
            y = plot_bottom - frac * (plot_bottom - plot_top)
            value = ymin + frac * (ymax - ymin)
            draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
            label = f"{value:.3g}"
            draw.text((plot_left - text_width(draw, label, font) - 8, y - 7), label, fill="#374151", font=font)

        for i, step in enumerate(x):
            px = x_positions[i]
            draw.line((px, plot_top, px, plot_bottom), fill="#f3f4f6", width=1)
            draw.text((px - 4, plot_bottom + 12), str(int(step)), fill="#374151", font=font)

        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
        panel_title = f"dim {dim}: {name}"
        draw.text((panel_x + 20, panel_y + 16), panel_title, fill="#111827", font=font)

        if include_original:
            draw_polyline(draw, original[:, dim], x_positions, ymin, ymax, plot_top, plot_bottom, "#111827", 3)
        for i, recon in enumerate(reconstructions.values()):
            draw_polyline(draw, recon[:, dim], x_positions, ymin, ymax, plot_top, plot_bottom, color_for_index(i), 2)

    legend_items: list[tuple[str, str]] = []
    if include_original:
        legend_items.append(("original", "#111827"))
    for i, keep_dim in enumerate(reconstructions.keys()):
        legend_items.append((f"DCT first {keep_dim}", color_for_index(i)))

    legend_x = 40
    legend_y = height - 46
    for label, color in legend_items:
        draw.line((legend_x, legend_y + 7, legend_x + 34, legend_y + 7), fill=color, width=4)
        draw.text((legend_x + 42, legend_y), label, fill="#111827", font=font)
        legend_x += 42 + text_width(draw, label, font) + 34

    image.save(out_path)


def main() -> None:
    args = parse_args()
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be positive")
    if args.chunk_start < 0:
        raise ValueError("--chunk-start must be non-negative")
    if any(keep_dim <= 0 or keep_dim > args.chunk_len for keep_dim in args.dct_keep_dims):
        raise ValueError(f"--dct-keep-dims must be in [1, {args.chunk_len}], got {args.dct_keep_dims}")
    if args.out_dir is None:
        args.out_dir = Path(f"examples/calvin/experiments/dct_action_chunk{args.chunk_len}_plots")

    chunk = load_action_chunk(args)
    if chunk.shape[1] != 7:
        raise ValueError(f"Expected 7 action dimensions, got shape {chunk.shape}")

    reconstructions = dct_reconstructions(chunk, args.dct_keep_dims)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(args.chunk_len)
    include_original = args.show_original

    try:
        import matplotlib  # noqa: F401

        backend = "matplotlib"
        plot_fn = plot_all_with_matplotlib
    except Exception:
        backend = "PIL"
        plot_fn = plot_all_with_pil

    keep_dims_label = "_".join(str(keep_dim) for keep_dim in args.dct_keep_dims)
    out_path = (
        args.out_dir
        / (
            f"ep{args.episode_index:06d}_start{args.chunk_start}_"
            f"all_action_dims_chunk{args.chunk_len}_dct_{keep_dims_label}.png"
        )
    )
    title = (
        f"{args.action_key}, chunk {args.chunk_len}, episode {args.episode_index}, "
        f"frames {args.chunk_start}-{args.chunk_start + args.chunk_len - 1}, "
        f"DCT {keep_dims_label}"
    )
    plot_fn(x, chunk, reconstructions, title, out_path, include_original)
    outputs = [str(out_path)]

    meta = {
        "dataset_root": str(args.dataset_root),
        "episode_index": args.episode_index,
        "chunk_start": args.chunk_start,
        "chunk_len": args.chunk_len,
        "action_key": args.action_key,
        "dct_keep_dims": args.dct_keep_dims,
        "plot_backend": backend,
        "show_original": include_original,
        "outputs": outputs,
    }
    meta_path = args.out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nSaved combined plot and metadata to {args.out_dir}")


if __name__ == "__main__":
    main()
