"""Utilities for saving source-code snapshots alongside experiment outputs."""

import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path

from accelerate.logging import get_logger

logger = get_logger(__name__)


def _log_info(message: str) -> None:
    try:
        logger.info(message)
    except RuntimeError:
        print(message)


def _log_warning(message: str) -> None:
    try:
        logger.warning(message)
    except RuntimeError:
        print(f"WARNING: {message}")


def save_framework_source_snapshot(model, output_dir, framework_name=None) -> None:
    """Copy the active framework implementation into the run directory.

    The snapshot follows the model class MRO and keeps only source files under
    ``starVLA/model/framework``. This captures wrapper variants plus their
    framework parents without pulling unrelated package code into every run.
    """
    output_dir = Path(output_dir)
    framework_root = Path(__file__).resolve().parents[2] / "model" / "framework"
    snapshot_root = output_dir / "framework_source"

    copied = []
    seen_sources = set()
    for cls in inspect.getmro(type(model)):
        try:
            source = inspect.getsourcefile(cls)
        except (TypeError, OSError):
            continue
        if source is None:
            continue

        source_path = Path(source).resolve()
        try:
            relative_path = source_path.relative_to(framework_root)
        except ValueError:
            continue

        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)

        snapshot_path = snapshot_root / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot_path)
        copied.append(
            {
                "class_name": cls.__name__,
                "module": cls.__module__,
                "source_path": str(source_path),
                "snapshot_path": str(snapshot_path),
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "framework_name": framework_name,
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "copied_sources": copied,
    }
    snapshot_root.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if copied:
        _log_info(f"Framework source snapshot saved at {snapshot_root}")
    else:
        _log_warning(f"No framework source files found to snapshot for {type(model).__name__}")
