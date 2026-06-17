#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${BACKBONE:-dinov2_vits14}"
TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

case "$BACKBONE" in
  dinov2_vits14|dinov2_vitb14|dinov2_vitl14|dinov2_vitg14)
    ;;
  *)
    echo "Unsupported BACKBONE=$BACKBONE" >&2
    echo "Choose one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14" >&2
    exit 1
    ;;
esac

HUB_DIR="$TORCH_HOME/hub"
CHECKPOINT_DIR="$HUB_DIR/checkpoints"
CODE_DIR="$HUB_DIR/facebookresearch_dinov2_main"
WEIGHT_FILE="$CHECKPOINT_DIR/${BACKBONE}_pretrain.pth"
WEIGHT_URL="https://dl.fbaipublicfiles.com/dinov2/${BACKBONE}/${BACKBONE}_pretrain.pth"

mkdir -p "$HUB_DIR" "$CHECKPOINT_DIR"

echo "[dinov2] TORCH_HOME=$TORCH_HOME"
echo "[dinov2] code dir: $CODE_DIR"
echo "[dinov2] weight:   $WEIGHT_FILE"

if [ ! -d "$CODE_DIR" ]; then
  echo "[dinov2] cloning facebookresearch/dinov2"
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$CODE_DIR"
else
  echo "[dinov2] code already exists, skipping clone"
fi

if [ ! -f "$WEIGHT_FILE" ]; then
  echo "[dinov2] downloading $BACKBONE weights"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --output "$WEIGHT_FILE" "$WEIGHT_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$WEIGHT_FILE" "$WEIGHT_URL"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
else
  echo "[dinov2] weights already exist, skipping download"
fi

echo "[dinov2] done"
echo "[dinov2] dino.py will load this local cache first."
