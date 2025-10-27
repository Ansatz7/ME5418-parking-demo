#!/usr/bin/env bash
set -euo pipefail

# Create/update the project environment and install the package in editable mode.
# - Prefers mamba if available, otherwise falls back to conda.
# - Does not require shell activation; uses `run -n` to install inside the env.

ENV_NAME=${ENV_NAME:-parking-rl}
PY_VER=${PY_VER:-3.10}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

PM=""
if has_cmd mamba; then
  PM=mamba
elif has_cmd conda; then
  PM=conda
else
  echo "Neither mamba nor conda found in PATH." >&2
  exit 1
fi

echo "Using package manager: $PM"

# Create env if missing
if ! $PM env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating env '$ENV_NAME' with Python $PY_VER..."
  $PM create -n "$ENV_NAME" python="$PY_VER" pip -y
else
  echo "Env '$ENV_NAME' already exists; skipping creation."
fi

echo "Upgrading pip inside '$ENV_NAME'..."
$PM run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "Installing project in editable mode inside '$ENV_NAME'..."
$PM run -n "$ENV_NAME" pip install -e .

echo "Done. To use the environment:"
echo "  $PM activate $ENV_NAME"
echo "Then run demos, e.g.:"
echo "  parking-gym-demo --mode random --episodes 1 --max-steps 400"

