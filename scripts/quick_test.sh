#!/usr/bin/env bash
set -euo pipefail

# Minimal non-interactive smoke tests for graders/CI.
# Assumes the package is already installed (editable or not) in the active env.

echo "Running random-mode demo (headless)..."
parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --quiet || {
  echo "Random-mode demo failed." >&2
  exit 1
}

echo "Generating a randomized config and running a second short rollout..."
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_quick.json --seed 123

parking-gym-demo --mode random --episodes 1 --max-steps 200 \
  --config parking_project_submission/configs/train_quick.json --no-visualize --quiet || {
  echo "Second rollout failed." >&2
  exit 1
}

echo "All quick tests passed."

