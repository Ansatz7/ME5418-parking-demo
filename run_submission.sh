#!/usr/bin/env bash
set -euo pipefail

# Single entrypoint for evaluation. No arguments required.
# Runs a short headless rollout using the default demo configuration.

echo "Parking Environment Demo: evaluation run (visualized random rollout)"
parking-gym-demo --mode random --episodes 1 --max-steps 400 --per-step || {
  echo "Evaluation run (GYM) failed." >&2
  exit 1
}

echo "Neural Network Demo: Lidar+Residual+LSTM (short sequence)"
python -m parking_project_submission.neural_network_demo --seq-len 4 || {
  echo "Neural network demo failed." >&2
  exit 1
}

echo "Done."
