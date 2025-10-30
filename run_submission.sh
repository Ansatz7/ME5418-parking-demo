#!/usr/bin/env bash
set -euo pipefail

# Single entrypoint for evaluation. No arguments required.
# Runs a short headless rollout using the default demo configuration.

echo "Parking Environment Demo: evaluation run"
parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --quiet || {
  echo "Evaluation run failed." >&2
  exit 1
}

echo "Neural Network Demo: Lidar+Residual+LSTM (short sequence)"
python -m parking_project_submission.neural_network_demo --seq-len 4 || {
  echo "Neural network demo failed." >&2
  exit 1
}

echo "Done."
