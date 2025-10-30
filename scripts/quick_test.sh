#!/usr/bin/env bash
set -euo pipefail

# Minimal non-interactive smoke tests for graders/CI.
# Assumes the package is already installed (editable or not) in the active env.

echo "[GYM] Running random-mode demo (headless)..."
parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --quiet || {
  echo "Random-mode demo failed." >&2
  exit 1
}

echo "[GYM] Generating a randomized config and running a second short rollout..."
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_quick.json --seed 123

parking-gym-demo --mode random --episodes 1 --max-steps 200 \
  --config parking_project_submission/configs/train_quick.json --no-visualize --quiet || {
  echo "Second rollout failed." >&2
  exit 1
}

echo "[NN] Running neural-network demo (Lidar+Residual+LSTM, T=1)..."
python -m parking_project_submission.neural_network_demo --seq-len 1 || {
  echo "Neural-network demo (T=1) failed." >&2
  exit 1
}

echo "[NN] Running neural-network demo (Lidar+Residual+LSTM, T=8)..."
python -m parking_project_submission.neural_network_demo --seq-len 8 || {
  echo "Neural-network demo (T=8) failed." >&2
  exit 1
}

# Optional ONNX export if onnx is available
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('onnx') else 1)
PY
then
  echo "[NN] Exporting ONNX for NN demo (optional)..."
  python -m parking_project_submission.neural_network_demo \
    --seq-len 4 --export-onnx artifacts/model_lstm.onnx || true
else
  echo "ONNX not installed; skipping export step."
fi

echo "All quick tests passed."
