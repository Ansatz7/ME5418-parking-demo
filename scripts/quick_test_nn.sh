#!/usr/bin/env bash
set -euo pipefail

echo "[NN] Neural-network demo (Lidar+Residual+LSTM) quick checks"

echo "[NN] T=1 (CPU)"
python -m parking_project_submission.neural_network_demo --seq-len 1 || {
  echo "[NN] Demo failed (T=1)" >&2; exit 1;
}

echo "[NN] T=8 (CPU)"
python -m parking_project_submission.neural_network_demo --seq-len 8 || {
  echo "[NN] Demo failed (T=8)" >&2; exit 1;
}

# Optional: export ONNX if available
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('onnx') else 1)
PY
then
  echo "[NN] Exporting ONNX (optional)"
  python -m parking_project_submission.neural_network_demo --seq-len 4 --export-onnx artifacts/model_lstm.onnx || true
else
  echo "[NN] ONNX not installed; skipping export"
fi

echo "[NN] Done"

