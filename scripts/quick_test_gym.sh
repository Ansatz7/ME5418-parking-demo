#!/usr/bin/env bash
set -euo pipefail

echo "[GYM] Environment quick checks"

echo "[GYM] Random rollout (headless)"
parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --quiet || {
  echo "[GYM] Random rollout failed" >&2; exit 1;
}

echo "[GYM] Generate randomized config"
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_quick.json --seed 123

echo "[GYM] Rollout with randomized config (headless)"
parking-gym-demo --mode random --episodes 1 --max-steps 200 \
  --config parking_project_submission/configs/train_quick.json --no-visualize --quiet || {
  echo "[GYM] Rollout with randomized config failed" >&2; exit 1;
}

echo "[GYM] Done"

