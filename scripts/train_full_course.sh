#!/usr/bin/env bash
# Recurrent PPO curriculum over multiple randomized maps.
#
# EN: This script generates 5 training maps and trains the agent sequentially,
# carrying the checkpoint from one map to the next.
# ZH: 本脚本先生成 5 张训练地图，然后按顺序进行训练，每一轮把上一轮
# 的模型权重作为下一轮的初始点，最终得到 model_5.pt。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_CFG_DIR="parking_project_submission/configs/train_course"
mkdir -p "$OUT_CFG_DIR"
mkdir -p artifacts

echo "[Course] Generating 5 randomized training maps into $OUT_CFG_DIR"
for i in 1 2 3 4 5; do
  python -m parking_project_submission.parking_env.generate_training_config \
    --out "$OUT_CFG_DIR/map_${i}.json" --seed $((100 + i))
done

CHECKPOINT=""
for i in 1 2 3 4 5; do
  CFG="$OUT_CFG_DIR/map_${i}.json"
  SAVE="artifacts/model_${i}.pt"
  echo "[Course] Training on $CFG -> $SAVE (resume from: ${CHECKPOINT:-none})"
  if [[ -n "${CHECKPOINT}" ]]; then
    python -m parking_project_submission.agent_learning \
      --config "$CFG" --checkpoint "$CHECKPOINT" \
      --total-steps 200000 --rollout-len 2048 --chunk-len 256 --epochs 4 \
      --save-path "$SAVE"
  else
    python -m parking_project_submission.agent_learning \
      --config "$CFG" \
      --total-steps 200000 --rollout-len 2048 --chunk-len 256 --epochs 4 \
      --save-path "$SAVE"
  fi
  CHECKPOINT="$SAVE"
done

echo "[Course] Finished. Final model: $CHECKPOINT"

