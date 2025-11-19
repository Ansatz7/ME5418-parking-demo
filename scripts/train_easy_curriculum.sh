#!/bin/bash
set -e  # 如果任何一步出错，脚本立刻停止

# ---------------------------------------------------------
# 配置区域
# ---------------------------------------------------------
# 每一关训练多少步 (建议 10万 - 20万步)
STEPS_PER_STAGE=200000

# 定义文件路径
CONFIG_DIR="parking_project_submission/configs"
ARTIFACT_DIR="artifacts"

# 确保目录存在
mkdir -p $CONFIG_DIR
mkdir -p $ARTIFACT_DIR

echo "======================================================="
echo "🚗 开始全自动课程学习 (Curriculum Learning)"
echo "======================================================="

# ---------------------------------------------------------
# 第一步：生成 4 张不同的宝宝级地图
# ---------------------------------------------------------
echo ""
echo "[Step 1/5] 正在生成 4 张不同的 Easy 地图..."

# 不加 --seed 让它随机生成，保证每张图的出生点都不一样
python -m parking_project_submission.parking_env.generate_easy_config --out $CONFIG_DIR/easy_map_1.json
python -m parking_project_submission.parking_env.generate_easy_config --out $CONFIG_DIR/easy_map_2.json
python -m parking_project_submission.parking_env.generate_easy_config --out $CONFIG_DIR/easy_map_3.json
python -m parking_project_submission.parking_env.generate_easy_config --out $CONFIG_DIR/easy_map_4.json

echo "✅ 地图生成完毕！请查看 $CONFIG_DIR 下的 .png 预览图。"

# ---------------------------------------------------------
# 第二步：开始接力训练
# ---------------------------------------------------------

# --- 关卡 1 ---
TARGET_STEP=$((STEPS_PER_STAGE * 1))
echo ""
echo "[Step 2/5] 训练关卡 1 (目标: $TARGET_STEP 步)..."
python -m parking_project_submission.agent_learning \
    --config $CONFIG_DIR/easy_map_1.json \
    --total-steps $TARGET_STEP \
    --save-path $ARTIFACT_DIR/model_easy_stage1.pt

# --- 关卡 2 (加载关卡 1 的模型继续练) ---
TARGET_STEP=$((STEPS_PER_STAGE * 2))
echo ""
echo "[Step 3/5] 训练关卡 2 (目标: $TARGET_STEP 步)..."
python -m parking_project_submission.agent_learning \
    --config $CONFIG_DIR/easy_map_2.json \
    --checkpoint $ARTIFACT_DIR/model_easy_stage1.pt \
    --total-steps $TARGET_STEP \
    --save-path $ARTIFACT_DIR/model_easy_stage2.pt

# --- 关卡 3 (加载关卡 2 的模型继续练) ---
TARGET_STEP=$((STEPS_PER_STAGE * 3))
echo ""
echo "[Step 4/5] 训练关卡 3 (目标: $TARGET_STEP 步)..."
python -m parking_project_submission.agent_learning \
    --config $CONFIG_DIR/easy_map_3.json \
    --checkpoint $ARTIFACT_DIR/model_easy_stage2.pt \
    --total-steps $TARGET_STEP \
    --save-path $ARTIFACT_DIR/model_easy_stage3.pt

# --- 关卡 4 (最终关卡) ---
TARGET_STEP=$((STEPS_PER_STAGE * 4))
echo ""
echo "[Step 5/5] 训练关卡 4 (目标: $TARGET_STEP 步)..."
python -m parking_project_submission.agent_learning \
    --config $CONFIG_DIR/easy_map_4.json \
    --checkpoint $ARTIFACT_DIR/model_easy_stage3.pt \
    --total-steps $TARGET_STEP \
    --save-path $ARTIFACT_DIR/model_easy_final.pt

echo ""
echo "======================================================="
echo "🎉 恭喜！全部训练完成。"
echo "最终模型已保存至: $ARTIFACT_DIR/model_easy_final.pt"
echo "你可以运行以下命令来欣赏它的表演："
echo "parking-gym-demo --mode policy --checkpoint $ARTIFACT_DIR/model_easy_final.pt --config $CONFIG_DIR/easy_map_4.json"
echo "======================================================="