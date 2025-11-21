#!/bin/bash
set -e  # 如果任何一步出错，脚本立刻停止

# ==============================================================================
# 1. 全局配置区域 (Configuration) - 在这里修改所有设置！
# ==============================================================================

# --- 🚩 起始点设置 (手动修改这里) ---
# 如果是从零开始训练，请把 START_CHECKPOINT 留空 ("")，并将 START_STEPS 设为 0
# 如果要接力旧模型，请填入路径和该模型已训练的步数
# START_CHECKPOINT="artifacts/model_final_1.pt"  # 例如: "artifacts/my_best.pt" 或 ""
# START_STEPS=50000                                # 例如: 500000 或 0

START_CHECKPOINT=""  # 从零开始训练
START_STEPS=0          # 从零开始训练

# --- 🗺️ 地图设置 (在此处填写你需要训练的地图路径) ---
# 填写几个训练几个，脚本会自动进行接力
MAP_LIST=(
    "parking_project_submission/configs/my_custom_medium6.json"
    # "parking_project_submission/configs/my_custom_medium3.json"
    # "parking_project_submission/configs/my_custom_medium4.json"
    # "parking_project_submission/configs/my_custom_medium5.json"
)

# --- 训练步数设置 ---
# 每一张地图 **额外** 训练多少步？
# 最终的总步数 = 起始步数 + (关卡数 * 单关步数)
STEPS_PER_STAGE=2000000

# --- PPO 超参数设置 (Hyperparameters) ---
LR=1e-4         # 学习率 (3e-4 ~ 1e-4)
ENT_COEF=0.01   # 熵系数 (0.01 ~ 0.001)
EPOCHS=4        # 每次更新的 Epochs
CHUNK_LEN=256   # LSTM 序列块长度

# --- 路径设置 ---
CONFIG_DIR="parking_project_submission/configs"
ARTIFACT_DIR="artifacts"
VIDEO_DIR="${ARTIFACT_DIR}/videos"  # 视频输出目录

# ------------------------------------------------------------------------------
# (以下逻辑无需修改)
# ------------------------------------------------------------------------------

# 确保目录存在
mkdir -p $CONFIG_DIR
mkdir -p $ARTIFACT_DIR
mkdir -p $VIDEO_DIR

# 计算总关卡数
NUM_STAGES=${#MAP_LIST[@]}

echo "======================================================="
echo "🚗 开始全自动课程学习 (Custom Curriculum)"
echo "   - 待训练地图数: $NUM_STAGES"
echo "   - 单关新增步数: $STEPS_PER_STAGE"
echo "   - 起始模型:     ${START_CHECKPOINT:-'无 (从零开始)'}"
echo "   - 起始步数:     $START_STEPS"
echo "   - 学习率:       $LR"
echo "   - 熵系数:       $ENT_COEF"
echo "======================================================="


# ==============================================================================
# 2. 接力训练阶段 (Curriculum Training)
# ==============================================================================
echo ""
echo "[Phase 1] 开始接力训练..."

# 初始化变量：上一关的模型路径
PREV_MODEL=""

# 遍历地图列表进行训练
for ((i=0; i<NUM_STAGES; i++))
do
    # 获取当前地图路径
    CURRENT_MAP="${MAP_LIST[$i]}"
    
    # 关卡编号 (从1开始)
    STAGE_NUM=$((i + 1))
    
    # --- 核心逻辑：计算目标步数 ---
    # 目标步数 = (手动设定的起始步数) + (当前关卡数 * 每关训练步数)
    TARGET_STEP=$((START_STEPS + STEPS_PER_STAGE * STAGE_NUM))
    
    # 定义当前关卡模型保存路径
    CURRENT_MODEL="$ARTIFACT_DIR/model_stage_${STAGE_NUM}.pt"
    
    echo ""
    echo ">>> [Stage $STAGE_NUM/$NUM_STAGES] 训练地图: $CURRENT_MAP"
    echo "    目标累积步数: $TARGET_STEP"
    
    # 构建基础训练命令
    CMD="python -m parking_project_submission.agent_learning \
        --config $CURRENT_MAP \
        --total-steps $TARGET_STEP \
        --save-path $CURRENT_MODEL \
        --lr $LR \
        --ent-coef $ENT_COEF \
        --epochs $EPOCHS \
        --chunk-len $CHUNK_LEN"

    # --- 接力逻辑 ---
    if [ $i -gt 0 ]; then
        # 情况 A: 循环内部接力 (加载上一关刚练好的模型)
        echo "    🔄 接力加载上一关模型: $PREV_MODEL"
        CMD="$CMD --checkpoint $PREV_MODEL"
    else
        # 情况 B: 循环的第一关
        if [ -n "$START_CHECKPOINT" ] && [ -f "$START_CHECKPOINT" ]; then
            # 如果用户填了起始模型，就加载它
            echo "    🚀 加载配置的起始模型: $START_CHECKPOINT"
            CMD="$CMD --checkpoint $START_CHECKPOINT"
        else
            # 如果没填，就从零开始
            echo "    🌱 从零开始训练 (无 Checkpoint)"
        fi
    fi

    # 执行命令
    eval $CMD
    
    # 更新变量，供下一次循环使用
    PREV_MODEL=$CURRENT_MODEL
done

# 复制最终模型
FINAL_MODEL="$ARTIFACT_DIR/model_final_6.pt"
cp "$PREV_MODEL" "$FINAL_MODEL"

echo ""
echo "🎉 全部训练完成！最终模型已保存至: $FINAL_MODEL"


# ==============================================================================
# 3. 自动可视化验收 (Visualization & Recording)
# ==============================================================================
echo ""
echo "[Phase 2] 正在进行可视化验收 (录制视频)..."

# 使用列表中最后一张地图进行测试
LAST_MAP_INDEX=$((NUM_STAGES - 1))
TEST_MAP="${MAP_LIST[$LAST_MAP_INDEX]}"
VIDEO_PATH="$VIDEO_DIR/demo_final_check_6.mp4"

echo "   - 测试模型: $FINAL_MODEL"
echo "   - 测试地图: $TEST_MAP"
echo "   - 视频输出: $VIDEO_PATH"

# 运行演示脚本 (录制视频)
parking-gym-demo \
    --mode policy \
    --checkpoint "$FINAL_MODEL" \
    --config "$TEST_MAP" \
    --record "$VIDEO_PATH" \
    # --no-visualize  # 如果需要静默运行可取消注释

echo ""
echo "✅ 验收完成！请查看录像文件: $VIDEO_PATH"
echo "======================================================="