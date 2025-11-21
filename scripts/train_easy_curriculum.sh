#!/bin/bash
set -e  # 如果任何一步出错，脚本立刻停止

# ==============================================================================
# 1. 全局配置区域 (Configuration)
# ==============================================================================

# --- 课程设置 ---
# 生成几张地图来轮流训练？(建议至少 3-5 张以保证泛化性)
NUM_STAGES=1

# 每一关(每张地图)训练多少步？
# 步数越多，基础越牢；总步数 = NUM_STAGES * STEPS_PER_STAGE
STEPS_PER_STAGE=500000

# --- PPO 超参数设置 (Hyperparameters) ---
# 学习率 (默认 3e-4)。如果发现 loss 震荡严重可调小 (如 1e-4)
LR=1e-4

# 熵系数 (默认 0.01)。控制探索欲望。
# 如果 agent 太早“躺平”不愿动，调大它 (0.05); 如果 agent 乱动不收敛，调小它 (0.001)
ENT_COEF=0.01

# 每次更新的 Epochs (默认 4)。每批数据反复学习几次。
EPOCHS=4

# 批次大小 (默认 256)。LSTM 序列块的长度。
CHUNK_LEN=256

# --- 路径设置 ---
CONFIG_DIR="parking_project_submission/configs"
ARTIFACT_DIR="artifacts"
VIDEO_DIR="${ARTIFACT_DIR}/videos"  # 视频输出目录

# 确保目录存在
mkdir -p $CONFIG_DIR
mkdir -p $ARTIFACT_DIR
mkdir -p $VIDEO_DIR

echo "======================================================="
echo "🚗 开始全自动课程学习 (Curriculum Learning)"
echo "   - 关卡数量: $NUM_STAGES"
echo "   - 单关步数: $STEPS_PER_STAGE"
echo "   - 学习率:   $LR"
echo "   - 熵系数:   $ENT_COEF"
echo "======================================================="


# # ==============================================================================
# # 2. 地图生成阶段 (Map Generation)
# # ==============================================================================
# echo ""
# echo "[Phase 1] 正在生成 $NUM_STAGES 张不同的 Easy 地图..."

# for ((i=1; i<=NUM_STAGES; i++))
# do
#     # 不加 --seed 让它随机生成，保证每张图的出生点都不一样
#     # 自动生成文件名: easy_map_1.json, easy_map_2.json ...
#     python -m parking_project_submission.parking_env.generate_easy_config \
#         --out "$CONFIG_DIR/easy_map_${i}.json"
#     echo "  -> 生成完毕: easy_map_${i}.json (含预览图)"
#     # python -m parking_project_submission.parking_env.generate_easy_config \
#     #     --out "$CONFIG_DIR/easy_map_test_${i}.json"
#     # echo "  -> 生成完毕: easy_map_test_${i}.json (含预览图)"
# done


# ==============================================================================
# 3. 接力训练阶段 (Curriculum Training)
# ==============================================================================
echo ""
echo "[Phase 2] 开始接力训练..."

# 初始化变量
PREV_MODEL=""  # 上一关的模型路径

for ((i=1; i<=NUM_STAGES; i++))
do
    # 计算当前关卡的累积目标步数 (例如: 20万 -> 40万 -> 60万...)
    TARGET_STEP=$((STEPS_PER_STAGE * i))
    CURRENT_MAP="$CONFIG_DIR/easy_map_${i}.json"
    CURRENT_MODEL="$ARTIFACT_DIR/model_easy_stage${i}.pt"
    
    echo ""
    echo ">>> [Stage $i/$NUM_STAGES] 训练地图: easy_map_${i}.json"
    echo "    目标总步数: $TARGET_STEP"
    
    # 构建基础命令
    CMD="python -m parking_project_submission.agent_learning \
        --config $CURRENT_MAP \
        --total-steps $TARGET_STEP \
        --save-path $CURRENT_MODEL \
        --lr $LR \
        --ent-coef $ENT_COEF \
        --epochs $EPOCHS \
        --chunk-len $CHUNK_LEN"

    # 如果不是第一关，则加载上一关的模型作为 Checkpoint
    if [ $i -gt 1 ]; then
        echo "    加载 Checkpoint: $PREV_MODEL"
        CMD="$CMD --checkpoint $PREV_MODEL"
    else
        echo "    (第一关: 从零开始训练)"
    fi

    # 执行命令
    eval $CMD
    
    # 更新上一关模型变量，供下一次循环使用
    PREV_MODEL=$CURRENT_MODEL
done

# 将最后一个模型复制为 final
FINAL_MODEL="$ARTIFACT_DIR/model_easy_4map.pt"
cp "$PREV_MODEL" "$FINAL_MODEL"

echo ""
echo "🎉 全部训练完成！最终模型已保存至: $FINAL_MODEL"


# ==============================================================================
# 4. 自动可视化验收 (Visualization & Recording)
# ==============================================================================
echo ""
echo "[Phase 3] 正在进行可视化验收 (录制视频)..."

# 使用最后一张训练地图进行测试
TEST_MAP="$CONFIG_DIR/easy_map_${NUM_STAGES}.json"
VIDEO_PATH="$VIDEO_DIR/demo_easy_4map.mp4"

echo "   - 测试模型: $FINAL_MODEL"
echo "   - 测试地图: $TEST_MAP"
echo "   - 视频输出: $VIDEO_PATH"

# 运行演示脚本
# --no-visualize: 不弹窗 (适合在服务器/后台运行)
# --record: 录制视频
parking-gym-demo \
    --mode policy \
    --checkpoint "$FINAL_MODEL" \
    --config "$TEST_MAP" \
    --record "$VIDEO_PATH"\
    # --no-visualize 

echo ""
echo "✅ 验收完成！请查看录像文件: $VIDEO_PATH"
echo "======================================================="