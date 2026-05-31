#!/bin/bash
# ETF策略演化运行脚本
# 用法: bash run_etf_evolution.sh [iterations]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 默认迭代次数
ITERATIONS=${1:-200}

# 检查环境变量
if [ -f .env ]; then
    source .env 2>/dev/null || true
fi

if [ -z "$TUSHARE_TOKEN" ]; then
    echo "错误: 缺少 TUSHARE_TOKEN 环境变量"
    echo "请在 .env 文件中设置: TUSHARE_TOKEN=你的token"
    exit 1
fi

if [ -z "$LLM_API_KEY" ]; then
    echo "错误: 缺少 LLM_API_KEY 环境变量"
    echo "请在 .env 文件中设置: LLM_API_KEY=你的key"
    exit 1
fi

echo "=========================================="
echo "  A股ETF策略自动演化"
echo "=========================================="
echo "迭代次数: $ITERATIONS"
echo "数据范围: 2019-2024 (6折CV)"
echo "目标: 年化25%+, 回撤<15%"
echo "=========================================="

# 检查是否有checkpoint可恢复
CHECKPOINT_DIR="./openevolve_etf_output/checkpoints"
LATEST_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/checkpoint_* 2>/dev/null | sort -V | tail -1)
fi

OPENEVOLVE_RUN="$(cd "$SCRIPT_DIR/../.." && pwd)/openevolve-run.py"

# 运行演化
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "从检查点恢复: $LATEST_CHECKPOINT"
    python "$OPENEVOLVE_RUN" \
        initial_strategy.py \
        etf_evaluator.py \
        --config etf_config.yaml \
        --checkpoint "$LATEST_CHECKPOINT" \
        --iterations "$ITERATIONS" \
        --output ./openevolve_etf_output
else
    echo "全新启动演化"
    python "$OPENEVOLVE_RUN" \
        initial_strategy.py \
        etf_evaluator.py \
        --config etf_config.yaml \
        --iterations "$ITERATIONS" \
        --output ./openevolve_etf_output
fi

echo ""
echo "演化完成！结果保存在 ./openevolve_etf_output/"
