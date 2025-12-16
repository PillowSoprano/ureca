#!/bin/bash
# 检查训练进程状态并处理

echo "=== 检查训练进程 ==="
TRAIN_PID=$(pgrep -f "python train.py kovae")

if [ -z "$TRAIN_PID" ]; then
    echo "✓ 训练进程已退出"
    echo ""
    echo "可以直接测试模型了！"
    exit 0
fi

echo "进程 ID: $TRAIN_PID"
echo ""

# 检查 CPU 使用率
CPU_USAGE=$(ps -p $TRAIN_PID -o %cpu --no-headers | xargs)
echo "CPU 使用率: ${CPU_USAGE}%"

# 检查运行时间
ELAPSED=$(ps -p $TRAIN_PID -o etime --no-headers | xargs)
echo "运行时间: $ELAPSED"

echo ""

# 判断是否卡住
if (( $(echo "$CPU_USAGE < 5.0" | bc -l) )); then
    echo "⚠️  CPU 使用率很低 (${CPU_USAGE}%)，可能卡住了"
    echo ""
    echo "建议操作："
    echo "1. 模型已经保存（因为显示了 'store!!!'）"
    echo "2. 可以安全地终止进程"
    echo "3. 直接测试模型"
    echo ""
    echo "要终止进程吗？(y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "终止进程 $TRAIN_PID ..."
        kill $TRAIN_PID
        sleep 2
        if ps -p $TRAIN_PID > /dev/null 2>&1; then
            echo "进程还在，强制终止..."
            kill -9 $TRAIN_PID
        fi
        echo "✓ 进程已终止"
        echo ""
        echo "现在可以测试模型了："
        echo "  python test_kovae_prediction.py"
    fi
else
    echo "✓ 进程正常运行 (CPU: ${CPU_USAGE}%)"
    echo "可能正在生成可视化，再等 1-2 分钟"
fi
