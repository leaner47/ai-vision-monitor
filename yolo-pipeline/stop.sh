#!/bin/bash
# 一键停止：Dashboard + 卸载模型释放内存 + （若由 start.sh 启动的）Ollama 服务
cd "$(dirname "$0")"
VLM="qwen3-vl:8b"

# 1. 停 Dashboard
if [ -f .dashboard.pid ]; then
    kill "$(cat .dashboard.pid)" 2>/dev/null && echo "[1/3] Dashboard 已停止"
    rm -f .dashboard.pid
else
    pkill -f "scripts/dashboard.py" 2>/dev/null && echo "[1/3] Dashboard 已停止"
fi

# 2. 卸载模型，立即释放内存
ollama stop "$VLM" 2>/dev/null && echo "[2/3] 模型已卸载，内存已释放"

# 3. 停 Ollama 服务（仅当是 start.sh 启动的）
if [ -f .ollama.pid ]; then
    kill "$(cat .ollama.pid)" 2>/dev/null && echo "[3/3] Ollama 服务已停止"
    rm -f .ollama.pid
else
    echo "[3/3] Ollama 是手动/应用启动的，未动它（需要的话手动退出应用）"
fi

echo "完成"
