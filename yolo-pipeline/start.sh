#!/bin/bash
# 一键启动：Ollama 服务 + 加载 VLM 模型 + 巡检 Dashboard
# 用法: ./start.sh [摄像头编号或流地址] [端口]   默认: ./start.sh 1 8080
set -e
cd "$(dirname "$0")"

SOURCE="${1:-auto}"
PORT="${2:-8080}"
VLM="qwen3-vl:8b"
# Ollama 模型放在项目所在文件夹旁的 ollama/ 目录；bin/ 里可能有随包携带的 ollama
export OLLAMA_MODELS="$(cd .. && pwd)/ollama"
export PATH="$(pwd)/bin:$PATH"

source venv/bin/activate

# 1. Ollama 没在运行就后台启动
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[1/2] 启动 Ollama 服务…"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    echo $! > .ollama.pid
    for i in $(seq 1 20); do
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break
        sleep 1
    done
else
    echo "[1/2] Ollama 已在运行"
fi

# 2. 启动 Dashboard（模型在网页点"开始"时才加载，点"结束"自动释放内存）
echo "[2/2] 启动 Dashboard…"
nohup python scripts/dashboard.py --source "$SOURCE" --port "$PORT" --vlm "$VLM" > /tmp/dashboard.log 2>&1 &
echo $! > .dashboard.pid
sleep 2

echo ""
echo "全部就绪 → 浏览器打开 http://localhost:$PORT"
echo "停止运行: ./stop.sh    查看日志: tail -f /tmp/dashboard.log"
