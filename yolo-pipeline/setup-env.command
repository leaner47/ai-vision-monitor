#!/bin/bash
# AI 巡检 一键环境配置（macOS）——把整个文件夹拷到新电脑后运行一次即可
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "======== AI 巡检 · 一键环境配置 ========"

# 1/4 Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ 未找到 Python3。请先安装（任选其一）："
    echo "   - 终端执行: xcode-select --install"
    echo "   - 或官网下载: https://www.python.org/downloads/"
    read -p "按回车退出" _; exit 1
fi
echo "[1/4] Python: $(python3 --version)"

# 2/4 虚拟环境 + 依赖
if ! venv/bin/python -c "import sys" >/dev/null 2>&1; then
    echo "[2/4] 创建虚拟环境…"
    rm -rf venv
    python3 -m venv venv
else
    echo "[2/4] 虚拟环境已存在，检查依赖…"
fi
source venv/bin/activate
python -m pip install -q --upgrade pip
echo "      安装依赖（新电脑首次约 5~15 分钟，请耐心等待）…"
pip install -q -r requirements.txt || {
    echo "      默认源较慢/失败，改用清华镜像重试…"
    pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
} || { echo "❌ 依赖安装失败，请检查网络后重跑"; read -p "按回车退出" _; exit 1; }

# 3/4 依赖自检
echo "[3/4] 依赖自检…"
venv/bin/python - <<'EOF' || { echo "❌ 自检未通过"; read -p "按回车退出" _; exit 1; }
import cv2, flask, requests, ultralytics, torch
dev = "mps(Apple GPU)" if torch.backends.mps.is_available() else ("cuda(NVIDIA)" if torch.cuda.is_available() else "cpu")
print(f"      核心依赖全部就绪，推理设备: {dev}")
EOF

# 4/4 Ollama 与模型
export PATH="$(pwd)/bin:$PATH"
if ! command -v ollama >/dev/null 2>&1; then
    if [ -f "../installers/ollama-darwin.tgz" ]; then
        echo "[4/4] 从随包安装文件解压 Ollama（免安装版）…"
        mkdir -p bin
        tar -xzf ../installers/ollama-darwin.tgz -C bin
        chmod +x bin/ollama 2>/dev/null
        command -v ollama >/dev/null 2>&1 || { echo "❌ 解压失败"; read -p "按回车退出" _; exit 1; }
        echo "      已就位: yolo-pipeline/bin/ollama"
    elif command -v brew >/dev/null 2>&1; then
        echo "[4/4] 未检测到 Ollama，用 Homebrew 在线安装…"
        brew install ollama
    else
        echo "[4/4] ❌ 未检测到 Ollama，也没有随包安装文件（installers/ollama-darwin.tgz）。"
        echo "      请到 https://ollama.com/download 下载安装后重跑本配置"
        read -p "按回车退出" _; exit 1
    fi
else
    echo "[4/4] Ollama 已安装"
fi
if [ -d "../ollama/manifests" ]; then
    echo "      VLM 模型库已就位（随文件夹携带，无需下载）"
else
    echo "      ⚠️ 未找到同级 ollama/ 模型库。首次使用前需下载（约 6GB）："
    echo "      OLLAMA_MODELS=\"$(cd .. && pwd)/ollama\" ollama pull qwen3-vl:8b"
fi

echo ""
echo "✅ 环境配置完成！现在可以双击 🚨 AI巡检 图标启动系统。"
read -p "按回车关闭本窗口" _
