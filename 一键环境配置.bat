@echo off
chcp 65001 >nul
title AI巡检 一键环境配置
cd /d "%~dp0yolo-pipeline"
echo ======== AI 巡检 · 一键环境配置（Windows）========

rem [1/4] Python
where python >nul 2>&1
if errorlevel 1 (
    echo [1/4] 未找到 Python，尝试用 winget 自动安装…
    winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo ❌ 自动安装失败。请到 https://www.python.org/downloads/ 手动安装，
        echo    安装时务必勾选 "Add Python to PATH"，然后重跑本配置。
        pause & exit /b 1
    )
    echo 安装完成，请关闭本窗口后重新双击本配置（让 PATH 生效）。
    pause & exit /b 0
)
for /f "tokens=*" %%v in ('python --version') do echo [1/4] %%v

rem [2/4] 虚拟环境 + 依赖
venv\Scripts\python.exe -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [2/4] 创建虚拟环境…
    if exist venv rmdir /s /q venv
    python -m venv venv
) else (
    echo [2/4] 虚拟环境已存在，检查依赖…
)
call venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
echo       安装依赖（新电脑首次约 5~15 分钟，请耐心等待）…
pip install -q -r requirements.txt
if errorlevel 1 (
    echo       默认源较慢/失败，改用清华镜像重试…
    pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 ( echo ❌ 依赖安装失败，请检查网络后重跑 & pause & exit /b 1 )
)

rem [3/4] 依赖自检
echo [3/4] 依赖自检…
venv\Scripts\python.exe -c "import cv2, flask, requests, ultralytics, torch; print('      核心依赖全部就绪，推理设备:', 'cuda(NVIDIA)' if torch.cuda.is_available() else 'cpu')"
if errorlevel 1 ( echo ❌ 自检未通过 & pause & exit /b 1 )

rem [4/4] Ollama 与模型
set PATH=%~dp0yolo-pipeline\bin-win;%PATH%
where ollama >nul 2>&1
if errorlevel 1 (
    if exist "%~dp0installers\ollama-windows-amd64.zip" (
        echo [4/4] 从随包安装文件解压 Ollama（免安装版）…
        powershell -NoProfile -Command "Expand-Archive -Force '%~dp0installers\ollama-windows-amd64.zip' '%~dp0yolo-pipeline\bin-win'"
        where ollama >nul 2>&1 || ( echo ❌ 解压失败 & pause & exit /b 1 )
        echo       已就位: yolo-pipeline\bin-win\ollama.exe
    ) else (
        echo [4/4] 未检测到 Ollama，尝试用 winget 在线安装…
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
        if errorlevel 1 (
            echo ❌ 自动安装失败。请到 https://ollama.com/download 手动安装后重跑本配置。
            pause & exit /b 1
        )
    )
) else (
    echo [4/4] Ollama 已安装
)
if exist "..\ollama\manifests" (
    echo       VLM 模型库已就位（随文件夹携带，无需下载）
) else (
    echo       ⚠️ 未找到同级 ollama 模型库。首次使用前需下载（约 6GB）：
    echo       set OLLAMA_MODELS=%~dp0ollama ^&^& ollama pull qwen3-vl:8b
)

echo.
echo ✅ 环境配置完成！现在可以双击 yolo-pipeline\快速启动.bat 启动系统。
pause
