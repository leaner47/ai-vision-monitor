@echo off
chcp 65001 >nul
title AI巡检
rem 以本文件所在目录为项目目录，拷到任何盘符都能用（路径请勿含中文）
cd /d "%~dp0"
set AUTO_EXIT=1
set MODELS_DIR=%~dp0..\models
set OLLAMA_MODELS=%~dp0..\ollama
set PATH=%~dp0bin-win;%PATH%

rem [1/3] 启动 Ollama（未运行时）
curl -s -o nul http://localhost:11434/api/tags
if errorlevel 1 (
    echo [1/3] 启动 Ollama 服务…
    start "" /b ollama serve
) else (
    echo [1/3] Ollama 已在运行
)

rem [2/3] 启动 Dashboard
echo [2/3] 启动 Dashboard…
start "" /b venv\Scripts\python.exe scripts\dashboard.py --source auto --port 8080

rem [3/3] 等服务就绪后打开浏览器（最多等 60 秒）
echo [3/3] 等待服务就绪…
for /l %%i in (1,1,60) do (
    curl -s -o nul http://localhost:8080/train/status && goto ready
    timeout /t 1 /nobreak >nul
)
:ready
start "" http://localhost:8080

echo.
echo 监控运行中。关闭浏览器页面约 45 秒后自动停止全部服务。
echo 本窗口可最小化，请勿关闭。
:waitloop
timeout /t 3 /nobreak >nul
curl -s -o nul http://localhost:8080/train/status
if not errorlevel 1 goto waitloop

rem Dashboard 已自动退出，收尾释放内存
ollama stop qwen3-vl:8b >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1
taskkill /f /im "ollama app.exe" >nul 2>&1
echo 已全部停止，内存已释放。
timeout /t 5 >nul
exit
