#!/bin/bash
# AI 巡检快速启动（由 AI巡检.app 调起，或直接双击本文件）
DIR="/Volumes/DB_004/AI视觉智能监测模型/yolo-pipeline"
cd "$DIR" || exit 1
export PATH="$DIR/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# 已在运行则只打开页面
if [ -f .dashboard.pid ] && kill -0 "$(cat .dashboard.pid)" 2>/dev/null; then
    open "http://localhost:8080"
    exit 0
fi

export AUTO_EXIT=1
./start.sh auto 8080

echo "等待服务就绪…"
for i in $(seq 1 60); do
    curl -s -o /dev/null "http://localhost:8080/state" && break
    sleep 1
done
open "http://localhost:8080"

echo ""
echo "监控运行中。关闭浏览器页面约 45 秒后会自动停止全部服务并释放内存。"
echo "（本窗口可最小化，勿手动关闭）"
while [ -f .dashboard.pid ] && kill -0 "$(cat .dashboard.pid)" 2>/dev/null; do
    sleep 3
done
./stop.sh
osascript -e 'display notification "监控系统已全部停止，内存已释放" with title "AI巡检"'
exit 0
