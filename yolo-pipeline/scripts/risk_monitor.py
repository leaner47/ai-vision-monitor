# -*- coding: utf-8 -*-
"""VLM 安全隐患分析：用本地视觉大模型（Ollama + Qwen3-VL）推理画面中的风险因果。

示例：点燃的蜡烛 → 火灾风险；线缆乱放/插头未插好 → 漏电触电风险。

用法（需先安装并启动 Ollama，见《VLM风险分析指南.md》）：
    python scripts/risk_monitor.py --source 0              # 电脑摄像头，每 10 秒分析一帧
    python scripts/risk_monitor.py --source 1              # iPhone 连续互通相机（编号 0/1/2 试）
    python scripts/risk_monitor.py --source http://192.168.1.8:8080/video   # 手机 IP 摄像头 App
    python scripts/risk_monitor.py --source rtsp://用户:密码@IP/stream      # 监控摄像头 RTSP 流
    python scripts/risk_monitor.py --source 办公室.jpg      # 单张图片
    python scripts/risk_monitor.py --source 照片文件夹/     # 批量图片
    python scripts/risk_monitor.py --source 0 --interval 5 # 每 5 秒分析一次

发现风险的帧会连同分析结果存到 runs/risk/。按 Ctrl+C 退出。
"""
import argparse
import base64
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "runs" / "risk"

OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = (
    "你是安全巡检员。仔细观察这张画面，找出所有潜在安全隐患，并推断对应风险。"
    "例如：明火/点燃的蜡烛→火灾风险；电线线缆乱放→绊倒和漏电风险；"
    "插头未插紧/插座过载→触电和电气火灾风险；液体靠近电器→短路风险。"
    "按以下格式逐条输出：\n[风险] 隐患描述 → 风险类型 → 处理建议\n"
    "如果画面没有明显安全隐患，只回答：未发现明显安全隐患。用中文，简洁。"
)


def analyze(image_bgr, model: str) -> str:
    """把一帧图像发给本地 VLM，返回风险分析文本。"""
    ok, jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "（图像编码失败）"
    b64 = base64.b64encode(jpg.tobytes()).decode()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            # 限制上下文为 8K，避免按默认 256K 分配几十 GB 缓存挤爆内存
            "options": {"num_ctx": 8192},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def save_alert(image_bgr, text: str):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(SAVE_DIR / f"{ts}.jpg"), image_bgr)
    (SAVE_DIR / f"{ts}.txt").write_text(text, encoding="utf-8")
    print(f"  [已存档] runs/risk/{ts}.jpg")


def check_ollama(model: str):
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        if not any(model.split(":")[0] in n for n in names):
            print(f"警告：本地未找到模型 {model}，请先执行: ollama pull {model}")
    except Exception:
        raise SystemExit("连不上 Ollama（localhost:11434）。请先启动：ollama serve（或打开 Ollama 应用）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="0=摄像头 / 图片路径 / 图片文件夹")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Ollama 视觉模型名")
    parser.add_argument("--interval", type=float, default=10, help="摄像头模式下的分析间隔（秒）")
    args = parser.parse_args()

    check_ollama(args.model)

    is_stream = args.source.isdigit() or args.source.startswith(("http://", "https://", "rtsp://"))
    if is_stream:  # 摄像头 / 网络视频流模式
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit("视频源打开失败（本机摄像头检查权限或换编号；网络流检查地址和同一Wi-Fi）")
        print(f"摄像头已开启，每 {args.interval} 秒分析一帧。")
        print("预览窗口按 q 退出。注意：第一次分析需加载模型，可能要等 1~2 分钟，之后每帧 5~15 秒。\n")
        last_analyze = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("读帧失败，重试…")
                    time.sleep(1)
                    continue
                cv2.imshow("Risk Monitor (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if time.time() - last_analyze >= args.interval:
                    print(f"--- {datetime.now().strftime('%H:%M:%S')} 分析中…（画面会暂停几秒）")
                    text = analyze(frame, args.model)
                    print(text + "\n")
                    if "未发现" not in text:
                        save_alert(frame, text)
                    last_analyze = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n已退出")
    else:  # 图片/文件夹模式
        p = Path(args.source)
        files = sorted(p.glob("*")) if p.is_dir() else [p]
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not files:
            raise SystemExit(f"没有找到图片: {args.source}")
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"{f.name}: 读取失败（路径含中文时请重命名为英文）")
                continue
            print(f"=== {f.name} ===")
            text = analyze(img, args.model)
            print(text + "\n")
            if "未发现" not in text:
                save_alert(img, text)


if __name__ == "__main__":
    main()
