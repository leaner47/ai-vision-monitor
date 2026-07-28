# -*- coding: utf-8 -*-
"""安全巡检 Dashboard（网页版）。

布局：
  左侧  检查对象——YOLO 实时识别出的类别，动态更新，勾选决定画面框选和 VLM 分析重点
  中间  实时画面（勾选类别画框，无置信度文字）
  右侧  VLM 推理结果
  底部  开始 / 结束按钮——打开网页只有预览；点"开始"才加载 YOLO+VLM；点"结束"全部停止并释放内存

用法（Ollama 需已安装，模型已 pull）：
    python scripts/dashboard.py --source 1 --port 8080
然后浏览器打开 http://localhost:8080
"""
import argparse
import base64
import gc
import os
import random
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
from flask import Flask, Response, jsonify, request

from common import ROOT, get_device, resolve_model, save_to_models_dir

STAGING = ROOT / "datasets" / "custom_staging"   # 上传素材暂存
CUSTOM = ROOT / "datasets" / "custom"            # 自动划分后的训练集
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OLLAMA_URL = "http://localhost:11434"

# COCO 80 类中文名
COCO_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船",
    "traffic light": "红绿灯", "fire hydrant": "消防栓", "stop sign": "停车标志",
    "parking meter": "停车计时器", "bench": "长椅", "bird": "鸟", "cat": "猫", "dog": "狗",
    "horse": "马", "sheep": "羊", "cow": "牛", "elephant": "大象", "bear": "熊",
    "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "滑雪板", "snowboard": "单板滑雪板", "sports ball": "球", "kite": "风筝",
    "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板",
    "surfboard": "冲浪板", "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗",
    "banana": "香蕉", "apple": "苹果", "sandwich": "三明治", "orange": "橙子",
    "broccoli": "西兰花", "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨",
    "donut": "甜甜圈", "cake": "蛋糕", "chair": "椅子", "couch": "沙发",
    "potted plant": "盆栽", "bed": "床", "dining table": "桌子", "toilet": "马桶",
    "tv": "显示器", "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱",
    "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱", "book": "书",
    "clock": "钟表", "vase": "花瓶", "scissors": "剪刀", "teddy bear": "玩偶",
    "hair drier": "吹风机", "toothbrush": "牙刷",
}

app = Flask(__name__)
state = {
    "running": False,      # 是否已点"开始"
    "loading": False,      # 模型加载中
    "yolo": None,
    "cams": {},            # id -> {"name","src","raw","annotated","online","enabled"}
    "detected": {},        # {英文类名: {"count": n, "ts": 最后出现时间}}
    "selected": None,      # None=全选；否则为勾选的英文类名列表
    "results": [],
    "interval": 15,
    "analyzing": False,
    "vlm_model": "qwen3-vl:8b",
    "vlm_idx": 0,          # VLM 轮询分析到第几路摄像头
}
lock = threading.Lock()
DEVICE = "cpu"


def zh(name: str) -> str:
    return COCO_ZH.get(name, name)


def init_cams(sources: str):
    """解析/探测摄像头列表并为每路启动采集线程。sources 形如 "auto" 或 "0,1,rtsp://..."。"""
    srcs = []
    if sources.strip().lower() == "auto":
        print("自动探测本机摄像头…")
        for i in range(4):
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                srcs.append(str(i))
        if not srcs:
            srcs = ["0"]
    else:
        srcs = [s.strip() for s in sources.split(",") if s.strip()]

    n_url = 0
    for s in srcs:
        if s.isdigit():
            cam_id, name = f"cam{s}", f"摄像头 {s}"
        else:
            n_url += 1
            cam_id, name = f"url{n_url}", f"网络流 {n_url}"
        state["cams"][cam_id] = {"name": name, "src": s, "raw": None,
                                 "annotated": None, "online": False, "enabled": True}
        threading.Thread(target=capture_loop, args=(cam_id,), daemon=True).start()
    print(f"共 {len(srcs)} 路视频源: " + ", ".join(c['name'] for c in state['cams'].values()))


def capture_loop(cam_id):
    """单路采集：只负责抓帧，断线自动重连。"""
    cam = state["cams"][cam_id]
    src = int(cam["src"]) if str(cam["src"]).isdigit() else cam["src"]
    while True:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cam["online"] = False
            time.sleep(5)
            continue
        cam["online"] = True
        while True:
            ret, frame = cap.read()
            if not ret:
                cam["online"] = False
                break
            with lock:
                cam["raw"] = frame
        cap.release()
        time.sleep(3)


def detect_loop():
    """单一检测线程轮询所有启用摄像头，避免多线程争抢 GPU。"""
    while True:
        with lock:
            yolo = state["yolo"]
            running = state["running"]
            selected = state["selected"]
        if not (running and yolo is not None):
            time.sleep(0.2)
            continue
        counts_all = {}
        for cam in list(state["cams"].values()):
            if not cam["enabled"]:
                continue
            with lock:
                frame = None if cam["raw"] is None else cam["raw"].copy()
            if frame is None:
                continue
            r = yolo.predict(frame, conf=0.4, device=DEVICE, verbose=False)[0]
            annotated = frame.copy()
            for box in r.boxes:
                name = r.names[int(box.cls)]
                counts_all[name] = counts_all.get(name, 0) + 1
                if selected is None or name in selected:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 80), 2)
            with lock:
                cam["annotated"] = annotated
        now = time.time()
        with lock:
            for name, n in counts_all.items():
                state["detected"][name] = {"count": n, "ts": now}
            state["detected"] = {k: v for k, v in state["detected"].items() if now - v["ts"] < 5}
        time.sleep(0.02)


def build_prompt():
    with lock:
        selected = state["selected"]
        detected = dict(state["detected"])
    if selected is None:
        names = [zh(k) for k in detected]
    else:
        names = [zh(k) for k in selected if k in detected]
    focus = "、".join(names) if names else "画面中的所有物体"
    return (
        f"你是安全巡检员。画面中已识别出这些对象：{focus}。"
        "请围绕这些对象仔细检查安全隐患并推断风险"
        "（如明火→火灾风险；线缆乱放→绊倒漏电风险；插头未插紧→触电风险；液体靠近电器→短路风险）。"
        "按格式逐条输出：\n[风险] 隐患描述 → 风险类型 → 处理建议\n"
        "如无明显隐患，只回答：未发现明显安全隐患。用中文，简洁。"
    )


def vlm_loop():
    while True:
        time.sleep(1)
        with lock:
            if not state["running"]:
                continue
            # 轮流分析各启用摄像头
            cams = [(cid, c) for cid, c in state["cams"].items() if c["enabled"] and c["raw"] is not None]
            if not cams:
                continue
            state["vlm_idx"] = state["vlm_idx"] % len(cams)
            cam_name = cams[state["vlm_idx"]][1]["name"]
            frame = cams[state["vlm_idx"]][1]["raw"].copy()
            state["vlm_idx"] += 1
            interval = state["interval"]
            model = state["vlm_model"]
        state["analyzing"] = True
        try:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": build_prompt(),
                                  "images": [base64.b64encode(jpg.tobytes()).decode()]}],
                    "stream": False,
                    "options": {"num_ctx": 8192},
                    "keep_alive": "2h",
                },
                timeout=300,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
        except Exception as e:
            text = f"（分析失败：{e}）"
        with lock:
            state["results"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                        "text": f"【{cam_name}】\n{text}"})
            state["results"] = state["results"][:30]
        state["analyzing"] = False
        # 等待剩余间隔，期间若被"结束"则立即退出等待
        end = time.time() + interval
        while time.time() < end:
            with lock:
                if not state["running"]:
                    break
            time.sleep(0.5)


def do_start():
    if train_state["status"] in ("preparing", "training"):
        return  # 训练期间不允许启动监控
    with lock:
        if state["running"] or state["loading"]:
            return
        state["loading"] = True
    try:
        from ultralytics import YOLO
        yolo = YOLO(resolve_model("yolo11n.pt"))
        # 预热 VLM（异步加载进内存）
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": state["vlm_model"], "prompt": "",
                                "options": {"num_ctx": 8192}, "keep_alive": "2h"},
                          timeout=300)
        except Exception:
            pass
        with lock:
            state["yolo"] = yolo
            state["running"] = True
    finally:
        state["loading"] = False


def do_stop():
    with lock:
        state["running"] = False
        state["yolo"] = None
        state["detected"] = {}
        state["analyzing"] = False
    gc.collect()
    try:  # 卸载 VLM，立即释放内存
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": state["vlm_model"], "prompt": "", "keep_alive": 0},
                      timeout=30)
    except Exception:
        pass


# ---------------- 训练校准 ----------------
train_state = {"status": "idle", "epoch": 0, "total": 0, "acc": None, "error": None, "msg": ""}


def _safe_cls(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_").replace("..", "_")[:40]


@app.route("/train/reset", methods=["POST"])
def train_reset():
    if train_state["status"] in ("preparing", "training"):
        return jsonify({"ok": False, "error": "训练进行中"}), 400
    shutil.rmtree(STAGING, ignore_errors=True)
    train_state.update(status="idle", epoch=0, total=0, acc=None, error=None, msg="")
    return jsonify({"ok": True})


@app.route("/train/classes")
def train_classes():
    """返回已积累数据集中的类别及图片数，供前端展示为可续训的已有类别。"""
    out = {}
    for sub in ("train", "val"):
        d = CUSTOM / sub
        if d.exists():
            for c in sorted(d.iterdir()):
                if c.is_dir():
                    n = len([p for p in c.glob("*") if p.suffix.lower() in IMG_EXTS])
                    out[c.name] = out.get(c.name, 0) + n
    return jsonify(out)


@app.route("/train/upload", methods=["POST"])
def train_upload():
    cls = _safe_cls(request.form.get("cls", ""))
    if not cls:
        return jsonify({"ok": False, "error": "类别名为空"}), 400
    d = STAGING / cls
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in request.files.getlist("files"):
        ext = Path(f.filename or "").suffix.lower()
        if ext in IMG_EXTS:
            f.save(str(d / f"{time.time_ns()}_{n}{ext}"))
            n += 1
    return jsonify({"ok": True, "saved": n})


def _train_worker(classes, epochs):
    try:
        train_state.update(status="preparing", epoch=0, total=epochs, acc=None, error=None, msg="")
        do_stop()  # 暂停监控、卸载 VLM，把算力和内存让给训练
        # 新素材按 8:2 划分后合并进已有数据集（同名类别=继续补充）
        for cls in classes:
            imgs = [p for p in (STAGING / cls).glob("*") if p.suffix.lower() in IMG_EXTS]
            if len(imgs) < 2:
                raise RuntimeError(f"新增类别「{cls}」至少需要 2 张图片（建议 30 张以上）")
            random.shuffle(imgs)
            n_val = max(1, round(len(imgs) * 0.2))
            if n_val >= len(imgs):
                n_val = len(imgs) - 1
            for sub, part in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
                d = CUSTOM / sub / cls
                d.mkdir(parents=True, exist_ok=True)
                for p in part:
                    shutil.copy2(p, d / p.name)
        shutil.rmtree(STAGING, ignore_errors=True)

        all_classes = [c.name for c in (CUSTOM / "train").iterdir() if c.is_dir()]
        if len(all_classes) < 2:
            raise RuntimeError(
                f"当前数据集只有 1 个类别「{all_classes[0]}」。分类模型需要至少 2 个类别才有可学习的区分对象，"
                "请再添加一个类别（例如加一批“其他/背景”图片）后重试")

        train_state["status"] = "training"
        from ultralytics import YOLO
        model = YOLO(resolve_model("yolo11n-cls.pt"))

        def on_epoch(trainer):
            train_state["epoch"] = trainer.epoch + 1
            met = getattr(trainer, "metrics", None) or {}
            acc = met.get("metrics/accuracy_top1")
            if acc:
                train_state["acc"] = round(float(acc) * 100, 1)

        model.add_callback("on_fit_epoch_end", on_epoch)
        model.train(data=str(CUSTOM), epochs=epochs, imgsz=224, device=get_device(),
                    project=str(ROOT / "runs"), name="custom", exist_ok=True, verbose=False)
        dest = save_to_models_dir(model.trainer.best, "custom")
        train_state.update(status="done", msg=str(dest))
    except Exception as e:
        train_state.update(status="error", error=str(e))
    finally:
        gc.collect()


@app.route("/train/start", methods=["POST"])
def train_start():
    if train_state["status"] in ("preparing", "training"):
        return jsonify({"ok": False, "error": "已有训练在进行"}), 400
    data = request.get_json(force=True)
    classes = [_safe_cls(c) for c in data.get("classes", []) if _safe_cls(c)]
    epochs = max(5, min(300, int(data.get("epochs", 50))))
    if not classes:
        return jsonify({"ok": False, "error": "没有新增素材"}), 400
    threading.Thread(target=_train_worker, args=(classes, epochs), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/train/status")
def train_status():
    return jsonify(train_state)


@app.route("/control", methods=["POST"])
def control():
    action = request.get_json(force=True).get("action")
    if action == "start":
        threading.Thread(target=do_start, daemon=True).start()
    elif action == "stop":
        threading.Thread(target=do_stop, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/config", methods=["POST"])
def config():
    data = request.get_json(force=True)
    with lock:
        if "selected" in data:
            state["selected"] = data["selected"]
        if "interval" in data:
            state["interval"] = max(5, int(data["interval"]))
    return jsonify({"ok": True})


@app.route("/state")
def get_state():
    state["last_seen"] = time.time()   # 浏览器心跳
    with lock:
        detected = [{"key": k, "name": zh(k), "count": v["count"]}
                    for k, v in sorted(state["detected"].items())]
        cams = [{"id": cid, "name": c["name"], "online": c["online"], "enabled": c["enabled"]}
                for cid, c in state["cams"].items()]
        return jsonify({
            "running": state["running"], "loading": state["loading"],
            "analyzing": state["analyzing"], "detected": detected,
            "results": state["results"], "cams": cams,
        })


def mjpeg():
    while True:
        with lock:
            first = next((c for c in state["cams"].values() if c["enabled"]), None)
            frame = None if first is None else (first["annotated"] if state["running"] and first["annotated"] is not None else first["raw"])
        if frame is not None:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.04)


@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame")
def frame():
    """单帧 JPEG，按 ?cam=id 取指定摄像头画面。"""
    cam = state["cams"].get(request.args.get("cam", ""))
    if cam is None:
        return Response(status=404)
    with lock:
        f = cam["annotated"] if (state["running"] and cam["enabled"] and cam["annotated"] is not None) else cam["raw"]
        f = None if f is None else f.copy()
    if f is None:
        return Response(status=204)
    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpg.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/cams/toggle", methods=["POST"])
def cams_toggle():
    data = request.get_json(force=True)
    cam = state["cams"].get(data.get("id", ""))
    if cam:
        cam["enabled"] = bool(data.get("enabled", True))
    return jsonify({"ok": True})


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>安全巡检 Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #14171c; color: #dfe4ea; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 24px; background: #1c2027; border-bottom: 1px solid #2a2f38;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 17px; font-weight: 600; }
  #status { font-size: 12px; padding: 3px 10px; border-radius: 10px; background: #2a2f38; }
  #status.busy { background: #b7791f; color: #fff; }
  #status.on { background: #2f855a; color: #fff; }
  main { flex: 1; display: flex; gap: 14px; padding: 14px; min-height: 0; }
  .panel { background: #1c2027; border: 1px solid #2a2f38; border-radius: 10px;
           padding: 16px; overflow-y: auto; }
  #left { width: 220px; }
  h2 { font-size: 13px; color: #8b95a5; margin-bottom: 12px; letter-spacing: 1px; }
  #left label { display: flex; gap: 8px; align-items: center; font-size: 14px;
                padding: 7px 6px; border-radius: 6px; cursor: pointer; }
  #left label:hover { background: #242a33; }
  #left input { accent-color: #4c8dff; }
  #left .cnt { margin-left: auto; font-size: 12px; color: #8b95a5; }
  #left .empty { font-size: 13px; color: #566072; padding: 8px 6px; }
  #center { flex: 1; background: #000; border-radius: 10px; border: 1px solid #2a2f38;
            min-width: 0; padding: 6px; display: flex; }
  #grid { flex: 1; display: grid; gap: 6px; align-items: stretch; }
  .tile { position: relative; background: #0a0c0f; border-radius: 6px; overflow: hidden;
          display: flex; align-items: center; justify-content: center; min-height: 0; }
  .tile img { max-width: 100%; max-height: 100%; }
  .tile .tag { position: absolute; top: 8px; left: 10px; font-size: 12px; color: #fff;
               background: rgba(0,0,0,.55); padding: 2px 8px; border-radius: 4px; }
  .tile .off { color: #8b95a5; font-size: 13px; }
  #camList label { display: flex; gap: 8px; align-items: center; font-size: 14px;
                   padding: 7px 6px; border-radius: 6px; cursor: pointer; }
  #camList label:hover { background: #242a33; }
  #camList input { accent-color: #4c8dff; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot.on { background: #48bb78; }
  .dot.offline { background: #e05d5d; }
  #right { width: 340px; }
  .item { border-left: 3px solid #4c8dff; background: #242a33; border-radius: 6px;
          padding: 10px 12px; margin-bottom: 10px; font-size: 13px; line-height: 1.55; }
  .item.risk { border-left-color: #e05d5d; }
  .item .t { color: #8b95a5; font-size: 11px; margin-bottom: 4px; }
  .item pre { white-space: pre-wrap; font-family: inherit; }
  footer { padding: 12px 24px; background: #1c2027; border-top: 1px solid #2a2f38;
           display: flex; align-items: center; justify-content: center; gap: 16px; }
  button { font-size: 15px; padding: 9px 42px; border-radius: 8px; border: none;
           cursor: pointer; font-weight: 600; }
  #btnStart { background: #2f855a; color: #fff; }
  #btnStop  { background: #c53030; color: #fff; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  #btnTrain { background: #3b5bcc; color: #fff; }
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,.65); display: none;
             align-items: center; justify-content: center; z-index: 20; }
  .overlay.show { display: flex; }
  .modal { background: #1c2027; border: 1px solid #2a2f38; border-radius: 12px;
           width: 580px; max-height: 82vh; overflow-y: auto; padding: 22px; }
  .modal h3 { font-size: 16px; margin-bottom: 8px; }
  .hint { font-size: 12px; color: #8b95a5; line-height: 1.5; }
  .addrow { display: flex; gap: 10px; align-items: center; margin: 14px 0; }
  .addrow input[type=text], .addrow input[type=number] {
    background: #14171c; color: #dfe4ea; border: 1px solid #2a2f38;
    border-radius: 6px; padding: 8px 10px; flex: 1; font-size: 14px; }
  .addrow input[type=number] { flex: none; }
  .addrow button { padding: 8px 18px; font-size: 13px; background: #4c8dff; color: #fff; }
  .addrow button.ghost { background: #2a2f38; color: #dfe4ea; }
  .clsrow { margin-bottom: 12px; }
  .clsrow .name { font-size: 14px; margin-bottom: 4px; display: flex; justify-content: space-between; }
  .clsrow .name .del { color: #e05d5d; cursor: pointer; font-size: 12px; }
  .dropzone { border: 2px dashed #3a4150; border-radius: 8px; padding: 16px;
              text-align: center; color: #8b95a5; font-size: 12px; cursor: pointer; }
  .dropzone.drag { border-color: #4c8dff; color: #dfe4ea; background: #20263199; }
  .bar { height: 10px; background: #2a2f38; border-radius: 5px; overflow: hidden; margin-top: 14px; }
  .bar div { height: 100%; width: 0%; background: #4c8dff; transition: width .5s; }
</style>
</head>
<body>
<header>
  <h1>安全巡检 Dashboard</h1>
  <span id="status">预览中（未启动分析）</span>
</header>
<main>
  <div class="panel" id="left">
    <h2>摄像头</h2>
    <div id="camList"></div>
    <h2 style="margin-top:18px">检查对象（实时识别）</h2>
    <div id="targets"><div class="empty">点击"开始"后自动列出画面中识别到的对象</div></div>
  </div>
  <div id="center"><div id="grid"></div></div>
  <div class="panel" id="right">
    <h2>模型推理结果</h2>
    <div id="results"></div>
  </div>
</main>
<footer>
  <button id="btnStart">开 始</button>
  <button id="btnStop" disabled>结 束</button>
  <button id="btnTrain">训练校准</button>
</footer>

<div class="overlay" id="trainOverlay">
  <div class="modal">
    <h3>训练校准</h3>
    <p class="hint">输入识别类别并把训练图片拖入对应区域，系统自动按 8:2 划分训练/验证集并训练 YOLO 分类模型。素材会累积保留：向已有类别拖入新图片即为继续补充训练；也可以只新增 1 个类别（与已有类别合并训练）。每类建议 30 张以上。</p>
    <div class="addrow">
      <input type="text" id="clsInput" placeholder="输入类别名，如：电动车">
      <button id="btnAddCls">添加类别</button>
    </div>
    <div id="clsList"></div>
    <div class="addrow">
      <span class="hint">训练轮数</span>
      <input type="number" id="epochs" value="50" min="5" max="300" style="width:70px">
      <button id="btnStartTrain" disabled>开始训练</button>
      <button id="btnCloseTrain" class="ghost">关闭</button>
    </div>
    <div id="trainProgress" style="display:none">
      <div class="bar"><div id="barFill"></div></div>
      <div id="trainMsg" class="hint" style="margin-top:8px"></div>
    </div>
  </div>
</div>
<script>
const checked = new Map();   // key -> bool，记住用户勾选
let running = false;

// ---------------- 多摄像头：左侧选择区 + 网格分屏 ----------------
let cams = [];               // [{id,name,online,enabled}]
const tiles = new Map();     // camId -> {img, lastUrl}
const camDisabled = new Map();  // 本地记忆勾选状态

function renderCams() {
  const list = document.getElementById('camList');
  list.innerHTML = cams.map(c =>
    `<label><input type="checkbox" data-id="${c.id}" ${c.enabled ? 'checked' : ''}>
     <span class="dot ${c.online ? 'on' : 'offline'}"></span>${c.name}</label>`).join('') ||
    '<div class="empty">未发现摄像头</div>';
  list.querySelectorAll('input').forEach(i => i.onchange = () => {
    fetch('/cams/toggle', {method: 'POST', headers: {'Content-Type': 'application/json'},
                           body: JSON.stringify({id: i.dataset.id, enabled: i.checked})});
  });
  syncGrid();
}

function syncGrid() {
  const grid = document.getElementById('grid');
  const active = cams.filter(c => c.enabled);
  const cols = active.length <= 1 ? 1 : 2;
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  // 移除已停用的
  for (const [id, t] of tiles) {
    if (!active.find(c => c.id === id)) {
      t.el.remove(); if (t.lastUrl) URL.revokeObjectURL(t.lastUrl);
      tiles.delete(id);
    }
  }
  // 添加新启用的
  for (const c of active) {
    if (!tiles.has(c.id)) {
      const el = document.createElement('div');
      el.className = 'tile';
      el.innerHTML = `<img alt=""><span class="tag">${c.name}</span>`;
      grid.appendChild(el);
      tiles.set(c.id, {el, img: el.querySelector('img'), lastUrl: null});
    }
  }
}

async function refreshFrames() {
  for (const [id, t] of tiles) {
    try {
      const r = await fetch('/frame?cam=' + encodeURIComponent(id));
      if (r.status === 200) {
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        t.img.src = url;
        if (t.lastUrl) URL.revokeObjectURL(t.lastUrl);
        t.lastUrl = url;
      }
    } catch (e) {}
  }
  setTimeout(refreshFrames, tiles.size > 1 ? 60 : 80);
}
refreshFrames();

function sendControl(action) {
  fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'},
                     body: JSON.stringify({action})});
}
document.getElementById('btnStart').onclick = () => sendControl('start');
document.getElementById('btnStop').onclick  = () => { sendControl('stop'); };

function pushSelection() {
  const selected = [...checked.entries()].filter(([,v]) => v).map(([k]) => k);
  fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({selected})});
}

function renderTargets(detected) {
  const div = document.getElementById('targets');
  if (!detected.length) {
    div.innerHTML = `<div class="empty">${running ? '暂未识别到对象' : '点击"开始"后自动列出画面中识别到的对象'}</div>`;
    return;
  }
  let changed = false;
  for (const d of detected) if (!checked.has(d.key)) { checked.set(d.key, true); changed = true; }
  div.innerHTML = detected.map(d =>
    `<label><input type="checkbox" data-key="${d.key}" ${checked.get(d.key) ? 'checked' : ''}>
     ${d.name}<span class="cnt">×${d.count}</span></label>`).join('');
  div.querySelectorAll('input').forEach(i => i.onchange = () => {
    checked.set(i.dataset.key, i.checked); pushSelection();
  });
  if (changed) pushSelection();
}

async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    running = s.running;
    const st = document.getElementById('status');
    if (s.loading)        { st.textContent = '模型加载中…'; st.className = 'busy'; }
    else if (s.analyzing) { st.textContent = '分析中…';     st.className = 'busy'; }
    else if (s.running)   { st.textContent = '监控中';       st.className = 'on'; }
    else                  { st.textContent = '预览中（未启动分析）'; st.className = ''; }
    document.getElementById('btnStart').disabled = s.running || s.loading;
    document.getElementById('btnStop').disabled  = !s.running;
    if (JSON.stringify(s.cams) !== JSON.stringify(cams)) { cams = s.cams; renderCams(); }
    renderTargets(s.detected);
    document.getElementById('results').innerHTML = s.results.map(r =>
      `<div class="item ${r.text.includes('[风险]') ? 'risk' : ''}">
         <div class="t">${r.time}</div><pre>${r.text}</pre></div>`).join('');
  } catch (e) {}
}
setInterval(poll, 1500); poll();

// ---------------- 训练校准 ----------------
const overlay = document.getElementById('trainOverlay');
const clsList = document.getElementById('clsList');
const staged = new Map();   // 类别 -> File[]
let trainPolling = null;

let existing = {};   // 已积累数据集：类别 -> 图片数
document.getElementById('btnTrain').onclick = async () => {
  await fetch('/train/reset', {method: 'POST'});
  existing = await (await fetch('/train/classes')).json();
  staged.clear();
  for (const name of Object.keys(existing)) staged.set(name, []);
  renderCls();
  document.getElementById('trainProgress').style.display = 'none';
  document.getElementById('barFill').style.width = '0%';
  overlay.classList.add('show');
};
document.getElementById('btnCloseTrain').onclick = () => {
  overlay.classList.remove('show');
  if (trainPolling) { clearInterval(trainPolling); trainPolling = null; }
};

document.getElementById('btnAddCls').onclick = () => {
  const name = document.getElementById('clsInput').value.trim();
  if (!name || staged.has(name)) return;
  staged.set(name, []);
  document.getElementById('clsInput').value = '';
  renderCls();
};

function renderCls() {
  clsList.innerHTML = '';
  for (const [name, files] of staged) {
    const old = existing[name] || 0;
    const label = old
      ? `${name}（已有 ${old} 张${files.length ? '，新增 ' + files.length + ' 张' : ''}）`
      : `${name}（新类别，${files.length} 张）`;
    const row = document.createElement('div');
    row.className = 'clsrow';
    row.innerHTML = `<div class="name"><span>${label}</span>
                     ${old ? '' : `<span class="del" data-n="${name}">删除</span>`}</div>
                     <div class="dropzone" data-n="${name}">拖入图片，或点击选择文件</div>
                     <input type="file" multiple accept="image/*" style="display:none" data-n="${name}">`;
    clsList.appendChild(row);
  }
  clsList.querySelectorAll('.del').forEach(el => el.onclick = () => { staged.delete(el.dataset.n); renderCls(); });
  clsList.querySelectorAll('.dropzone').forEach(dz => {
    const input = clsList.querySelector(`input[data-n="${dz.dataset.n}"]`);
    dz.onclick = () => input.click();
    input.onchange = () => addFiles(dz.dataset.n, input.files);
    dz.ondragover = e => { e.preventDefault(); dz.classList.add('drag'); };
    dz.ondragleave = () => dz.classList.remove('drag');
    dz.ondrop = e => { e.preventDefault(); dz.classList.remove('drag'); addFiles(dz.dataset.n, e.dataTransfer.files); };
  });
  // 可开始训练的条件：至少一个类别有新增素材，且每个有新增素材的类别 ≥2 张
  const withNew = [...staged.values()].filter(f => f.length > 0);
  document.getElementById('btnStartTrain').disabled =
    withNew.length === 0 || withNew.some(f => f.length < 2);
}

function addFiles(name, fileList) {
  const arr = staged.get(name);
  for (const f of fileList) if (f.type.startsWith('image/')) arr.push(f);
  renderCls();
}

document.getElementById('btnStartTrain').onclick = async () => {
  const btn = document.getElementById('btnStartTrain');
  btn.disabled = true;
  const prog = document.getElementById('trainProgress');
  const msg = document.getElementById('trainMsg');
  prog.style.display = 'block';
  msg.textContent = '上传素材中…';
  const newClasses = [...staged.entries()].filter(([, f]) => f.length > 0);
  for (const [name, files] of newClasses) {
    const fd = new FormData();
    fd.append('cls', name);
    for (const f of files) fd.append('files', f);
    await fetch('/train/upload', {method: 'POST', body: fd});
  }
  const epochs = parseInt(document.getElementById('epochs').value) || 50;
  const r = await (await fetch('/train/start', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({classes: newClasses.map(([n]) => n), epochs})})).json();
  if (!r.ok) { msg.textContent = '启动失败：' + r.error; btn.disabled = false; return; }
  msg.textContent = '准备数据集…（监控已暂停，训练完成后可重新"开始"）';
  trainPolling = setInterval(async () => {
    const s = await (await fetch('/train/status')).json();
    if (s.status === 'training') {
      document.getElementById('barFill').style.width = (s.epoch / s.total * 100) + '%';
      msg.textContent = `训练中 ${s.epoch}/${s.total} 轮` + (s.acc ? `，当前验证准确率 ${s.acc}%` : '');
    } else if (s.status === 'done') {
      clearInterval(trainPolling); trainPolling = null;
      document.getElementById('barFill').style.width = '100%';
      msg.textContent = `训练完成！验证准确率 ${s.acc ?? '--'}%，模型已保存：${s.msg}`;
      btn.disabled = false;
    } else if (s.status === 'error') {
      clearInterval(trainPolling); trainPolling = null;
      msg.textContent = '训练失败：' + s.error;
      btn.disabled = false;
    }
  }, 2000);
};
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "--sources", dest="sources", default="auto",
                        help='"auto"=自动探测本机摄像头；或逗号分隔列表，如 "0,1,rtsp://…"')
    parser.add_argument("--vlm", default="qwen3-vl:8b")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    state["vlm_model"] = args.vlm
    DEVICE = get_device()
    init_cams(args.sources)
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=vlm_loop, daemon=True).start()

    if os.environ.get("AUTO_EXIT") == "1":
        def watchdog():
            """浏览器页面关闭（心跳消失 45 秒）后自动停止并退出，训练期间不退出。"""
            while True:
                time.sleep(5)
                ls = state.get("last_seen")
                if ls and time.time() - ls > 45 and train_state["status"] not in ("preparing", "training"):
                    print("浏览器已关闭，自动停止…")
                    do_stop()
                    os._exit(0)
        threading.Thread(target=watchdog, daemon=True).start()
    print(f"\nDashboard 已启动：http://localhost:{args.port}（预览模式，点网页里的\"开始\"才加载模型）\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
