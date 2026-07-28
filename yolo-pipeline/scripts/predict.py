# -*- coding: utf-8 -*-
"""推理脚本：对图片/文件夹/视频运行训练好的模型，Mac/Win 通用。

用法：
    python scripts/predict.py --model runs/detect/weights/best.pt --source 某张图.jpg
    python scripts/predict.py --model runs/classify/weights/best.pt --source 图片文件夹/
    python scripts/predict.py --model runs/detect/weights/best.pt --source 视频.mp4
    python scripts/predict.py --model xxx.pt --source 0        # 摄像头实时识别，按 q 退出

结果（画好框的图片/视频）保存在 runs/predict*/ 下，终端同时打印识别结果。
首次用摄像头时 macOS 会弹窗询问是否允许"终端"访问摄像头，选允许。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import get_device

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="权重文件路径（best.pt 或 .onnx）")
    parser.add_argument("--source", required=True, help="图片 / 文件夹 / 视频路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.source.isdigit():  # 摄像头实时模式
        print("摄像头已开启，实时识别中，在画面窗口按 q 退出…")
        for r in model.predict(
            source=int(args.source),
            conf=args.conf,
            device=get_device(),
            show=True,
            stream=True,
        ):
            if r.probs is not None:
                print(f"\r当前: {r.names[r.probs.top1]} ({r.probs.top1conf:.0%})   ", end="")
        return

    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=get_device(),
        save=True,
        project=str(ROOT / "runs"),
        name="predict",
    )

    for r in results:
        if r.probs is not None:  # 分类模型
            top = r.probs.top1
            print(f"{Path(r.path).name}: {r.names[top]} ({r.probs.top1conf:.2%})")
        else:  # 检测模型
            found = [f"{r.names[int(b.cls)]}({float(b.conf):.2f})" for b in r.boxes]
            print(f"{Path(r.path).name}: {', '.join(found) if found else '未检出目标'}")

    print("\n带标注的结果已保存到 runs/predict*/")


if __name__ == "__main__":
    main()
