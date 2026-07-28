# -*- coding: utf-8 -*-
"""目标检测训练（示例一：办公室安全隐患检测）。

数据准备：
    1. 图片放入 datasets/detect/images/train 和 images/val
    2. 用 LabelImg / Roboflow / X-AnyLabeling 标注，导出 YOLO 格式 txt，
       放入 datasets/detect/labels/train 和 labels/val（文件名与图片一一对应）
    3. 修改 datasets/detect/dataset.yaml 中的类别名

运行：python scripts/train_detect.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import get_device, resolve_model, save_to_models_dir

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "datasets" / "detect" / "dataset.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型（首次运行自动下载）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = get_device()
    print(f"使用设备: {device}")

    model = YOLO(resolve_model(args.model))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project=str(ROOT / "runs"),
        name="detect",
    )
    dest = save_to_models_dir(model.trainer.best, "detect")
    print(f"\n训练完成。最优权重已存到: {dest}")


if __name__ == "__main__":
    main()
