# -*- coding: utf-8 -*-
"""图像分类训练（示例二：汽车/摩托车/电动车）。

数据准备：把图片按类别放入文件夹即可，无需画标注框：
    datasets/classify/train/car/xxx.jpg
    datasets/classify/train/motorcycle/xxx.jpg
    datasets/classify/train/ebike/xxx.jpg
    datasets/classify/val/...（同样结构，每类留 10~20% 图片做验证）

运行：python scripts/train_classify.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import get_device, resolve_model, save_to_models_dir

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "datasets" / "classify"), help="分类数据集根目录")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="预训练模型（首次运行自动下载）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
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
        name="classify",
    )
    dest = save_to_models_dir(model.trainer.best, "classify")
    print(f"\n训练完成。最优权重已存到: {dest}")


if __name__ == "__main__":
    main()
