# -*- coding: utf-8 -*-
"""把训练好的 .pt 权重导出为 ONNX，便于在 Windows / C# / C++ 等环境部署。

用法：
    python scripts/export_onnx.py --model runs/detect/weights/best.pt

导出的 .onnx 文件在权重同目录下。
（注：.pt 文件本身也可直接拷到 Windows 上用 ultralytics 加载，无需转换。）
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="best.pt 路径")
    args = parser.parse_args()

    model = YOLO(args.model)
    path = model.export(format="onnx")
    print(f"已导出: {path}")


if __name__ == "__main__":
    main()
