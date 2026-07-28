# -*- coding: utf-8 -*-
"""下载开源车辆分类数据集并自动整理到 datasets/classify。

数据源：HuggingFace aryadytm/vehicle-classification-512x512（512x512 高质量图，16 类车辆）
本脚本自动完成：下载 → 解压 → 抽取 3 类（car / motorcycle / bicycle）→
按 每类 35 张训练 + 8 张验证 整理成 YOLO 分类目录结构（共约 130 张）。

用法（在 yolo-pipeline 目录、激活 venv 后）：
    python scripts/download_dataset.py

注：数据集中没有"电动车"类，暂用 bicycle（自行车）作第三类跑通流程，
    之后可把 bicycle 文件夹里的图换成自己拍的电动车照片重新训练。
"""
import random
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = ROOT / "downloads"
CLASSIFY_DIR = ROOT / "datasets" / "classify"

# huggingface.co 连不上时自动换国内镜像 hf-mirror.com
HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
REPO_PATH = "/datasets/aryadytm/vehicle-classification-512x512/resolve/main/data/"
FILES = {"train.zip": 116150320, "validation.zip": 15465708}

CLASSES = {"Car": "car", "Motorcycle": "motorcycle", "Bicycle": "bicycle"}
N_TRAIN, N_VAL = 35, 8
random.seed(42)


def download(filename: str, expected_size: int) -> Path:
    dest = DOWNLOAD_DIR / filename
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[跳过] {filename} 已存在")
        return dest
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for host in HOSTS:
        url = host + REPO_PATH + filename
        try:
            print(f"[下载] {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                done = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r  {done / 1e6:.1f} / {expected_size / 1e6:.1f} MB", end="")
            print()
            return dest
        except Exception as e:
            last_err = e
            print(f"\n  该源失败（{e}），尝试下一个…")
    raise RuntimeError(f"两个源都下载失败: {last_err}")


def collect_images(extract_dir: Path, class_name: str) -> list:
    """在解压目录中找到指定类别文件夹下的所有图片。"""
    imgs = []
    for d in extract_dir.rglob("*"):
        if d.is_dir() and d.name.lower() == class_name.lower():
            imgs += [p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return imgs


def main():
    tmp = ROOT / "downloads" / "_extract"

    splits = {}  # {"train": 解压目录, "validation": 解压目录}
    for filename, size in FILES.items():
        zip_path = download(filename, size)
        out = tmp / filename.replace(".zip", "")
        if not out.exists():
            print(f"[解压] {filename}")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out)
        splits[filename.replace(".zip", "")] = out

    print("\n[整理] 抽取类别并复制：")
    total = 0
    for src_name, dst_name in CLASSES.items():
        train_imgs = collect_images(splits["train"], src_name)
        val_imgs = collect_images(splits["validation"], src_name)
        if not train_imgs:
            print(f"  警告：训练集中未找到 {src_name}，可用文件夹：")
            for d in sorted({p.name for p in splits['train'].rglob('*') if p.is_dir()}):
                print("   -", d)
            continue
        random.shuffle(train_imgs)
        random.shuffle(val_imgs)
        picked = {"train": train_imgs[:N_TRAIN], "val": val_imgs[:N_VAL]}
        for split, imgs in picked.items():
            dst_dir = CLASSIFY_DIR / split / dst_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(imgs):
                shutil.copy2(img, dst_dir / f"{dst_name}_{split}_{i:03d}{img.suffix.lower()}")
            print(f"  {dst_name:12s} {split}: {len(imgs)} 张")
            total += len(imgs)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n完成！共 {total} 张图片已就位 -> {CLASSIFY_DIR}")
    print("下一步运行: python scripts/train_classify.py")


if __name__ == "__main__":
    main()
