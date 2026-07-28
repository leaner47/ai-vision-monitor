# -*- coding: utf-8 -*-
"""跨平台工具函数：设备自动选择 + 模型统一存放目录。"""
import os
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# 所有 AI 模型（预训练权重 + 训练产出）统一存放目录。
# 优先级：环境变量 MODELS_DIR > 项目所在文件夹旁的 models/（如 AI视觉智能监测模型/models）
_env = os.environ.get("MODELS_DIR")
MODELS_DIR = Path(_env) if _env else ROOT.parent / "models"


def get_device():
    """自动检测设备：NVIDIA GPU > Apple Silicon (MPS) > CPU。"""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model(name: str) -> str:
    """预训练权重统一放在 MODELS_DIR，本地没有时自动下载到该目录。"""
    if Path(name).exists():
        return str(name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / Path(name).name
    if not target.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(target))
    return str(target)


def save_to_models_dir(best_pt, task_name: str) -> Path:
    """把训练产出的 best.pt 复制到 MODELS_DIR，按任务命名。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{task_name}_best.pt"
    shutil.copy2(best_pt, dest)
    return dest


if __name__ == "__main__":
    print(f"当前将使用的设备: {get_device()}")
    print(f"模型存放目录: {MODELS_DIR}")
