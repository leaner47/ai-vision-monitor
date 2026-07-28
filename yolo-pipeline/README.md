# 私有化图片识别模型训练工具（YOLO 跨平台版）

约 100 张图片 → 简单流程训练 → 输出可用的识别模型。Mac 上训练，Windows 上直接使用，同一套代码。

## 一、环境安装（Mac / Windows 相同）

需要 Python 3.9+（建议 3.10/3.11）。

```bash
cd yolo-pipeline
python -m venv venv
# Mac:      source venv/bin/activate
# Windows:  venv\Scripts\activate
pip install -r requirements.txt
```

验证设备识别（Mac 应显示 mps，Win 有 N 卡显示 0，否则 cpu）：

```bash
python scripts/common.py
```

> 注意：Windows 上项目路径不要含中文，否则 OpenCV 读图可能失败。

## 二、示例二：车辆分类（建议先做这个，零标注成本）

1. 把图片按类别放进文件夹（每类建议 30 张以上，10~20% 放 val）：

```
datasets/classify/train/car/         汽车图片
datasets/classify/train/motorcycle/  摩托车图片
datasets/classify/train/ebike/       电动车图片
datasets/classify/val/car/           每类分出几张做验证
datasets/classify/val/motorcycle/
datasets/classify/val/ebike/
```

2. 训练（M3 Pro 上约 10~30 分钟）：

```bash
python scripts/train_classify.py
```

3. 测试识别：

```bash
python scripts/predict.py --model runs/classify/weights/best.pt --source 测试图.jpg
```

## 三、示例一：办公室安全隐患检测（需要画标注框）

1. 图片放入 `datasets/detect/images/train`（约 80 张）和 `images/val`（约 20 张）
2. 标注：推荐 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)（开源、支持 Win/Mac、可 AI 辅助标注）或 LabelImg。导出 **YOLO 格式** txt，放入 `datasets/detect/labels/train` 和 `labels/val`，文件名与图片对应
3. 类别编号与名称在 `datasets/detect/dataset.yaml` 中修改
4. 训练（M3 Pro 上约 0.5~2 小时）：

```bash
python scripts/train_detect.py
```

5. 测试（支持图片、文件夹、视频）：

```bash
python scripts/predict.py --model runs/detect/weights/best.pt --source 办公室照片.jpg
```

## 四、拿到 Windows 上用

方式 A（最简单）：把 `runs/.../weights/best.pt` 拷到 Win 电脑，装好同样环境后直接用 `predict.py` 推理。

方式 B（嵌入其他程序）：导出 ONNX，可被 C# / C++ / Java 的 onnxruntime 调用：

```bash
python scripts/export_onnx.py --model runs/detect/weights/best.pt
```

## 五、模型存放位置

所有 AI 模型统一存放在 **models 文件夹**（`/Volumes/DB_004/AI视觉智能监测模型/models`）：

- 预训练权重（yolo11n.pt 等）首次训练时自动下载到该文件夹，之后复用
- 训练产出的最优权重自动复制为 `classify_best.pt` / `detect_best.pt` 存入该文件夹
- 换电脑（如 Windows）时：该硬盘不在时自动退回项目内 `models/` 文件夹，也可用环境变量 `MODELS_DIR` 指定其他位置

## 六、常见问题

- **准确率不到 60%？** 优先加数据（每类 50~100 张最有效）；其次加 epochs（`--epochs 200`）；再考虑换大一点的模型（`--model yolo11s.pt`）
- **想用最新的 YOLO26？** 训练时加 `--model yolo26n.pt` 即可，接口完全相同（需 ultralytics 最新版）
- **训练指标在哪看？** `runs/` 下每次训练会生成 results.png（损失和准确率曲线）、混淆矩阵等
- **授权说明**：Ultralytics 为 AGPL-3.0，内部使用/教学无问题；若做闭源商用产品需商业授权

## 目录结构

```
yolo-pipeline/
├── requirements.txt
├── scripts/
│   ├── common.py           # 设备自动检测（mps/cuda/cpu）
│   ├── train_classify.py   # 分类训练（示例二）
│   ├── train_detect.py     # 检测训练（示例一）
│   ├── predict.py          # 推理（图片/文件夹/视频）
│   └── export_onnx.py      # 导出 ONNX
├── datasets/
│   ├── classify/           # 分类数据：按类别分文件夹
│   └── detect/             # 检测数据：images + labels + dataset.yaml
└── runs/                   # 训练和推理输出
```
