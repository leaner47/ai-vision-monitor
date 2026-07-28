# Windows 运行指南

在 Windows 上使用 Mac 训练好的模型（或直接在 Win 上训练），全部命令如下。

## 零、双击启动（推荐日常使用）

完成下面"一、二"的环境准备后，日常使用只需**双击项目里的 `快速启动.bat`**：自动启动 Ollama + Dashboard → 自动打开浏览器 → 关闭浏览器页面约 45 秒后自动停止全部服务并释放内存。黑色窗口运行期间可最小化，勿关闭。可右键该文件 → 发送到 → 桌面快捷方式，把图标放到桌面。

前提：按本指南装好 Python 环境和 venv；装好 Windows 版 Ollama；把移动硬盘里的 `ollama/` 和 `models/` 文件夹与 `yolo-pipeline/` 保持同级目录关系（模型文件跨平台通用，直接拷贝即可，无需重新下载）。

## 一、准备

1. 安装 Python 3.10 或 3.11：https://www.python.org/downloads/ ，安装时**勾选 "Add Python to PATH"**
2. 把整个 `yolo-pipeline` 文件夹从移动硬盘拷到 Win 电脑，例如 `D:\yolo-pipeline`
   （**路径不要含中文**，否则读图可能失败）
3. 把训练好的模型文件（如 `classify_best.pt`、`detect_best.pt`）拷到 `D:\models`（或任意英文路径）

> 注意：Mac 上的 `venv` 文件夹不能在 Win 上用，不用拷，Win 上重新建。

## 二、安装环境（只需一次，用 cmd 或 PowerShell）

```bat
cd /d D:\yolo-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

如果电脑有 NVIDIA 显卡，想用 GPU 训练，先装 CUDA 版 PyTorch 再装其余依赖：

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

指定模型存放目录（可选；不设则自动用项目内 `models\` 文件夹）：

```bat
set MODELS_DIR=D:\models
```

验证（有 N 卡显示 0，没有显示 cpu）：

```bat
python scripts\common.py
```

## 三、用训练好的模型做识别（最常用）

```bat
cd /d D:\yolo-pipeline
venv\Scripts\activate

:: 识别单张图
python scripts\predict.py --model D:\models\classify_best.pt --source C:\photos\test.jpg

:: 识别整个文件夹
python scripts\predict.py --model D:\models\detect_best.pt --source C:\photos\

:: 识别视频
python scripts\predict.py --model D:\models\detect_best.pt --source C:\videos\road.mp4
```

结果（画好框/标签的图片）保存在 `runs\predict*\`，终端同时打印识别内容。

## 四、在 Win 上训练（可选）

数据放法与 Mac 完全相同（见项目 README），命令一样：

```bat
python scripts\train_classify.py
python scripts\train_detect.py
```

没有独立显卡也能训练（cpu 模式），100 张图规模大约几小时，能跑通但较慢。

## 五、嵌入其他程序（可选）

导出 ONNX 后可被 C# / C++ / Java 的 onnxruntime 调用，不依赖 Python：

```bat
python scripts\export_onnx.py --model D:\models\detect_best.pt
```

## 常见报错

| 现象 | 解决 |
|---|---|
| `python 不是内部或外部命令` | 重装 Python 并勾选 Add to PATH，或用 `py` 代替 `python` |
| `venv\Scripts\activate 无法加载` (PowerShell) | 先执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 读不到图片 | 检查路径是否含中文，改为纯英文路径 |
| 训练极慢 | 正常（CPU 模式）；减小 `--epochs` 或换有 N 卡的机器 |
