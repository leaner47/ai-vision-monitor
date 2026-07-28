# VLM 风险分析指南（画面因果推理）

用本地视觉大模型（Qwen3-VL）分析画面并推断风险：点燃的蜡烛 → 火灾风险；线缆乱放、插头未插好 → 漏电触电风险。类似 UniFi 监控的 AI 事件分析，但完全本地运行、免费、开源。

## 一、安装 Ollama（只需一次）

1. 下载安装：https://ollama.com/download （Mac 版，拖进应用程序即可）
2. 让模型存到 AI视觉智能监测模型 文件夹（而不是默认的用户目录），在终端执行：

```bash
launchctl setenv OLLAMA_MODELS /Volumes/DB_004/AI视觉智能监测模型/ollama
```

然后**完全退出并重启 Ollama 应用**使其生效。

3. 下载视觉模型（约 6 GB，一次性）：

```bash
ollama pull qwen3-vl:8b
```

## 二、运行风险分析

在 yolo-pipeline 目录、激活 venv 后：

```bash
# 摄像头实时巡检（每 10 秒分析一帧）
python scripts/risk_monitor.py --source 0

# 分析单张照片
python scripts/risk_monitor.py --source office.jpg

# 批量分析文件夹
python scripts/risk_monitor.py --source photos/
```

输出示例：

```
[风险] 桌面有一支点燃的蜡烛，靠近纸质文件 → 火灾风险 → 立即熄灭或移离可燃物
[风险] 地面多根电线缠绕乱放 → 绊倒及漏电风险 → 整理入线槽并固定
```

发现风险的画面会自动连同分析文字存档到 `runs/risk/`，可作为巡检记录。

## 三、现场演示建议

桌上放一个打火机点燃的蜡烛（或手机播放明火视频对着摄像头）、故意把插头拔出一半、散乱几根线缆，跑摄像头模式即可现场演示风险提示。

## 四、Windows 部署

Ollama 有 Windows 版（同一下载页）。安装后同样 `ollama pull qwen3-vl:8b`，脚本原样运行。无独立显卡时用 CPU 推理，每帧约 30~60 秒，适合定时抽帧巡检而非实时；有 NVIDIA 显卡则快得多。

## 五、与 YOLO 的关系（架构说明）

- **YOLO**（已跑通）：毫秒级，认识"训练过的东西"，适合实时逐帧盯防
- **VLM**（本指南）：秒级，能推理没见过的因果关系，适合抽帧深度分析
- 生产系统通常两层结合：YOLO 实时监测 → 触发或定时 → VLM 分析因果 → 告警

## 常见问题

| 现象 | 解决 |
|---|---|
| 连不上 Ollama | 先打开 Ollama 应用，或终端运行 `ollama serve` |
| 提示模型不存在 | 执行 `ollama pull qwen3-vl:8b` |
| 分析很慢 | M3 Pro 每帧约 5~15 秒属正常；可换小模型 `--model qwen3-vl:4b` |
| 嫌 8B 不够准 | 内存 36GB 可试 `ollama pull qwen3-vl:32b`（量化版约 20GB，速度较慢） |
