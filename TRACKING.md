# 自动无人机检测与持续跟踪

## 处理流程

1. 对中心帧 `t` 与前后 `t-2`、`t+2` 做稀疏 LK 光流和 RANSAC 单应配准。
2. 计算两幅绝对差分的均值。检测器使用未施加形态学处理和显示增益的 `difference`；第二路输入是三通道灰度差分，不是热力图。
3. 用作者官方 `ARD100_mask32-1280_uavs/weights/best.pt` 进行双路 YOLOMG 无人机检测及 NMS。预处理按模型要求做 letterbox、BGR 转 RGB、浮点除以 255，再把检测框恢复至原视频坐标。
4. 连续两次空间一致的有效检测后自动锁定一个目标。锁定期间按位置、尺寸关联，防止突然切换到远处高分候选。默认网络每 3 帧运行一次，中间帧通过局部归一化相关匹配跟踪外观。
5. 模板仅在有效检测确认位置时更新；局部匹配失败或超过 15 个原视频帧没有有效检测时，停止画框并进入 LOST。再次出现需重新确认，分配新的 ID。

此版本复用官方网络与训练权重，并增加单目标跟踪逻辑，没有重跑论文训练或验证集评测。对小目标、遮挡、相似干扰物和明显域差异仍可能漏检或误检；本视频的锁定覆盖率不能用作检测准确率。置信度与模板相关系数也不是同一个指标。

## 运行

本机 Python：`C:/dev/python/python.exe`，PyTorch 2.8.0+cpu、torchvision 0.23.0。CPU 可运行，实际速度见各次 `tracks.json` 的 `metadata`。现有环境已具备依赖；新环境需要基础项目依赖及 `requirements-detection.txt`。

```powershell
Set-Location 'D:/Users/Desktop/bxk/small/yolomg_motion'
python -m yolomg_motion.track_cli --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/auto_tracking --lock-confidence 0.10 --max-detector-gap 90
```

复用之前已生成的完整差分缓存可以节省配准计算：

```powershell
python -m yolomg_motion.track_cli --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/auto_tracking_cached --motion-cache runs/full_72eb_20260911 --lock-confidence 0.10 --max-detector-gap 90
```

`--output` 必须不存在或为空。缓存必须来自同一路径的输入视频，`k=2` 且第一中心帧为 2；不要用同名替换后的视频复用旧缓存。无缓存时自动计算差分。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--detect-every` | 3 | 每隔多少个输出帧运行一次网络。 |
| `--confidence` | 0.10 | 网络 NMS 后保留的候选最低置信度。 |
| `--lock-confidence` | 0.20 | 允许参与锁定和关联的最低检测置信度。 |
| `--max-detector-gap` | 15 | 允许图像匹配补偿的最长无有效检测帧数；本视频示例使用 90。 |
| `--image-size` | 1280 | 网络输入最长边大小；保留长宽比。 |
| `--threads` | 4 | CPU 推理线程数。 |
| `--device` | cpu | 神经网络后端：cpu、cuda（NVIDIA）或 directml（Windows GPU）。本机 AMD 使用 directml。 |
| `--gpu-id` | 0 | CUDA 设备编号或 DirectML 的 DXGI 适配器编号。 |
| `--motion-backend` | cpu | cpu 或 opencl，控制光流配准及差分图像运算。 |
| `--tracking-backend` | cpu | cpu 或 opencl，控制灰度与模板匹配；轨迹状态逻辑仍用 CPU。 |
| `--weights` | 官方 1280 权重 | 可明确指定兼容权重路径。 |
| `--repo` | `vendor/YOLOMG` | 与权重配套的官方源码目录。 |
| `--motion-cache` | 无 | 复用三帧差分模块输出的完整缓存。 |
| `--detection-cache` | 无 | 复用已有跟踪运行目录中的真实模型候选；严格核对输入路径、模型路径、尺度、阈值、检测间隔及逐帧状态。 |
| `--max-frames` | 0 | 最多读取多少原视频帧，0 为完整视频。 |

## 输出与阅读方式

每个跟踪结果目录只保留两个文件：

- `tracked_zoom.mp4`：原画面加黄色目标框、目标 ID、状态和局部像素放大窗口。放大只为看清原有像素，不做图像生成或细节增强。
- `tracks.json`：`schema_version` 为 2；`metadata` 保存模型、参数、帧数、锁定状态统计和实际耗时；`frames` 数组保存每帧的原始框坐标 `bbox=[x1,y1,x2,y2]`、目标 ID、状态、当前观测来源、当帧网络候选等。

不再自动生成截图、无放大视频、单独的 H.264 副本或 `summary.json`。读取逐帧轨迹示例：

```python
import json
from pathlib import Path

result = json.loads(Path('runs/auto_tracking/tracks.json').read_text(encoding='utf-8'))
metadata = result['metadata']
frames = result['frames']
```

检测缓存重放和验收工具兼容旧版“轨迹数组 + summary.json”目录，也可读取新版单 JSON。已有跟踪结果按此结构整理，原来的逐帧轨迹保持不变。

`LOCKED` 表示当前跟踪状态；`DETECTOR` 表示该帧来自网络检测，`TEMPLATE` 表示该帧来自图像匹配。`SEARCHING`、`CONFIRMING`、`LOST` 都不绘制目标框。模板帧不伪造网络检测置信度。全景黄色框对极小目标加少量显示边距，JSON 保存未加边距的原始坐标。

默认三帧差分需要未来两帧，前瞻延迟约为 `2/FPS`，另加计算时间。完整 552 帧视频输出 548 个中心帧，对应原帧 2 至 549。输出视频无音轨；帧率保持原视频帧率，这不表示计算达到了实时速度。

验收示例：

```powershell
python tools/audit_tracking.py --run runs/auto_tracking_cached
python -m pytest -q
```

验收默认只在终端显示结果，不生成额外报告文件。确有诊断需要时可显式传入 `--report` 指定保存路径。

## 文件与来源

- `yolomg_motion/yolo_detector.py`：官方双路网络的 CPU 推理适配。
- `yolomg_motion/tracking.py`：自动确认、检测关联和有限时长的局部模板跟踪。
- `yolomg_motion/track_cli.py`：视频处理、画框和日志输出。
- `vendor/YOLOMG/SOURCE.json`：官方源码固定提交和文件 SHA-256。
- `vendor/YOLOMG/LICENSE`：作者源码许可证。

官方来源：[Irisky123/YOLOMG](https://github.com/Irisky123/YOLOMG)。本地另一个 FD2/RK3588 交付包使用因果两帧残差和改造架构，本运行不使用该包的权重。

## 72eb 视频实际结果（2026-09-11）

最终输出目录：`runs/drone_lock_72eb_20260911`。

此目录已整理为 `tracked_zoom.mp4` 和 `tracks.json`，删除了重复视频、截图和单独的报告；参数及统计保存在 JSON 的 `metadata` 中。

- 原视频 552 帧，1280×720、30 FPS；处理中心帧 2–549，共 548 帧。
- 无手动框选。前期自动搜索、确认；原视频第 56 帧（1.867 秒）锁定 ID 1，此后连续跟踪至第 549 帧。
- 494 个 LOCKED 帧，其中 63 帧来自已关联网络检测，431 帧来自局部图像跟踪；锁定后无 ID 切换、无失锁。
- 最终参数：网络候选阈值 0.10、锁定/关联阈值 0.10、检测间隔 3 帧、最大无有效检测间隔 90 帧。该段视频实际最长间隔 80 帧，期间依赖每帧外观匹配。与默认 15 帧上限相比，这允许跨越更长的模型漏检，同时增加需要留意的漂移风险。
- 保留的视频完整解码 548 帧，通过坐标、身份和 JSON 结构验收；抽查中段靠近灯杆和长时间模型漏检的帧，框仍位于空中目标。没有逐帧人工标注，90.1% 锁定状态覆盖不是识别准确率。
- 首次完整神经网络运行 183 次，CPU 检测累计 112.41 秒；含缓存差分读取、跟踪及视频编码约 127.99 秒（不含模型加载及之前差分计算）。最终调整跟踪参数时复用了这批真实检测，重放与导出耗时约 14.63 秒，不能将重放速度当成端到端推理速度。

完整网络候选保存在 `runs/tracking_full_72eb_20260911`。最终视频复现命令：

```powershell
python -m yolomg_motion.track_cli --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/auto_tracking_replay --motion-cache runs/full_72eb_20260911 --detection-cache runs/tracking_full_72eb_20260911 --lock-confidence 0.10 --max-detector-gap 90
```

不传 `--detection-cache` 即重新运行模型。不要用已被替换的同名视频或权重复用旧缓存。切片和整帧放大到 1920 的独立实测未改善漏检且产生更多误检，因此最终版本保留 1280 整帧模型。

## GPU 运行

本机已确认 AMD Radeon 780M。GPU 运行使用独立 `.venv-gpu` 环境中的 DirectML 网络推理和 OpenCV OpenCL 图像处理；原有 CPU 命令仍可使用。GPU 入口为 `run_gpu.ps1`，安装方法、实际执行范围和速度验证见 [GPU_ACCELERATION.md](GPU_ACCELERATION.md)。运行结果仍只有 `tracked_zoom.mp4` 和 `tracks.json`；所选后端、实际运行信息与分阶段计时保存在 `metadata` 中。
