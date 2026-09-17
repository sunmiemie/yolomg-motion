# 首次检测后 ROI 跟踪

功能分支：`feature/roi-tracking`。`main` 保留原有 CPU / GPU 持续检测版本。

目标设备为用户指定的 **ATK-DLRK3588**，输入按 **2K = 2560×1440** 准备；若实际相机的“2K”为其他宽高，算法读取真实尺寸，不强制拉伸。60 FPS 是待板端验证的要求，目前没有连接板卡，也没有完成 RKNN 模型转换、板上取流及延迟测试。

## 实现行为

1. 搜索目标时，默认每 3 个源帧尝试一次全图配准、三帧差分和 YOLOMG 检测。
2. 两次空间一致的检测确认目标，随后仅在预测位置附近裁剪 ROI、转灰度、做归一化模板匹配。正常锁定帧不调用检测器，不计算全图光流或差分，也不对全图转灰度。
3. 当前模板匹配低于阈值、模板无有效纹理、目标出界等情况返回 `LOST`，不输出仅靠预测补出的目标框；下一可处理帧立即尝试重新检测。重新获取目标使用新 ID。
4. 默认 `--verify-every 0`，不因检测器长期未运行而强制失锁。可设置 `--verify-every 60` 等间隔做检测校验，校验失败则解除锁定。它会增加计算量并产生同步停顿，不是后台异步检测。

ROI 使用速度预测和默认 32 像素搜索半径，返回原图坐标。模板为检测框加 2 像素总边长、最小 8×8 像素，以减少小无人机周边背景对匹配的影响。这里的 8×8 是最小模板尺寸，不是光流采样网格。框尺寸在两次检测间保持不变，暂未实现尺度自适应；目标转向、缩放、遮挡及相似背景仍可能导致失锁或错误锁定。模板只由检测更新，未使用人工目标坐标或标注参与跟踪。

## 本机运行

```powershell
.\run_roi_gpu.ps1 -InputVideo 'input.mp4' -OutputDir 'runs/roi_demo'
```

默认只输出 `tracked_zoom.mp4` 与 `tracks.json`。视频保持输入 FPS，既不补帧，也不把视频标签改成 60 FPS 冒充性能。只测坐标流程、跳过绘制与编码时加 `-NoVideo`，这时仅输出 `tracks.json`。

对应 Python 命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m yolomg_motion.roi_cli --input 'input.mp4' --output 'runs/roi_demo' --device directml --motion-backend opencl --verify-every 0 --target-fps 60
```

`--max-frames 120` 测试短片，`--search-radius` 调整搜索范围，`--appearance-threshold` 调整外观匹配门限，`--confirmations 1` 可改为一次检测后锁定，但错误初始检测更容易直接成为跟踪目标。默认保留两次确认。

## 时间、延迟和输出记录

继续使用作者模型所需的中心三帧，间隔 k=2；环形缓冲只保存 5 个原始帧，按源帧顺序输出，首尾各 2 帧不补造。相机为 60 FPS 时，这本身有约 33.3 毫秒前瞻等待；30 FPS 时约 66.7 毫秒。吞吐量 60 FPS 和从拍摄到坐标输出的延迟不是同一个指标。2K BGR 的 5 帧缓冲约 52.7 MiB，不包括网络、解码缓冲和其他工作内存。

`tracks.json` 明确记录：

- `motion_computed=false`、`motion_valid=null`：该帧主动跳过全图运动处理；null 不代表配准失败。
- `mode`、`detection_reason`、`roi_tracking`、`search_roi`：处理路径和搜索区域。
- `timing.roi_compute`：锁定后局部计算的均值、P95、P99、最大值及超过 16.7 毫秒的帧数。
- `timing.roi_with_decode_and_output`：同类帧增加解码及可选绘制编码后的统计。
- `timing.acquisition_compute`：首次检测、重新捕获及校验耗时。
- `processing_fps`：整段处理帧数除以 wall time。包括文件解码和启用的视频输出，不含前置模型加载和末尾 JSON 序列化；`setup_seconds` 单独记录前置初始化。

首次识别及重新检测目前是同步路径，可能超过 16.7 毫秒。因此，即使平均 FPS 或局部跟踪达到 60，也不表示每一帧都能按期输出；输出失锁标志也不能当作正确目标坐标。

## RK3588 上的验证与后续接入

本分支的 ROI 核心仅依赖 NumPy / OpenCV CPU，可独立运行；电脑上的 DirectML 不作为 RK3588 的板端推理实现。Rockchip 官方流程需要先用 RKNN-Toolkit2 转换模型，再用板端 RKNN-Toolkit-Lite2 或 C/C++ Runtime 推理，官方支持 RK3588：[官方 RKNN 仓库](https://github.com/airockchip/rknn-toolkit2)。

YOLOMG 是 RGB 和运动图双输入、含自定义融合模块的网络。官方 Model Zoo 的普通 YOLOv5 示例不能直接证明该模型已成功转换；本分支不包含伪装为可用的 `.rknn` 文件，也未宣称 NPU 已跑通。`ROIPipeline` 的检测接口为 `detect(frame_bgr, motion_float32)`，返回含 `bbox_xyxy` 和 `confidence` 的对象列表，便于接入经过验证的板端检测器。

可先在板上测局部 CPU 跟踪开销，无需模型、Torch 或 GPU：

```bash
python3 -m pip install -e .
python3 tools/benchmark_roi.py --width 2560 --height 1440 --frames 600 --threads 4
```

这条命令是合成目标的局部跟踪微基准：初始框由程序提供，排除了相机、解码、图像生成、检测及输出，不能当作自动检测效果或端到端 60 FPS 证明。默认只打印统计，不生成文件。

上板验收还需要测真实 2K60 输入、长时间运行和失锁重捕获，并核对 P95/P99 延迟、目标坐标正确性及丢帧数；若运行全图检测时也必须保证 60 Hz 输出，则还需要相机采集与检测异步化、过期结果处理，以及明确失锁期间的输出协议。

## 验证

自动测试覆盖正常锁定后跳过检测与运动计算、超过旧检测间隔仍能跟踪、目标消失与重新捕获、校验失败、无效光流、图像边缘、2K 局部灰度转换、原图帧号与视频一致、无视频模式、异常保留部分 JSON 及拒绝覆盖已有结果。

实测结果见 [ROI_VALIDATION.md](ROI_VALIDATION.md)。
