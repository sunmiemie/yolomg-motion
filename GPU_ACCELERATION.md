# AMD GPU 加速

2026-09-14 本机检查：显卡为 AMD Radeon 780M，OpenCL 设备为 `gfx1103`；DXGI 适配器 0 为 AMD 硬件设备。原环境为 Python 3.9、CPU 版 PyTorch，无可用 CUDA。现使用 DirectML 执行 YOLOMG 网络前向，使用 OpenCV OpenCL/UMat 执行支持的图像运算。

## 运行

在项目目录用 PowerShell 调用：

```powershell
.\run_gpu.ps1 -InputVideo 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' -OutputDir 'runs/gpu_72eb'
```

目录必须不存在或为空。该入口保持此前参数：模型尺寸 1280、检测和锁定阈值 0.10、每 3 帧检测一次、最大无有效检测间隔 90 帧。`-MaxFrames 120` 可做短片检查，0 为整段；`-GpuId 0` 指定 DirectML 适配器。默认将计算量较小的模板匹配保留在 CPU，因为本机短片实测 OpenCL 模板匹配更慢；仍可显式传 `-TrackingBackend opencl` 比较。

等价命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m yolomg_motion.track_cli --input '视频路径.mp4' --output '新输出目录' --device directml --gpu-id 0 --motion-backend opencl --tracking-backend cpu --image-size 1280 --confidence 0.10 --lock-confidence 0.10 --detect-every 3 --max-detector-gap 90 --threads 4
```

输出仍严格保留 `tracked_zoom.mp4` 和 `tracks.json`。JSON 的 `metadata` 保存所选后端、实际运行信息和分阶段耗时。CPU 命令仍可使用，不会自动改用 GPU；显式选择不可用的 GPU 后端会报错。

## 独立环境

已创建 `.venv-gpu`，继承原有 CPU 环境的已装依赖，并在该环境中安装以下包；原全局 PyTorch 和 ONNX Runtime 未替换。

```powershell
python -m venv --system-site-packages .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-amd.txt
```

已安装版本：`onnx==1.17.0`、`onnxruntime-directml==1.19.2`，适配本机 Python 3.9。它们是已验证的固定依赖，不表示当前最新版本。

## 执行范围

| 环节 | 执行方式 |
|---|---|
| YOLOMG 双输入网络前向 | ONNX Runtime DirectML，首次推理分析实际执行的 provider |
| 高斯滤波、灰度、缩放、稀疏 LK、透视变换、有效区域、差分 | OpenCV UMat/OpenCL 路径 |
| 局部模板匹配 | 推荐 CPU；可选 OpenCL；轨迹关联规则保持不变 |
| RANSAC、单应矩阵检查、目标关联和状态管理 | CPU |
| 网络输入整理、NMS、绘图、视频编码和 JSON 写入 | CPU |

UMat 表示请求 OpenCL 加速，OpenCV 可能将个别算子回退到 CPU；本实现不将启用 OpenCL 等同于逐算子 GPU 执行证明。JSON 中明确记录该限制。CPU 与 OpenCL 的 LK 实现存在数值差异，可能影响配准、候选框和后续轨迹，不能声称加速前后结果逐帧完全相同或精度必然提高。

差异也涉及 LK 参数语义：本机 OpenCV 4.13.0 的 OpenCL 内核固定使用每个坐标轴 `0.01` 的收敛阈值和行列式检查，没有接收 CPU 路径的 `criteria.epsilon`（本项目为 `0.003`）与 `minEigThreshold`。因此该路径属于 GPU 加速实现，不能当成 CPU LK 参数完全一致的复现。需要保持原 CPU 光流参数语义时，使用 `--device directml --motion-backend cpu --tracking-backend cpu`，只加速网络。[OpenCV 4.13.0 LK 内核源码](https://github.com/opencv/opencv/blob/4.13.0/modules/video/src/opencl/pyrlk.cl#L302)；[收敛条件](https://github.com/opencv/opencv/blob/4.13.0/modules/video/src/opencl/pyrlk.cl#L518)。

DirectML 首次运行需导出并初始化与输入尺寸匹配的 ONNX 模型，缓存位于 `.cache/onnx`，属于模型缓存，不放在视频结果目录。第一次耗时包含这项开销，后续复用缓存；改变权重或输入尺寸可能生成新的模型缓存。

缓存重放与实际 GPU 推理在运行信息中区分。旧 CPU 检测缓存不能直接当作 OpenCL 运动输入或 DirectML 检测结果使用；保存的运动缓存不运行 GPU 配准。

## 本机对比实测

2026-09-14，作者 ARD100 的 `phantom02.mp4` 前 120 个输入帧（约 4 秒），输出中心帧 2–117，共 116 帧。三组使用同一权重、同一输入、同一参数、4 个 CPU 线程，顺序运行，每组一次，不重放检测或运动缓存。固定尺寸 ONNX 模型已缓存。下表为 CLI 的处理阶段计时，包含视频输出，不包含进程启动、最初的 PyTorch 模型加载及运动后端初始化；DirectML 的延迟 session 初始化包含在表中。首次导出模型另有开销。

| 环节 | 全 CPU | 检测/光流/模板均选 GPU 后端 | 推荐：GPU 检测与光流，CPU 模板 |
|---|---:|---:|---:|
| 检测，39 次 | 29.05 秒 | 9.81 秒 | 10.93 秒 |
| 光流对齐与三帧差分 | 25.49 秒 | 9.93 秒 | 10.49 秒 |
| 跟踪 | 0.17 秒 | 12.18 秒 | 0.18 秒 |
| 可视化与编码 | 2.14 秒 | 2.35 秒 | 2.20 秒 |
| **总处理耗时** | **57.62 秒** | **34.95 秒** | **24.69 秒** |
| 相对 CPU 加速 | 1.00 倍 | 1.65 倍 | **2.33 倍** |

小块模板匹配启用 OpenCL 后在本次测试中明显变慢，故默认保留 CPU。以上并非多轮稳定性能统计，也不是实时处理承诺；约 4 秒输入仍需约 24.69 秒处理。此前整段视频约一分钟的耗时不能直接与本表比较。

三组都对照作者标注验证，采用 IoU ≥ 0.5 判定正确锁定，无标注反馈给检测器或跟踪器：116 帧中均有 106 帧正确锁定、0 帧错误锁定、10 帧未锁定。全部有目标帧的平均 IoU（未锁定计 0）从 CPU 的 0.66695 变为 GPU 两组的 0.66381，数值并不完全一致。该结果只覆盖这段短片，不代表整个 ARD100 测试集精度或论文 AP。

推荐组合的 `tracks.json` 中保存完整后端信息、三组计时和标注对照结果。首次网络推理 profiling 记录 1 个融合后的 DirectML 图节点、0 个 CPU 图节点；这仅证明网络图的执行方式，不包括预处理、NMS 或逐个 OpenCL 算子。最终视频完整解码 116 帧通过。工程测试共 50 项通过。

本次短片结果目录：`runs/gpu_ard100_phantom02_120f_20260914`，只包含 `tracked_zoom.mp4` 和 `tracks.json`。

## 迁移至 NVIDIA

网络适配器也提供 `--device cuda --gpu-id 0`，需要在目标机器安装匹配的 CUDA 版 PyTorch。本机没有 NVIDIA 显卡，因此未完成 CUDA 硬件实测。图像后端当前提供 CPU/OpenCL，没有宣称使用 OpenCV CUDA；普通 pip 安装的 OpenCV 在本机不含 CUDA 光流模块。

## 依据

- [ONNX Runtime DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)：DirectX 12 GPU、串行 session、禁用 memory pattern 和静态尺寸要求。
- [Microsoft PyTorch DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)：torch-directml 的版本限制；本项目选择 ONNX Runtime 以保留原 PyTorch 环境。
- [OpenCV OpenCL 优化说明](https://github.com/opencv/opencv/wiki/OpenCL-optimizations)：UMat 支持与 CPU 回退说明。
