# YOLOMG 运动特征与无人机检测跟踪

首次克隆后的安装、权重下载与代码来源见 [REPOSITORY.md](REPOSITORY.md)。视频、测试集、模型权重与运行结果不随 Git 仓库上传。

本项目独立实现 YOLOMG 论文第 III-A 节的运动特征增强模块（MFEM）：高斯平滑与灰度化、网格采样、稀疏金字塔 Lucas–Kanade（LK）光流跟踪、RANSAC 单应矩阵估计、摄像头运动补偿、三帧差分，以及可调的形态学处理。运动模块由本项目实现；新增的检测模块使用 `vendor/YOLOMG` 中的作者源码和官方 1280 预训练权重，保留其 GPL-3.0 许可证和来源校验记录。

原有 `python -m yolomg_motion` 命令仍只输出配准与运动差分，运行只需 NumPy 和 OpenCV。新增 `python -m yolomg_motion.track_cli` 将原始彩色图和三帧差分输入官方 YOLOMG 网络，再用时序关联与局部模板匹配锁定目标，输出带黄色框的视频。检测需要 PyTorch 和预训练权重，支持 CPU。跟踪是本项目增加的功能，不属于论文原有检测网络，也没有重新训练模型。输入视频和输出图像均在本机处理，程序不会上传原图或视频。

## 自动检测并持续跟踪

完整参数与结果说明见 [TRACKING.md](TRACKING.md)。本机 AMD 780M 的 GPU 加速入口为 `run_gpu.ps1`，运行方法与 CPU/GPU 对比见 [GPU_ACCELERATION.md](GPU_ACCELERATION.md)。下面是默认 CPU 入口，在项目目录运行：

```powershell
python -m yolomg_motion.track_cli --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/auto_tracking --lock-confidence 0.10 --max-detector-gap 90
```

每次跟踪只输出两个文件：`tracked_zoom.mp4`（黄色目标框、编号、状态和目标局部放大）与 `tracks.json`（`metadata` 保存运行参数和统计，`frames` 保存逐帧位置、身份与检测/跟踪来源）。不再自动生成截图、重复视频或单独的汇总报告。默认每 3 帧运行一次网络，两次空间一致的检测后确认目标；中间帧使用局部图像匹配。上面的本视频配置将有效检测最低置信度设为 0.10，无有效检测上限设为 90 帧；期间每帧都必须通过外观匹配，否则立即失锁。省略两项参数时使用更保守的 0.20 和 15 帧。没有预设无人机坐标，也不需要手动框选。

## 快速运行

项目目录为 `D:/Users/Desktop/bxk/yolomg_motion`。运行需要 Python 3.9 或以上，以及 NumPy 和 OpenCV。项目声明的依赖为 `numpy>=1.23`、`opencv-python>=4.8`；测试额外需要 `pytest>=7`。现有 Python 环境已经能够导入 `cv2` 和 `numpy` 时，可以直接从项目目录运行，无需安装本项目或修改其他工程。

PowerShell 示例：

```powershell
Set-Location 'D:/Users/Desktop/bxk/yolomg_motion'
python -m yolomg_motion --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/demo --max-frames 120 --save-video --save-maps
```

该命令最多读取 120 个输入帧。默认帧间隔为 `k=2`，若成功读取全部 120 帧，将产生 116 个中心帧的结果记录，其中可能包含明确标记的配准失败记录。`runs/demo` 必须不存在或为空；重复运行时请更换输出目录，例如 `runs/demo2`，程序不会覆盖已有结果。

完整参数帮助：

```powershell
python -m yolomg_motion --help
```

## 三帧差分与时间含义

对于中心时刻 `t`，取 `t-k`、`t`、`t+k` 三帧。前帧和后帧分别配准到中心帧坐标系，然后按照论文公式 (1) 计算：

$$
E_t = \frac{\left|I_t-\widehat{I}_{t-k}\right|+\left|I_t-\widehat{I}_{t+k}\right|}{2}.
$$

其中，`I_t` 为预处理后的中心灰度图，带帽的前后帧为经过运动补偿的灰度图。这里使用的是**两幅绝对差分图的算术平均**，不是二值图的 AND，也没有先进行二值阈值化。

`difference` 是公式输出，已将无效边界区域置零；`motion` 是在其基础上进行可选开、闭运算后的结果。若关闭形态学处理，二者相同。

流式接口默认 `k=2`，最多缓存 `2k+1=5` 个帧时刻。收到第 5 帧（从 0 开始编号时为帧 4）后，才输出中心帧 2 的结果。在 30 FPS 视频中，需要等待中心帧之后的两帧，前瞻延迟为 `2/30 ≈ 66.7 ms`，另加计算耗时。程序不会补造序列两端缺少前后邻帧的结果：完整读取 `N` 帧时，最多输出 `max(N-2k, 0)` 个中心帧。

## 命令行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--input` | 必填 | 本地输入视频路径。 |
| `--output` | 必填 | 输出目录，须不存在或为空。 |
| `--frame-step` | `2` | 三帧差分的间隔 `k`，必须为正整数。 |
| `--max-frames` | `0` | 最多读取的输入帧数；`0` 表示读取到视频结束，不是输出帧数限制。 |
| `--start-frame` | `0` | 从 0 开始编号的起始输入帧。程序顺序跳过此前帧，以保持源视频帧号一致。 |
| `--save-video` | 关闭 | 保存六格对比视频 `comparison.mp4`。 |
| `--save-maps` | 关闭 | 保存每个中心帧的浮点 NPZ 数据及未施加显示增益的灰度 PNG。 |
| `--snapshot-every` | `30` | 首个结果保存一张六格快照，此后每隔该数量的结果再保存一张。 |
| `--preview-gain` | `6.0` | 差分热力图的固定显示增益，必须大于 0；不影响原始差分数据。 |
| `--fb-threshold` | `1.5` | LK 正反向跟踪误差阈值，单位为估计分辨率下的像素；`0` 关闭此项筛选。 |
| `--alignment-width` | `1920` | 光流与单应矩阵估计使用的图像宽度。 |
| `--alignment-height` | `1080` | 光流与单应矩阵估计使用的图像高度。 |
| `--config` | 无 | JSON 文件，用于覆盖 `MotionConfig` 配置。 |
| `--no-morphology` | 关闭 | 最后将开运算、闭运算迭代数均设为 `0`。 |

配置优先级为：`MotionConfig` 默认值 → 命令行中的相关参数 → `--config` 中的同名字段 → `--no-morphology`。因此，**JSON 不仅覆盖命令行默认值，也会覆盖显式传入的同名配置**；`--no-morphology` 则最终覆盖 JSON 中的开、闭运算设置。JSON 只能包含下表列出的 `MotionConfig` 字段，不用于设置 `input`、`save_video` 等命令行选项。

## 核心配置参数

可以通过 `MotionConfig(...)` 或 JSON 调整以下参数。

| JSON / Python 字段 | 默认值 | 含义 |
|---|---:|---|
| `frame_step` | `2` | 流式接口的前后帧间隔。 |
| `blur_size` | `11` | 高斯平滑核尺寸，正奇数。 |
| `grid_step` | `64` | 估计图像中网格采样点的水平间距，单位为像素。 |
| `grid_step_y` | `48` | 估计图像中网格采样点的垂直间距，单位为像素。 |
| `alignment_width` | `1920` | 估计图像宽度。 |
| `alignment_height` | `1080` | 估计图像高度。 |
| `lk_window` | `15` | LK 搜索窗口边长，正奇数。 |
| `lk_levels` | `3` | OpenCV `maxLevel`，即最高金字塔层级编号。 |
| `lk_iterations` | `30` | LK 最大迭代次数。 |
| `lk_epsilon` | `0.003` | LK 迭代终止精度。 |
| `ransac_threshold` | `3.0` | RANSAC 重投影误差阈值，单位为估计图像中的像素。 |
| `min_inliers` | `15` | 最少有效跟踪点数及最少 RANSAC 内点数，至少为 `4`。 |
| `min_inlier_ratio` | `0.25` | RANSAC 最低内点比例，取值 `(0, 1]`。 |
| `fb_threshold` | `1.5` | 正反向 LK 一致性阈值；`0` 关闭。 |
| `min_overlap` | `0.4` | 每次配准在边界剔除后的最低有效区域占比，取值 `(0, 1]`。 |
| `morph_kernel` | `3` | 方形形态学核边长，正奇数。 |
| `open_iterations` | `1` | 开运算次数，`0` 关闭。 |
| `close_iterations` | `1` | 闭运算次数，`0` 关闭。 |

例如，可以在项目目录另存 `config.json`：

```json
{
  "frame_step": 2,
  "alignment_width": 1280,
  "alignment_height": 720,
  "grid_step": 48,
  "grid_step_y": 36,
  "fb_threshold": 1.5,
  "morph_kernel": 3,
  "open_iterations": 1,
  "close_iterations": 1
}
```

然后运行：

```powershell
python -m yolomg_motion --input 'D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4' --output runs/custom --max-frames 120 --config config.json --save-video --save-maps
```

降低估计分辨率会改变光流、网格采样和误差阈值所对应的实际尺度，需要重新检查配准质量。保存的差分图仍使用输入视频的原始分辨率。

## 输出如何阅读

输出目录中包含：

| 文件 | 内容 |
|---|---|
| `comparison_000002.png` 等 | 六格对比快照，文件名中的编号为源视频中心帧编号；实际首个编号由起始帧与 `k` 决定。 |
| `comparison.mp4` | 启用 `--save-video` 时保存的六格对比视频，使用源视频 FPS。 |
| `summary.json` | 输入、实际配置、帧数、成功/失败数量、前瞻延迟、运行耗时及配准前后差分统计。 |
| `diagnostics.json` | 每个中心帧的成功状态、两次配准的失败原因、跟踪点数、RANSAC 内点与单应矩阵等信息。 |
| `maps/000002.npz` 等 | 启用 `--save-maps` 时保存的原始数值数据。 |
| `maps/000002.png` 等 | `motion` 四舍五入后保存的 `uint8` 灰度图，未乘显示增益。 |

六格图按从左至右、从上至下排列：

1. 当前中心帧原图。
2. 配准到中心帧的前帧灰度图。
3. 配准到中心帧的后帧灰度图。
4. 未配准的平均绝对差分热力图。
5. 配准后的平均绝对差分热力图。
6. 形态学处理后的运动差分热力图。

下排三个热力图使用相同的固定 `preview_gain`，先乘增益、截断到 `[0, 255]`，再映射伪彩色；程序不会对每幅图分别归一化。第 4 格也使用与第 5 格相同的有效区域掩码，以便比较。**显示增益只影响快照和对比视频，NPZ 数据及单独保存的灰度 PNG 不受影响。**

NPZ 包含以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `difference` | `float32`，`H×W`，范围 `[0, 255]` | 配准后的平均绝对差分。 |
| `motion` | `float32`，`H×W`，范围 `[0, 255]` | 可选形态学处理后的结果。 |
| `valid_mask` | `bool`，`H×W` | 前后配准共同有效的区域。 |
| `center_index` | `int64` 标量 | 源视频中心帧编号。 |
| `success` | `bool` 标量 | 当前三帧配准与差分结果是否有效。 |

读取时先检查成功状态：

```python
import numpy as np

with np.load("runs/demo/maps/000002.npz") as data:
    if not bool(data["success"]):
        raise RuntimeError("该帧配准失败，请查看 diagnostics.json")
    motion = data["motion"]
    valid = data["valid_mask"]
    valid_values = motion[valid]
```

如果任一方向的配准失败，`success=False`，`difference` 与 `motion` 全零，`valid_mask` 全为 `False`；预览图会标注 `INVALID REGISTRATION`。**全零图可能表示配准失败，不能据此判定画面没有运动，必须检查 `success`。**诊断信息会记录诸如跟踪点不足、RANSAC 内点不足、单应矩阵异常或有效区域不足等原因。

`summary.json` 中的配准前后均值在相同有效区域内计算，并且仍包含真正运动的目标。这些均值和 `difference_reduction` 用于观察配准后的差分变化，**不是无人机识别精度、检测 AP 或召回率**。同样，`processing_fps` 是当前实现的处理速度统计，不能用来声称复现了论文完整 YOLOMG 检测器的 FPS。

## 作为 Python 模块使用

公共导入接口位于 `yolomg_motion/__init__.py`：

```python
from yolomg_motion import MotionConfig, MotionExtractor, MotionResult, three_frame_difference
```

`process_triplet(previous, current, following)` 接收三幅相同尺寸的 `uint8` NumPy 图像，支持 `H×W` 灰度图或 `H×W×3` BGR 图，返回 `MotionResult`。调用者负责选择对应 `t-k`、`t`、`t+k` 的帧；这个离线接口不会根据 `frame_step` 自动选帧。

```python
from yolomg_motion import MotionConfig, MotionExtractor

# previous_frame、current_frame、next_frame 由调用者解码或采集。
extractor = MotionExtractor(MotionConfig(frame_step=2))
result = extractor.process_triplet(previous_frame, current_frame, next_frame)

if result.success:
    motion_map = result.motion
    valid_mask = result.valid_mask
    previous_to_center = result.previous_alignment.homography
else:
    print(result.previous_alignment.reason, result.next_alignment.reason)
```

两个 `homography` 都将来源帧坐标映射到中心帧坐标。`result.current` 保留中心帧原图；`result.center_gray` 为预处理后的中心灰度图；`unaligned_difference` 提供未配准的对照差分。离线调用的 `center_index` 默认为 `None`。

流式输入使用 `push(frame)`。缓存未满时返回 `None`，之后返回延迟 `k` 帧的中心结果：

```python
import cv2
from yolomg_motion import MotionConfig, MotionExtractor

cap = cv2.VideoCapture("D:/Users/Desktop/bxk/视频/72ebd203198e372d6b64af2d4fb9531e.mp4")
extractor = MotionExtractor(MotionConfig(frame_step=2))
try:
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = extractor.push(frame)
        if result is None:
            continue
        if result.success:
            print(result.center_index, result.motion.shape)
        else:
            print("配准失败", result.center_index,
                  result.previous_alignment.reason,
                  result.next_alignment.reason)
finally:
    cap.release()
```

开始新序列或改变输入图像尺寸、通道格式前，应调用 `extractor.reset()`。核心 API 不依赖 GUI，也不负责读取、保存视频。

## 与论文、官方参考代码的对应关系

本项目以论文方法为依据，同时参考作者公开代码核对实现细节。它是独立实现的 MFEM 子模块，不能视为作者官方实现的逐字复制，也未宣称已复现论文完整实验指标。

| 项目 | 本实现的处理与说明 |
|---|---|
| 运动特征提取流程 | 使用网格关键点、金字塔 LK、RANSAC 单应矩阵、前后帧向中心帧配准及平均绝对差分，对应论文第 III-A 节。 |
| 估计分辨率 | 默认在 `1920×1080` 上估计，与所核对的官方参考流程保持对应。若输入尺寸不同，令 `S=diag(1920/W, 1080/H, 1)`；采用 `H=S⁻¹ H_est S` 将矩阵正确还原到原图坐标，再对原图进行变换。自定义估计尺寸时，`S` 使用对应宽高比。 |
| 数值运算 | 减法和加法前转换为 `float32`，避免 `uint8` 减法回绕或加法溢出。输出保持 `[0, 255]` 的灰度量纲。 |
| 稳健性补充 | 正反向 LK 一致性筛选、变换几何检查、有效边界掩码、失败状态记录等属于本实现的补充措施，不应归为论文新增贡献。 |
| 形态学处理 | 论文描述了开运算和闭运算，但所核对的 `FD5_mask.py` 中相关操作实际处于注释状态。本实现默认 `3×3` 方形核、开运算 `1` 次、闭运算 `1` 次，这是可调选项，**并非确认过的原始实验参数**。可以使用 `--no-morphology` 关闭，或通过 JSON 调整。 |

参考来源：

- [YOLOMG 论文，arXiv:2503.07115](https://arxiv.org/abs/2503.07115)。
- [作者官方仓库 Irisky123/YOLOMG](https://github.com/Irisky123/YOLOMG)。
- [test_code/FD5_mask.py](https://github.com/Irisky123/YOLOMG/blob/main/test_code/FD5_mask.py)：三帧差分及相关图像处理参考。
- [test_code/MOD_Functions.py](https://github.com/Irisky123/YOLOMG/blob/main/test_code/MOD_Functions.py)：运动配准相关函数参考。
- [test_code/generate_mask5.py](https://github.com/Irisky123/YOLOMG/blob/main/test_code/generate_mask5.py)：离线运动图生成流程参考。

以上链接指向官方 `main` 分支，其内容可能随作者后续提交变化。

## 测试

在项目目录执行：

```powershell
python -m pytest -q
```

测试使用本地合成场景，无需下载数据、权重或联网。现有测试覆盖浮点绝对差分与平均的数值正确性、已知摄像头平移和透视变换的配准方向、非原始估计分辨率下的坐标还原、背景差分抑制与独立运动小目标保留、无纹理场景的失败标记，以及 `k=2` 流式缓存和中心帧编号。

测试文件为 `tests/test_motion.py`。真实视频效果应查看对应运行目录中的快照、对比视频和诊断记录；本文档不预填尚未实际运行的视频指标，也不将合成测试结果外推为无人机检测精度。
