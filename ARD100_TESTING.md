# ARD100 作者测试划分验证

## 本次范围

先测试作者固定 35 段测试序列中的第一段 `phantom02`。测试整个视频，不根据预测好坏挑选片段；不是完整 35 段评测，也不将本次统计称为论文 AP。

视频原始尺寸 1920×1080，实际 FPS 为 30000/1001（约 29.970），1799 帧。XML 标注编号 0001–1799 连续，均为一个 Drone、difficult=0，无缺失。模型与跟踪保持之前的视频参数：官方 1280 权重，候选和锁定阈值 0.10、每 3 帧检测一次、最大无有效检测间隔 90 帧。

2026-09-11 已完成本序列：1795 个有效帧中，1446 帧正确锁定（IoU≥0.5，80.56%），291 帧错框，58 帧未锁定，平均 IoU 0.64345。599 个实际检测帧上 TP/FP/FN 为 569/192/30，固定阈值 Precision 74.77%、Recall 94.99%。独立复算一致，存在明显模板漂移，不能称为全程稳定锁定。此目录现只保留 `tracked_zoom.mp4` 与 `tracks.json`；原评测汇总保存在轨迹 JSON 的 `metadata.ground_truth_evaluation` 中，其余 34 段尚未运行。

## 数据来源与校验

作者公开入口：[YOLOMG GitHub](https://github.com/Irisky123/YOLOMG)，README 指向百度网盘。

本次数据取自公开的 [Zenodo ARD100 镜像](https://zenodo.org/records/15870538)，上传者为 Yu-Hsi Chen，不能将镜像上传者说成论文作者。完整外层 ZIP 大小 27,351,535,415 字节，发布页给出的 MD5 为 `b52a5c8aae7b7661b57dde5490c56580`。为避免下载无关训练数据，通过标准 HTTP Range 仅读取需要的压缩包成员。

已下载并通过 ZIP 成员 CRC 校验：

- `ARD100/annotations.zip`：104,543,325 字节，CRC32 `8072ce9a`。
- `ARD100/test_videos/phantom02.mp4`：238,451,079 字节，CRC32 `8d3a3b06`。

已用 AST 读取官方 `generate_dataset.py` 中的训练/测试列表，镜像内 65 个训练视频和 35 个测试视频的名单逐项一致。记录在 `datasets/ard100_public/split_verification.json`；完整成员信息和来源在 `archive_manifest.json`。仅下载部分成员，未计算完整外层 ZIP MD5，也未对照百度网盘逐字节验证。

## 运行命令

在 `D:/Users/Desktop/bxk/small/yolomg_motion` 中：

```powershell
python -m yolomg_motion.track_cli --input datasets/ard100_public/ARD100/test_videos/phantom02.mp4 --output runs/ard100_phantom02_20260911 --image-size 1280 --confidence 0.10 --lock-confidence 0.10 --detect-every 3 --max-detector-gap 90 --threads 4
```

输出目录必须不存在或为空，重复运行请换目录。读取原视频重新计算三帧差分和检测，不使用真实框初始化跟踪。正常跟踪只生成 `tracked_zoom.mp4` 与 `tracks.json`，不自动生成以下诊断材料。

## 可选标注诊断

仅在需要额外标注诊断时手动执行以下命令；它们会另外创建对照视频、截图和评测报告。当前交付目录中的这些诊断文件已按精简要求清理。

生成标注对齐预览并目视核对：

```powershell
python tools/compare_ard100_ground_truth.py --run runs/ard100_phantom02_20260911 --annotations datasets/ard100_public/ARD100/annotations.zip --sequence phantom02 --preview-only
```

绿色框与实际目标对齐后，生成整段对照视频和指标：

```powershell
python tools/compare_ard100_ground_truth.py --run runs/ard100_phantom02_20260911 --annotations datasets/ard100_public/ARD100/annotations.zip --sequence phantom02 --alignment-verified
```

输出在运行目录的 `ground_truth_comparison`。绿色框为作者标注，黄色框为当前 LOCKED 预测；放大窗以真值为中心，仅供显示。GT 绝不进入检测或跟踪，也不用于挑选预测。文件编号用 `frame_index+1`；坐标按作者 VOC 转换惯例将四个 xyxy 值各减 1，并在报告中记录。

## 指标含义

- **正确锁定帧**：处于 LOCKED 且预测与真值 IoU≥0.5。
- **错框帧**：处于 LOCKED，但与真值未达到 IoU≥0.5；包括定位偏差和跟错背景，不能一律称为背景误锁。
- **漏跟帧**：存在真值目标，却没有 LOCKED 框。
- **平均 IoU**：所有 GT 阳性帧的 IoU 均值，未锁定帧按 0 计。
- **固定阈值检测 Precision/Recall**：仅在实际运行网络的帧上，按置信度≥0.10、IoU≥0.5 做一对一匹配，保留 TP、FP、FN。模板续跟踪框不混入网络检测统计。

缺失 XML 单独报告，不能当作无目标帧。本序列标注全部为 difficult=0；工具忽略 difficult 对象的策略不应直接扩展为含困难目标数据集的标准 AP 评测。

## 与论文整体评测的边界

论文 AP 需要完整固定测试清单、官方评测预处理和低候选阈值（例如 0.001）。本次用现有跟踪管线做单序列实测，因此固定阈值 Precision/Recall 不等于 AP50，更不等于整个 ARD100 测试集的成绩。

公开脚本还存在 mask22/mask32 路径与起始帧编号差异、面积筛选等口径问题；当前运动模块修复了 uint8 加法溢出，并增加有效边界及配准失败检查。数据包包含原视频与 XML，未包含作者最终 RGB/mask32 评测清单。要严格复现论文 AP，需另行固定这些预处理和帧筛选协议。
