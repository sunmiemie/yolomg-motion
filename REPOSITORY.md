# 仓库与本地资源

`main` 保存已有的 YOLOMG 运动特征、自动检测跟踪以及 CPU / AMD GPU 加速版本。功能分支用于新的 ROI 跟踪实验。

仓库包含源码、测试、说明和固定版本的作者源码。原始视频、测试集、运行结果、Python 环境、ONNX 缓存及模型权重属于本地资源，不提交 Git。

## 从源码运行

```powershell
python -m pip install -e ".[test]"
python -m pip install -r requirements-detection.txt
python tools/fetch_weights.py
python -m pytest -q
python -m yolomg_motion.track_cli --input 'input.mp4' --output 'runs/demo' --lock-confidence 0.10 --max-detector-gap 90
```

模型下载脚本固定作者提交与 SHA-256；已有权重先校验，不覆盖不同内容。AMD 环境安装与运行见 [GPU_ACCELERATION.md](GPU_ACCELERATION.md)。源码安装命令需要在项目目录执行；输入视频由使用者提供。

`vendor/YOLOMG` 来源于 [Irisky123/YOLOMG](https://github.com/Irisky123/YOLOMG)，固定提交 `090a74cd3ece15c66e857cc2e08e01cb103c4550`。原始文件的来源和摘要保存在 `vendor/YOLOMG/SOURCE.json`，作者代码保留 [GPL-3.0 许可证](vendor/YOLOMG/LICENSE) 与原有版权说明。本项目增加的跟踪与加速实现不属于论文作者原始功能。文档中的本机测试结果不能视为嵌入式板卡性能承诺。
