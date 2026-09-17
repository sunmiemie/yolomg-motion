"""CPU/CUDA inference adapter for the original two-input YOLOMG checkpoint.

The vendor model is loaded as published, including its custom fusion modules.
Its original demo's fixed CUDA device and fixed 1920 x 1080 output geometry
are intentionally replaced with the requested device and actual frame size.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int


class YOLOMGDetector:
    """Run published YOLOMG weights on a BGR frame and its raw motion map.

    ``motion_float32`` must be a same-size grayscale map on the absolute
    0..255 scale. Preview heatmaps and contrast-normalized images are not valid
    model inputs. The checkpoint must come from a trusted source: the original
    release stores a pickled model, requiring ``weights_only=False`` on Torch
    2.6 and newer.
    """

    def __init__(
        self,
        repo_dir,
        weights,
        image_size=1280,
        device="cpu",
        conf_threshold=0.05,
        iou_threshold=0.45,
        threads=4,
    ):
        import torch

        self.repo_dir = Path(repo_dir).resolve()
        self.weights = Path(weights).resolve()
        if not (self.repo_dir / "models" / "yolo.py").is_file():
            raise FileNotFoundError(f"Official YOLOMG source not found: {self.repo_dir}")
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLOMG checkpoint not found: {self.weights}")
        if not 0 < conf_threshold <= 1 or not 0 < iou_threshold <= 1:
            raise ValueError("Confidence and IoU thresholds must be in (0, 1]")
        if image_size < 32 or threads < 1:
            raise ValueError("image_size must be >= 32 and threads must be positive")

        # Original checkpoints reference the top-level modules models.* and
        # utils.*. Refuse to silently bind those names to another YOLO fork.
        for prefix in ("models", "utils"):
            for name, module in list(sys.modules.items()):
                if name == prefix or name.startswith(prefix + "."):
                    module_path = getattr(module, "__file__", None)
                    if module_path:
                        try:
                            Path(module_path).resolve().relative_to(self.repo_dir)
                        except ValueError as error:
                            raise RuntimeError(
                                f"Conflicting imported module {name}: {module_path}; "
                                "load YOLOMG in a fresh process"
                            ) from error
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        # The old plots module eagerly downloads Arial during model imports.
        # Select an already installed font before importing it; inference does
        # not need plotting, and a network font download should not be required.
        from PIL import ImageFont
        from utils import general as vendor_general
        for font_name in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                font_path = ImageFont.truetype(font_name, 10).path
                vendor_general.FONT = str(font_path)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("An installed Arial or DejaVuSans font is required by vendor imports")

        from models.common import Conv
        from models.yolo import Detect, Model
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords

        self._torch = torch
        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale_coords = scale_coords
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this Torch runtime has no available CUDA device")
        torch.set_num_threads(int(threads))
        # This mirrors attempt_load without altering vendor source or globally
        # monkey-patching torch.load. No automatic model download is performed.
        checkpoint = torch.load(str(self.weights), map_location=self.device, weights_only=False)
        model = (checkpoint.get("ema") or checkpoint["model"]).float()
        model = model.fuse().eval()
        for module in model.modules():
            if isinstance(module, (Detect, Model)):
                module.inplace = True
                if isinstance(module, Detect) and not isinstance(module.anchor_grid, list):
                    delattr(module, "anchor_grid")
                    module.anchor_grid = [torch.zeros(1, device=self.device)] * module.nl
            elif isinstance(module, Conv):
                module._non_persistent_buffers_set = set()
            elif isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
                module.recompute_scale_factor = None
        self.model = model.to(self.device)
        self.stride = int(self.model.stride.max())
        self.image_size = int(check_img_size(int(image_size), s=self.stride))
        self.names = self.model.names
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.last_inference_seconds = 0.0
        self.last_input_shape = None

    def _tensor(self, image):
        resized, ratio, pad = self._letterbox(image, self.image_size, stride=self.stride)
        chw_rgb = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))
        tensor = self._torch.from_numpy(chw_rgb).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        return tensor.unsqueeze(0) / 255.0, ratio, pad

    def _predict(self, rgb, motion):
        """Return decoded raw predictions before the shared CPU/GPU NMS."""
        return self.model(rgb, motion, augment=False)[0]

    def detect(self, frame_bgr, motion_float32):
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be a uint8 H x W x 3 BGR image")
        if motion_float32.ndim != 2 or motion_float32.shape != frame_bgr.shape[:2]:
            raise ValueError("Motion map must be grayscale with the same H x W as the frame")
        if not np.isfinite(motion_float32).all():
            raise ValueError("Motion map contains non-finite values")
        if np.any(motion_float32 < 0) or np.any(motion_float32 > 255):
            raise ValueError("Motion map must retain the raw 0..255 intensity scale")

        began = perf_counter()
        # The official training masks are grayscale JPEGs loaded by cv2.imread
        # as three identical channels. Quantize to uint8, without preview gain,
        # while avoiding another lossy JPEG encode/decode cycle.
        gray = np.rint(motion_float32).clip(0, 255).astype(np.uint8)
        motion_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb, ratio, pad = self._tensor(frame_bgr)
        motion, _, _ = self._tensor(motion_bgr)
        self.last_input_shape = tuple(rgb.shape)
        with self._torch.inference_mode():
            prediction = self._predict(rgb, motion)
            detections = self._nms(
                prediction, self.conf_threshold, self.iou_threshold,
                classes=None, agnostic=False,
            )[0]
            if len(detections):
                detections[:, :4] = self._scale_coords(
                    rgb.shape[2:], detections[:, :4], frame_bgr.shape[:2],
                    ratio_pad=(ratio, pad),
                )
            values = detections.detach().cpu().numpy()
        self.last_inference_seconds = perf_counter() - began
        return [
            Detection(tuple(float(v) for v in row[:4]), float(row[4]), int(row[5]))
            for row in values
            if np.isfinite(row).all() and row[2] > row[0] and row[3] > row[1]
        ]
