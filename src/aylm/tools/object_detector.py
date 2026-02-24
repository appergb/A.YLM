"""YOLO11 目标检测模块。

提供基于 YOLO11 的目标检测功能，支持实例分割和边界框检测。
自动检测 CUDA/MPS/CPU 设备，支持半精度推理加速。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .semantic_types import COCO_TO_SEMANTIC, Detection2D, SemanticLabel

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """检测器配置参数。"""

    model_name: str = "yolo11n-seg.pt"  # YOLO11 模型名称
    confidence_threshold: float = 0.25  # 置信度阈值（降低以检测更多目标）
    iou_threshold: float = 0.45  # NMS IoU 阈值
    device: str = "auto"  # 设备：auto/cuda/mps/cpu
    half_precision: bool = True  # 是否使用半精度（FP16）
    classes: list[int] = field(default_factory=list)  # 检测的类别，空列表表示使用默认


class ObjectDetector:
    """YOLO11 目标检测器。

    支持实例分割和边界框检测，自动检测最佳计算设备。

    使用示例:
        >>> config = DetectorConfig(confidence_threshold=0.6)
        >>> with ObjectDetector(config) as detector:
        ...     detections = detector.detect(image)
    """

    # 默认检测类别：人、自行车、汽车、摩托车、公交车、卡车
    DEFAULT_CLASSES: ClassVar[list[int]] = [0, 1, 2, 3, 5, 7]

    def __init__(self, config: Optional[DetectorConfig] = None):
        """初始化检测器。

        Args:
            config: 检测器配置，为 None 时使用默认配置
        """
        self.config = config or DetectorConfig()
        self._model = None
        self._device = self._detect_device()

        # 如果未指定类别，使用默认类别
        if not self.config.classes:
            self.config.classes = self.DEFAULT_CLASSES.copy()

        logger.info(f"ObjectDetector 初始化，设备: {self._device}")

    def _detect_device(self) -> str:
        """自动检测最佳计算设备。

        Returns:
            设备名称：cuda/mps/cpu
        """
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到 CUDA GPU: {device_name}")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("检测到 Apple MPS GPU")
                return "mps"
            else:
                logger.info("未检测到 GPU，使用 CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch 未安装，使用 CPU")
            return "cpu"

    def load(self) -> None:
        """加载 YOLO11 模型。

        Raises:
            ImportError: ultralytics 未安装
            RuntimeError: 模型加载失败
        """
        if self._model is not None:
            logger.debug("模型已加载，跳过")
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 未安装，请运行: pip install ultralytics"
            ) from e

        logger.info(f"加载 YOLO11 模型: {self.config.model_name}")

        try:
            self._model = YOLO(self.config.model_name)

            # 设置设备和精度
            if self._device != "cpu" and self.config.half_precision:
                logger.info(f"启用半精度推理 (FP16) on {self._device}")

            logger.info("YOLO11 模型加载完成")

        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}") from e

    def unload(self) -> None:
        """卸载模型，释放内存。"""
        if self._model is None:
            return

        logger.info("卸载 YOLO11 模型")
        del self._model
        self._model = None

        # 清理 GPU 内存
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS 没有显式的缓存清理方法
                pass
        except ImportError:
            pass

        logger.info("模型已卸载，内存已释放")

    def detect(
        self,
        image: NDArray[np.uint8],
        return_masks: bool = True,
    ) -> list[Detection2D]:
        """执行目标检测。

        Args:
            image: 输入图像，BGR 格式，shape (H, W, 3)
            return_masks: 是否返回实例分割掩码

        Returns:
            检测结果列表

        Raises:
            RuntimeError: 模型未加载
        """
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        # 执行推理
        results = self._model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            device=self._device,
            half=self.config.half_precision and self._device != "cpu",
            verbose=False,
        )

        detections: list[Detection2D] = []

        for result in results:
            boxes = result.boxes
            masks = result.masks if return_masks and hasattr(result, "masks") else None

            if boxes is None:
                continue

            # 获取原图尺寸（用于 mask 缩放）
            orig_shape = result.orig_shape if hasattr(result, "orig_shape") else None

            for i, box in enumerate(boxes):
                # 获取边界框
                bbox = box.xyxy[0].cpu().numpy().astype(np.float32)

                # 获取类别和置信度
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # 获取掩码（缩放到原图尺寸）
                mask = None
                if masks is not None and i < len(masks):
                    # 获取原始 mask（可能是低分辨率的）
                    mask_data = masks[i].data[0].cpu().numpy()

                    # 如果 mask 尺寸与原图不同，缩放到原图尺寸
                    if orig_shape is not None and mask_data.shape != orig_shape:
                        import cv2

                        # 使用双线性插值缩放，然后二值化
                        mask_resized = cv2.resize(
                            mask_data.astype(np.float32),
                            (orig_shape[1], orig_shape[0]),  # (width, height)
                            interpolation=cv2.INTER_LINEAR,
                        )
                        mask = (mask_resized > 0.5).astype(np.bool_)
                    else:
                        mask = mask_data.astype(np.bool_)

                # 映射到语义标签
                semantic_label = COCO_TO_SEMANTIC.get(class_id, SemanticLabel.OBSTACLE)

                detection = Detection2D(
                    bbox=bbox,
                    mask=mask,
                    class_id=class_id,
                    confidence=confidence,
                    semantic_label=semantic_label,
                )
                detections.append(detection)

        logger.debug(f"检测到 {len(detections)} 个目标")
        return detections

    def detect_file(
        self,
        image_path: Union[str, Path],
        return_masks: bool = True,
    ) -> list[Detection2D]:
        """从文件路径执行目标检测。

        Args:
            image_path: 图像文件路径
            return_masks: 是否返回实例分割掩码

        Returns:
            检测结果列表

        Raises:
            FileNotFoundError: 图像文件不存在
            RuntimeError: 模型未加载
        """
        import cv2

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        logger.debug(f"读取图像: {image_path}")
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        return self.detect(image, return_masks=return_masks)

    def __enter__(self) -> "ObjectDetector":
        """上下文管理器入口，自动加载模型。"""
        self.load()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """上下文管理器出口，自动卸载模型。"""
        self.unload()

    def save_detection_image(
        self,
        image: NDArray[np.uint8],
        detections: list[Detection2D],
        output_path: Union[str, Path],
        draw_masks: bool = True,
    ) -> Path:
        """保存带检测框的可视化图片。

        Args:
            image: 原始图像，BGR 格式
            detections: 检测结果列表
            output_path: 输出图片路径
            draw_masks: 是否绘制分割掩码

        Returns:
            保存的图片路径
        """
        import cv2

        output_path = Path(output_path)
        result_image = image.copy()

        # 类别颜色映射 (BGR)
        label_colors = {
            SemanticLabel.PERSON: (0, 0, 255),  # 红色
            SemanticLabel.VEHICLE: (255, 100, 0),  # 蓝色
            SemanticLabel.BICYCLE: (0, 165, 255),  # 橙色
            SemanticLabel.ANIMAL: (0, 255, 255),  # 黄色
            SemanticLabel.OBSTACLE: (255, 0, 255),  # 紫色
        }

        # 类别名称映射
        label_names = {
            SemanticLabel.PERSON: "Person",
            SemanticLabel.VEHICLE: "Vehicle",
            SemanticLabel.BICYCLE: "Bicycle",
            SemanticLabel.ANIMAL: "Animal",
            SemanticLabel.OBSTACLE: "Obstacle",
        }

        for det in detections:
            color = label_colors.get(det.semantic_label, (128, 128, 128))
            label_name = label_names.get(det.semantic_label, "Unknown")

            # 绘制边界框
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label_text = f"{label_name} {det.confidence:.2f}"
            (text_w, text_h), _baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result_image,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 绘制分割掩码
            if draw_masks and det.mask is not None:
                mask_resized = cv2.resize(
                    det.mask.astype(np.uint8) * 255,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_colored = np.zeros_like(result_image)
                mask_colored[mask_resized > 0] = color
                result_image = cv2.addWeighted(result_image, 1.0, mask_colored, 0.3, 0)

        # 保存图片
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        logger.info(f"检测结果图片已保存: {output_path}")

        return output_path
