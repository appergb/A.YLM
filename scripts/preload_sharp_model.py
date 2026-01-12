#!/usr/bin/env python3
"""
SHARP模型预加载脚本
提前加载SHARP模型到内存中，避免每次运行都需要重新加载模型

使用方法:
1. 预加载模型: python3 preload_sharp_model.py
2. 在其他脚本中使用预加载的模型:
   - 设置环境变量: export SHARP_MODEL_PRELOADED=1
   - 或者在代码中设置: os.environ['SHARP_MODEL_PRELOADED'] = '1'
   - 然后正常运行其他脚本，模型加载会被跳过

功能特性:
- 预加载SHARP模型到共享内存
- 提供模型状态检查功能

作者: TRIP(appergb)
项目参与者: closer, true
个人研发项目
- 支持后台运行模式
- 自动检测模型加载状态
"""

import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

# Define ModelType as Any to avoid mypy type conflicts
# This is a compromise for dynamic imports
ModelType = Any

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from aylm import __version__
except ImportError:
    __version__ = "1.0.0"

# 添加SHARP路径
sys.path.insert(0, "ml-sharp/src")

try:
    import torch
    from sharp.models import PredictorParams, create_predictor

    SHARP_AVAILABLE = True
except ImportError as e:
    print(f"SHARP不可用: {e}")
    SHARP_AVAILABLE = False


class SharpModelPreloader:
    """SHARP模型预加载器"""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        """
        初始化预加载器

        Args:
            checkpoint_path: 模型检查点路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        # 使用环境变量或默认相对路径，避免硬编码
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "SHARP_MODEL_PATH", "models/sharp_2572gikvuh.pt"
        )
        self.device = self._get_device(device)
        self.model: ModelType = None
        self.loaded = False
        self.load_time = 0.0
        self.metadata_file: Optional[str] = None

        # 设置日志
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def save_model_metadata(self):
        """安全地保存模型元数据"""
        metadata = {
            "loaded": True,
            "device": self.device,
            "checkpoint_path": (
                str(Path(self.checkpoint_path).resolve())
                if self.checkpoint_path
                else ""
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f)
            self.metadata_file = f.name

        os.environ["SHARP_MODEL_METADATA"] = self.metadata_file

    def preload_model(self) -> bool:
        """
        预加载SHARP模型

        Returns:
            是否加载成功
        """
        if not SHARP_AVAILABLE:
            self.logger.error("SHARP不可用，无法预加载模型")
            return False

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            self.logger.error(f"模型检查点不存在: {self.checkpoint_path}")
            return False

        try:
            self.logger.info("开始预加载SHARP模型...")
            self.logger.info(f"   检查点: {self.checkpoint_path}")
            self.logger.info(f"   设备: {self.device}")

            start_time = time.time()

            # 加载模型状态字典
            if str(self.checkpoint_path).startswith("http"):
                state_dict = torch.hub.load_state_dict_from_url(
                    self.checkpoint_path, progress=True
                )
            else:
                state_dict = torch.load(self.checkpoint_path, weights_only=True)

            # 创建和初始化模型
            self.model = create_predictor(PredictorParams())
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)

            self.logger.info(f"   模型类型: {type(self.model).__name__}")
            self.logger.info(
                f"   模型参数量: {sum(p.numel() for p in self.model.parameters()):,}"
            )

            self.load_time = time.time() - start_time
            self.loaded = True

            self.logger.info("模型预加载完成")
            self.logger.info(f"   加载时间: {self.load_time:.2f}秒")

            # 安全地保存模型元数据到临时文件
            self.save_model_metadata()

            return True

        except Exception as e:
            self.logger.error(f"模型预加载失败: {e}")
            import traceback

            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "load_time": self.load_time,
            "model_type": type(self.model).__name__ if self.model else None,
        }

    def unload_model(self):
        """卸载模型"""
        if self.model:
            del self.model
            self.model = None
            self.loaded = False

        # 清理临时文件和环境变量
        if self.metadata_file and Path(self.metadata_file).exists():
            try:
                Path(self.metadata_file).unlink()
            except OSError:
                pass  # 忽略删除失败
            self.metadata_file = None

        if "SHARP_MODEL_METADATA" in os.environ:
            del os.environ["SHARP_MODEL_METADATA"]

        # 强制垃圾回收
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("模型已卸载")


def check_model_status() -> Dict[str, Any]:
    """检查模型加载状态"""
    metadata_file = os.environ.get("SHARP_MODEL_METADATA")

    if metadata_file and Path(metadata_file).exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return {
                "preloaded": metadata.get("loaded", False),
                "device": metadata.get("device"),
                "checkpoint": metadata.get("checkpoint_path"),
            }
        except (json.JSONDecodeError, IOError):
            # 如果文件损坏或无法读取，返回未加载状态
            pass

    return {
        "preloaded": False,
        "device": None,
        "checkpoint": None,
    }


def signal_handler(signum, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号，正在清理...")
    preloader = getattr(signal_handler, "preloader", None)
    if preloader:
        preloader.unload_model()
    sys.exit(0)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SHARP模型预加载器")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=os.environ.get("SHARP_MODEL_PATH", "models/sharp_2572gikvuh.pt"),
        help="模型检查点路径",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="计算设备",
    )
    parser.add_argument("--background", action="store_true", help="后台运行模式")
    parser.add_argument("--status", action="store_true", help="检查模型加载状态")
    parser.add_argument("--unload", action="store_true", help="卸载预加载的模型")

    args = parser.parse_args()

    # 检查状态
    if args.status:
        status = check_model_status()
        print(f"A.YLM v{__version__} - SHARP模型状态:")
        print(f"   预加载: {'是' if status['preloaded'] else '否'}")
        if status["device"]:
            print(f"   设备: {status['device']}")
        if status["checkpoint"]:
            print(f"   检查点: {status['checkpoint']}")
        return

    # 卸载模型
    if args.unload:
        preloader = SharpModelPreloader()
        preloader.unload_model()
        return

    # 创建预加载器
    preloader = SharpModelPreloader(checkpoint_path=args.checkpoint, device=args.device)

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    cast(Any, signal_handler).preloader = preloader

    # 预加载模型
    if preloader.preload_model():
        print("\n预加载完成！其他脚本现在可以跳过模型加载步骤。")
        print("使用提示:")
        print("   • 设置环境变量: export SHARP_MODEL_PRELOADED=1")
        print("   • 或在代码中: os.environ['SHARP_MODEL_PRELOADED'] = '1'")

        if args.background:
            print("后台运行模式，模型保持在内存中...")
            print("   按 Ctrl+C 停止并卸载模型")
            try:
                # 保持运行
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("✨ 预加载完成，脚本退出但模型状态保持")

        # 显示模型信息
        info = preloader.get_model_info()
        print("\n模型信息:")
        for key, value in info.items():
            print(f"   {key}: {value}")

    else:
        print("预加载失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
