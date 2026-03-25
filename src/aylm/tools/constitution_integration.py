"""宪法评估流水线集成层。

封装 EgoState 构建、AIDecision 默认生成、障碍物适配、评估调用和
结果序列化的公共逻辑，供图像流水线和视频流水线复用。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ConstitutionIntegration:
    """宪法评估集成辅助类。

    供 PipelineProcessor 和 VideoPipelineProcessor 复用。

    Example:
        >>> from aylm.tools.constitution_integration import ConstitutionIntegration
        >>> integration = ConstitutionIntegration(config=None, ego_speed=10.0)
        >>> result = integration.evaluate_frame(obstacles_data, frame_id=0)
    """

    def __init__(
        self,
        config: Any | None = None,
        ego_speed: float = 0.0,
        ego_heading: float = 0.0,
    ):
        """初始化宪法评估集成。

        Args:
            config: ConstitutionConfig 实例，None 则使用默认配置
            ego_speed: 自车速度 m/s
            ego_heading: 自车航向（弧度）
        """
        self._evaluator = None
        self.ego_speed = ego_speed
        self.ego_heading = ego_heading
        self._init_evaluator(config)

    def _init_evaluator(self, config: Any | None) -> None:
        """延迟导入并初始化评估器。"""
        try:
            from aylm.constitution import ConstitutionConfig, ConstitutionEvaluator

            if config is None:
                config = ConstitutionConfig()
            self._evaluator = ConstitutionEvaluator(config)
            logger.info(
                "宪法评估器已初始化，活跃原则: %s",
                self._evaluator.active_principles,
            )
        except Exception as e:
            logger.warning("宪法评估器初始化失败: %s", e)
            self._evaluator = None

    @property
    def is_available(self) -> bool:
        """评估器是否可用。"""
        return self._evaluator is not None

    def evaluate_frame(
        self,
        obstacles_data: list[dict[str, Any]],
        frame_id: int = 0,
        timestamp: float = 0.0,
        decision: Any | None = None,
    ) -> dict[str, Any] | None:
        """执行单帧宪法评估。

        Args:
            obstacles_data: 流水线输出的障碍物字典列表
            frame_id: 帧 ID
            timestamp: 时间戳（秒）
            decision: 外部注入的 AIDecision（可选，None 则使用默认直行）

        Returns:
            评估结果字典，失败返回 None
        """
        if self._evaluator is None:
            return None

        try:
            from aylm.constitution import (
                AIDecision,
                ConstitutionObstacle,
                EgoState,
                SceneState,
                TrajectoryPoint,
            )

            # 1. 构建 EgoState（机器人坐标系：X 前、Y 左、Z 上）
            ego_state = EgoState(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                velocity=np.array(
                    [
                        self.ego_speed * math.cos(self.ego_heading),
                        self.ego_speed * math.sin(self.ego_heading),
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                heading=self.ego_heading,
                speed=self.ego_speed,
            )

            # 2. 适配障碍物数据
            constitution_obstacles = []
            for obs_dict in obstacles_data:
                if "center_robot" not in obs_dict:
                    continue
                try:
                    obs = ConstitutionObstacle.from_obstacle_dict(obs_dict)
                    constitution_obstacles.append(obs)
                except Exception as e:
                    logger.debug("障碍物适配失败: %s", e)

            # 3. 构建 SceneState
            scene = SceneState(
                frame_id=frame_id,
                timestamp=timestamp,
                ego_state=ego_state,
                obstacles=constitution_obstacles,
            )

            # 4. 构建 AIDecision
            if decision is not None:
                # 使用外部注入的决策
                ai_decision = decision
            else:
                # 构建默认 AIDecision（保持当前速度直行）
                trajectory = [
                    TrajectoryPoint(
                        position=np.array(
                            [
                                self.ego_speed * t * math.cos(self.ego_heading),
                                self.ego_speed * t * math.sin(self.ego_heading),
                                0.0,
                            ],
                            dtype=np.float32,
                        ),
                        timestamp=t,
                    )
                    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                ]
                ai_decision = AIDecision(
                    decision_type="trajectory",
                    trajectory=trajectory,
                    target_speed=self.ego_speed,
                    metadata={"source": "default_keep_straight"},
                )

            # 5. 执行评估
            result = self._evaluator.evaluate(scene, ai_decision)
            return result.to_dict()

        except Exception as e:
            logger.warning("宪法评估执行失败: %s", e)
            return None

    @staticmethod
    def load_config(config_path: str | Path | None):
        """从文件路径加载宪法配置。

        Args:
            config_path: YAML 或 JSON 配置文件路径，None 返回 None

        Returns:
            ConstitutionConfig 实例或 None
        """
        if config_path is None:
            return None

        path = Path(config_path)
        if not path.exists():
            logger.warning("宪法配置文件不存在: %s", path)
            return None

        try:
            from aylm.constitution import ConstitutionConfig

            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                return ConstitutionConfig.from_yaml(path)
            elif suffix == ".json":
                return ConstitutionConfig.from_json(path)
            else:
                logger.warning("不支持的配置文件格式: %s", suffix)
                return None
        except Exception as e:
            logger.warning("加载宪法配置失败: %s", e)
            return None
