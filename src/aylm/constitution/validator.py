"""指令验证器模块。

宪法评估的顶层开放接口，整合指令解析和安全评估。
外部系统（LLM / 端到端模型 / ROS / HTTP 客户端）通过此接口验证指令安全性。

Example:
    >>> from aylm.constitution import CommandValidator
    >>> validator = CommandValidator()
    >>>
    >>> # JSON 轨迹验证
    >>> result = validator.validate(
    ...     command={"type": "trajectory", "points": [[5,0,0,0.5]]},
    ...     scene=scene_state,
    ... )
    >>> print(result.approved)  # True/False
    >>>
    >>> # 自然语言验证
    >>> result = validator.validate(
    ...     command="向左转弯30度",
    ...     ego_speed=10.0,
    ...     obstacles=[{"center_robot": [5,2,0], ...}],
    ... )
    >>> if not result.approved:
    ...     print(f"否决: {result.reason}")
    ...     print(f"安全替代: {result.alternative_decision}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import ConstitutionConfig
from .evaluator import ConstitutionEvaluator
from .registry import ConstitutionRegistry
from .types import AIDecision, EgoState, SceneState, TrajectoryPoint

if TYPE_CHECKING:
    from .command_parser import CommandParser

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """指令验证结果。

    Attributes:
        approved: 是否批准执行
        safety_score: 安全分 0-1
        recommended_action: 推荐动作
        reason: 人类可读原因
        violations: 违规详情列表
        original_decision: 原始指令解析结果
        alternative_decision: 否决时的安全替代方案
        evaluation_detail: 完整评估结果
    """

    approved: bool
    safety_score: float
    recommended_action: str
    reason: str
    violations: list[dict[str, Any]] = field(default_factory=list)
    original_decision: dict[str, Any] = field(default_factory=dict)
    alternative_decision: dict[str, Any] | None = None
    evaluation_detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "approved": self.approved,
            "safety_score": self.safety_score,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
            "violations": self.violations,
            "original_decision": self.original_decision,
            "alternative_decision": self.alternative_decision,
            "evaluation_detail": self.evaluation_detail,
        }


class CommandValidator:
    """指令验证器 — 宪法评估的顶层开放接口。

    整合指令解析（CommandParser）和安全评估（ConstitutionEvaluator），
    提供一站式的指令验证服务。

    Example:
        >>> from aylm.constitution import CommandValidator
        >>> validator = CommandValidator()
        >>>
        >>> # 验证 JSON 轨迹
        >>> result = validator.validate(
        ...     command={"type": "trajectory", "points": [[5,0,0,0.5]]},
        ...     ego_speed=10.0,
        ... )
        >>> print(result.approved)
        >>>
        >>> # 验证自然语言
        >>> result = validator.validate(
        ...     command="向左转弯30度",
        ...     ego_speed=10.0,
        ...     obstacles=[{
        ...         "center_robot": [5, 2, 0],
        ...         "dimensions_robot": [1, 1, 1],
        ...         "_label": "VEHICLE",
        ...         "confidence": 0.9,
        ...     }],
        ... )
        >>> if not result.approved:
        ...     print(f"否决: {result.reason}")

    Args:
        config: 宪法配置，None 则使用默认
        approval_threshold: 批准阈值（安全分 >= 此值则批准）
        parsers: 自定义解析器列表，None 则使用注册的解析器
    """

    def __init__(
        self,
        config: ConstitutionConfig | None = None,
        approval_threshold: float = 0.6,
        parsers: list[CommandParser] | None = None,
    ):
        self.config = config or ConstitutionConfig()
        self.approval_threshold = approval_threshold
        self._evaluator = ConstitutionEvaluator(self.config)

        # 初始化解析器
        if parsers is not None:
            self._parsers = parsers
        else:
            self._parsers = self._init_default_parsers()

    def _init_default_parsers(self) -> list[CommandParser]:
        """从注册表初始化默认解析器。"""
        parsers = []
        for name in ConstitutionRegistry.list_command_parsers():
            cls = ConstitutionRegistry.get_command_parser(name)
            if cls is not None:
                try:
                    parsers.append(cls())
                except Exception as e:
                    logger.warning("初始化解析器 '%s' 失败: %s", name, e)
        return parsers

    def validate(
        self,
        command: str | dict,
        scene: SceneState | None = None,
        ego_speed: float = 0.0,
        ego_heading: float = 0.0,
        obstacles: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        """验证指令是否安全。

        Args:
            command: 外部指令（字符串或字典）
            scene: 完整场景状态（优先使用）
            ego_speed: 自车速度 m/s（简便参数）
            ego_heading: 自车航向弧度（简便参数）
            obstacles: 障碍物字典列表（简便参数）

        Returns:
            ValidationResult: 验证结果
        """
        # 1. 解析指令
        try:
            decision = self._parse_command(
                command, ego_speed=ego_speed, ego_heading=ego_heading
            )
        except ValueError as e:
            return ValidationResult(
                approved=False,
                safety_score=0.0,
                recommended_action="error",
                reason=f"指令解析失败: {e}",
            )

        # 2. 构建场景
        if scene is None:
            scene = self._build_scene(
                ego_speed=ego_speed,
                ego_heading=ego_heading,
                obstacles=obstacles or [],
            )

        # 3. 评估
        try:
            eval_result = self._evaluator.evaluate(scene, decision)
        except Exception as e:
            logger.error("宪法评估异常: %s", e)
            return ValidationResult(
                approved=False,
                safety_score=0.0,
                recommended_action="error",
                reason=f"评估异常: {e}",
                original_decision=decision.to_dict(),
            )

        # 4. 判定（分数阈值 + 推荐动作双重检查）
        score = eval_result.safety_score.overall
        action = eval_result.safety_score.recommended_action.value
        approved = score >= self.approval_threshold and action not in (
            "emergency_stop",
            "intervention",
        )

        # 5. 否决时生成安全替代
        alternative = None
        if not approved:
            alternative = self._generate_alternative(scene, decision, eval_result)

        # 6. 构建违规列表
        violations = [v.to_dict() for v in eval_result.violations]

        # 7. 生成原因
        if approved:
            reason = f"指令安全，安全分 {score:.2f}"
        else:
            violated_names = eval_result.safety_score.violations
            reason = (
                f"指令被否决，安全分 {score:.2f} < 阈值 {self.approval_threshold}，"
                f"违规原则: {', '.join(violated_names)}"
            )

        return ValidationResult(
            approved=approved,
            safety_score=score,
            recommended_action=action,
            reason=reason,
            violations=violations,
            original_decision=decision.to_dict(),
            alternative_decision=alternative,
            evaluation_detail=eval_result.to_dict(),
        )

    def _parse_command(self, command: str | dict, **context: Any) -> AIDecision:
        """自动选择合适的解析器并解析指令。"""
        # 如果已经是 AIDecision，直接返回
        if isinstance(command, AIDecision):
            return command

        for parser in self._parsers:
            if parser.can_parse(command):
                return parser.parse(command, **context)

        raise ValueError(
            f"无法解析指令（类型: {type(command).__name__}），"
            f"已注册解析器: {[type(p).__name__ for p in self._parsers]}"
        )

    @staticmethod
    def _build_scene(
        ego_speed: float,
        ego_heading: float,
        obstacles: list[dict[str, Any]],
    ) -> SceneState:
        """从简便参数构建场景。"""
        from .adapter import ConstitutionObstacle

        ego_state = EgoState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            velocity=np.array(
                [
                    ego_speed * math.cos(ego_heading),
                    ego_speed * math.sin(ego_heading),
                    0.0,
                ],
                dtype=np.float32,
            ),
            heading=float(ego_heading),
            speed=float(ego_speed),
        )

        constitution_obstacles = []
        for obs_dict in obstacles:
            if "center_robot" not in obs_dict:
                continue
            try:
                obs = ConstitutionObstacle.from_obstacle_dict(obs_dict)
                constitution_obstacles.append(obs)
            except Exception as e:
                logger.debug("障碍物适配失败: %s", e)

        return SceneState(
            frame_id=0,
            timestamp=0.0,
            ego_state=ego_state,
            obstacles=constitution_obstacles,
        )

    @staticmethod
    def _generate_alternative(
        scene: SceneState,
        original: AIDecision,
        eval_result: Any,
    ) -> dict[str, Any] | None:
        """否决时生成安全替代方案（减速/停车）。"""
        ego_speed = scene.ego_state.speed
        ego_heading = float(scene.ego_state.heading)

        # 安全策略：减速到当前速度的 30% 或停车
        action = eval_result.safety_score.recommended_action.value
        if action == "emergency_stop":
            target_speed = 0.0
            desc = "紧急停车"
        else:
            target_speed = ego_speed * 0.3
            desc = f"减速至 {target_speed:.1f} m/s"

        trajectory = [
            TrajectoryPoint(
                position=np.array(
                    [
                        target_speed * t * math.cos(ego_heading),
                        target_speed * t * math.sin(ego_heading),
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                timestamp=float(t),
            )
            for t in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
        ]

        alternative = AIDecision(
            decision_type="trajectory",
            trajectory=trajectory,
            target_speed=target_speed,
            metadata={"source": "safety_alternative", "description": desc},
        )
        return alternative.to_dict()
