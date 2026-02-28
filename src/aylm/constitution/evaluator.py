"""宪法评估编排器。

串联原则评估 → 打分 → 训练信号生成的完整流程。
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import ViolationResult
from .config import ConstitutionConfig
from .default_generator import DefaultTrainingSignalGenerator
from .registry import ConstitutionRegistry
from .scorer import SafetyScore
from .training import TrainingSignal
from .types import AIDecision, SceneState
from .weighted_scorer import WeightedSafetyScorer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """宪法评估结果。

    包含完整的评估信息：安全分数、违规详情、训练信号。

    Attributes:
        safety_score: 综合安全评分
        violations: 违规结果列表（包含所有原则的评估）
        training_signal: 训练信号（可选）
        principle_results: 各原则评估结果映射
    """

    safety_score: SafetyScore
    violations: list[ViolationResult] = field(default_factory=list)
    training_signal: TrainingSignal | None = None
    principle_results: dict[str, ViolationResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "safety_score": self.safety_score.to_dict(),
            "violations": [v.to_dict() for v in self.violations if v.violated],
            "training_signal": (
                self.training_signal.to_dict() if self.training_signal else None
            ),
            "principle_results": {
                name: result.to_dict()
                for name, result in self.principle_results.items()
            },
        }


class ConstitutionEvaluator:
    """宪法评估编排器。

    串联配置中启用的原则，依次执行：
    1. 从 Registry 实例化原则并评估
    2. WeightedSafetyScorer 聚合打分
    3. DefaultTrainingSignalGenerator 生成训练信号

    Example:
        >>> from aylm.constitution import ConstitutionEvaluator, ConstitutionConfig
        >>> evaluator = ConstitutionEvaluator(ConstitutionConfig())
        >>> result = evaluator.evaluate(scene_state, ai_decision)
        >>> print(result.safety_score.overall)
    """

    def __init__(self, config: ConstitutionConfig | None = None):
        self.config = config or ConstitutionConfig()
        self._scorer = WeightedSafetyScorer(config=self.config)
        self._generator = DefaultTrainingSignalGenerator(
            generate_positive=self.config.generate_positive_signals,
        )
        # Cache principle instances
        self._principles: dict[str, Any] = {}
        self._init_principles()

    def _init_principles(self) -> None:
        """从配置初始化原则实例。"""
        for pc in self.config.principles:
            if not pc.enabled:
                continue

            principle_cls = ConstitutionRegistry.get_principle(pc.name)
            if principle_cls is None:
                logger.warning(
                    "原则 '%s' 未注册，跳过。已注册: %s",
                    pc.name,
                    ConstitutionRegistry.list_principles(),
                )
                continue

            try:
                self._principles[pc.name] = principle_cls(**pc.params)
            except TypeError as e:
                logger.warning("原则 '%s' 初始化失败: %s", pc.name, e)

    def evaluate(
        self,
        state: SceneState,
        decision: AIDecision,
    ) -> EvaluationResult:
        """执行完整的宪法评估。

        Args:
            state: 当前场景状态
            decision: AI 决策

        Returns:
            EvaluationResult: 包含分数、违规、训练信号
        """
        # Step 1: 遍历原则评估
        all_results: list[tuple[str, ViolationResult]] = []
        principle_results: dict[str, ViolationResult] = {}
        violated_results: list[ViolationResult] = []

        for name, principle in self._principles.items():
            try:
                result = principle.evaluate(state, decision)
                all_results.append((name, result))
                principle_results[name] = result
                if result.violated:
                    violated_results.append(result)
            except Exception as e:
                logger.error("原则 '%s' 评估异常: %s", name, e)
                # 异常时创建安全结果，不阻塞流程
                safe_result = ViolationResult(
                    violated=False,
                    severity=principle.severity,
                    confidence=0.0,
                    description=f"评估异常: {e}",
                )
                all_results.append((name, safe_result))
                principle_results[name] = safe_result

        # Step 2: 打分
        safety_score = self._scorer.score_violations(all_results)

        # Step 3: 训练信号
        training_signal = self._generator.generate(
            safety_result=safety_score,
            scene_state=state,
            ai_decision=decision,
        )

        return EvaluationResult(
            safety_score=safety_score,
            violations=violated_results,
            training_signal=training_signal,
            principle_results=principle_results,
        )

    @property
    def active_principles(self) -> list[str]:
        """已激活的原则名称列表。"""
        return list(self._principles.keys())
