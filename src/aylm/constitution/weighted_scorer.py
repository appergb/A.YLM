"""加权安全打分器。

按类别权重聚合违规结果，计算综合安全分数。
"""

import logging
from typing import TYPE_CHECKING

from .base import Severity, ViolationResult
from .config import ConstitutionConfig
from .registry import ConstitutionRegistry
from .scorer import RecommendedAction, SafetyScore, SafetyScorer

if TYPE_CHECKING:
    from ..tools.obstacle_marker import ObstacleBox3D
    from .types import AIDecision, EgoState

logger = logging.getLogger(__name__)

# Principle name -> weight category mapping
CATEGORY_MAP: dict[str, str] = {
    "no_collision": "collision",
    "safe_following": "collision",
    "ttc_safety": "ttc",
    "lane_compliance": "boundary",
    "speed_limit": "boundary",
}


@ConstitutionRegistry.register_scorer("weighted")
class WeightedSafetyScorer(SafetyScorer):
    """加权安全打分器。

    按类别（collision, ttc, boundary）聚合违规，使用配置中的权重
    计算综合安全分数。

    每类别分数 = 1.0 - max(severity.value × confidence) for violations in category
    overall = Σ(weight × category_score) / Σ(weights)
    """

    def __init__(self, config: ConstitutionConfig | None = None):
        self.config = config or ConstitutionConfig()

    def score(
        self,
        obstacles: list["ObstacleBox3D"],
        ego_state: "EgoState",
        ai_decision: "AIDecision",
    ) -> SafetyScore:
        """Not used directly. Use score_violations instead."""
        raise NotImplementedError("Use score_violations() with ViolationResult list")

    def score_violations(
        self,
        violations: list[tuple[str, ViolationResult]],
    ) -> SafetyScore:
        """从违规结果列表计算安全分数。

        Args:
            violations: (principle_name, ViolationResult) pairs

        Returns:
            SafetyScore
        """
        # Group violations by category
        category_impacts: dict[str, float] = {
            "collision": 0.0,
            "ttc": 0.0,
            "boundary": 0.0,
        }
        violated_names: list[str] = []
        violation_details: list[dict] = []
        max_severity = Severity.LOW
        has_critical = False
        has_high = False

        for name, result in violations:
            if result.violated:
                violated_names.append(name)
                violation_details.append(result.to_dict())

                # Track max severity
                if result.severity == Severity.CRITICAL:
                    has_critical = True
                if result.severity in (Severity.CRITICAL, Severity.HIGH):
                    has_high = True
                if result.severity.value > max_severity.value:
                    max_severity = result.severity

                # Map to category
                category = CATEGORY_MAP.get(name, "boundary")
                impact = result.severity.value * result.confidence
                if impact > category_impacts[category]:
                    category_impacts[category] = impact

        # Compute category scores
        collision_score = 1.0 - category_impacts["collision"]
        ttc_score = 1.0 - category_impacts["ttc"]
        boundary_score = 1.0 - category_impacts["boundary"]

        # Weighted average
        weights = {
            "collision": self.config.collision_weight,
            "ttc": self.config.ttc_weight,
            "boundary": self.config.boundary_weight,
        }
        scores = {
            "collision": collision_score,
            "ttc": ttc_score,
            "boundary": boundary_score,
        }
        total_weight = sum(weights.values())
        overall = (
            sum(weights[k] * scores[k] for k in weights) / total_weight
            if total_weight > 0
            else 1.0
        )

        # Determine recommended action
        if has_critical:
            action = RecommendedAction.EMERGENCY_STOP
        elif has_high:
            action = RecommendedAction.INTERVENTION
        elif overall < 0.5:
            action = RecommendedAction.WARNING
        elif overall < 0.8:
            action = RecommendedAction.CAUTION
        else:
            action = RecommendedAction.SAFE

        # Confidence based on number of principles evaluated
        total_principles = len(violations)
        confidence = min(1.0, total_principles / 3.0) if total_principles > 0 else 0.5

        return SafetyScore(
            overall=round(overall, 4),
            collision_score=round(collision_score, 4),
            ttc_score=round(ttc_score, 4),
            boundary_score=round(boundary_score, 4),
            violations=violated_names,
            violation_details=violation_details,
            recommended_action=action,
            confidence=round(confidence, 4),
        )
