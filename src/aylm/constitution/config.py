"""宪法配置模块。

支持从 YAML/JSON 文件加载宪法配置，便于用户自定义规则。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PrincipleConfig:
    """单个宪法原则配置。"""

    name: str
    severity: str = "medium"  # critical, high, medium, low
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstitutionConfig:
    """宪法配置。

    支持从 YAML/JSON 文件加载，便于用户自定义安全规则。

    Example YAML:
        ```yaml
        principles:
          - name: no_collision
            severity: critical
            enabled: true
          - name: ttc_safety
            severity: high
            params:
              warning_threshold: 3.0
              critical_threshold: 1.5

        thresholds:
          min_safe_distance: 2.0
          speed_distance_factor: 0.5

        training:
          generate_positive_signals: false
          export_format: json
        ```
    """

    # 宪法原则列表
    principles: list[PrincipleConfig] = field(default_factory=list)

    # 安全阈值
    ttc_warning_threshold: float = 3.0  # TTC 警告阈值（秒）
    ttc_critical_threshold: float = 1.5  # TTC 关键阈值（秒）
    min_safe_distance: float = 2.0  # 最小安全距离（米）
    speed_distance_factor: float = 0.5  # 速度-距离系数

    # 训练信号配置
    generate_positive_signals: bool = False  # 是否生成正样本
    signal_export_format: str = "json"  # 导出格式

    # 打分权重
    collision_weight: float = 1.0
    ttc_weight: float = 0.8
    boundary_weight: float = 0.5

    def __post_init__(self):
        """初始化默认原则（如果未指定）。"""
        if not self.principles:
            self.principles = self._default_principles()

    @staticmethod
    def _default_principles() -> list[PrincipleConfig]:
        """默认宪法原则。"""
        return [
            PrincipleConfig(name="no_collision", severity="critical"),
            PrincipleConfig(name="safe_following", severity="high"),
            PrincipleConfig(name="ttc_safety", severity="high"),
            PrincipleConfig(name="lane_compliance", severity="medium"),
            PrincipleConfig(name="speed_limit", severity="medium"),
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstitutionConfig":
        """从字典创建配置。"""
        principles = []
        for p in data.get("principles", []):
            principles.append(
                PrincipleConfig(
                    name=p["name"],
                    severity=p.get("severity", "medium"),
                    enabled=p.get("enabled", True),
                    params=p.get("params", {}),
                )
            )

        thresholds = data.get("thresholds", {})
        training = data.get("training", {})
        weights = data.get("weights", {})

        return cls(
            principles=principles if principles else None,
            ttc_warning_threshold=thresholds.get("ttc_warning", 3.0),
            ttc_critical_threshold=thresholds.get("ttc_critical", 1.5),
            min_safe_distance=thresholds.get("min_safe_distance", 2.0),
            speed_distance_factor=thresholds.get("speed_distance_factor", 0.5),
            generate_positive_signals=training.get("generate_positive_signals", False),
            signal_export_format=training.get("export_format", "json"),
            collision_weight=weights.get("collision", 1.0),
            ttc_weight=weights.get("ttc", 0.8),
            boundary_weight=weights.get("boundary", 0.5),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ConstitutionConfig":
        """从 JSON 文件加载配置。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConstitutionConfig":
        """从 YAML 文件加载配置。"""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("需要安装 pyyaml: pip install pyyaml") from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "principles": [
                {
                    "name": p.name,
                    "severity": p.severity,
                    "enabled": p.enabled,
                    "params": p.params,
                }
                for p in self.principles
            ],
            "thresholds": {
                "ttc_warning": self.ttc_warning_threshold,
                "ttc_critical": self.ttc_critical_threshold,
                "min_safe_distance": self.min_safe_distance,
                "speed_distance_factor": self.speed_distance_factor,
            },
            "training": {
                "generate_positive_signals": self.generate_positive_signals,
                "export_format": self.signal_export_format,
            },
            "weights": {
                "collision": self.collision_weight,
                "ttc": self.ttc_weight,
                "boundary": self.boundary_weight,
            },
        }

    def save_json(self, path: str | Path) -> None:
        """保存为 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_yaml(self, path: str | Path) -> None:
        """保存为 YAML 文件。"""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("需要安装 pyyaml: pip install pyyaml") from e

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    def get_principle(self, name: str) -> PrincipleConfig | None:
        """获取指定名称的原则配置。"""
        for p in self.principles:
            if p.name == name:
                return p
        return None

    def is_principle_enabled(self, name: str) -> bool:
        """检查原则是否启用。"""
        p = self.get_principle(name)
        return p.enabled if p else False
