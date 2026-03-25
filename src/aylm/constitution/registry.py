"""宪法原则注册表。

提供插件机制，允许第三方注册自定义的宪法原则和打分器。
"""

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .base import ConstitutionPrinciple
    from .command_parser import CommandParser
    from .scorer import SafetyScorer
    from .training import TrainingSignalGenerator


class ConstitutionRegistry:
    """宪法原则注册表。

    支持第三方插件注册自定义的宪法原则、打分器和训练信号生成器。

    Example:
        >>> from aylm.constitution import ConstitutionRegistry, ConstitutionPrinciple
        >>>
        >>> @ConstitutionRegistry.register_principle("my_custom_rule")
        ... class MyCustomPrinciple(ConstitutionPrinciple):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_custom_rule"
        ...     # ... 实现其他方法
        >>>
        >>> # 获取已注册的原则
        >>> principle_cls = ConstitutionRegistry.get_principle("my_custom_rule")
        >>> principle = principle_cls()
    """

    _principles: ClassVar[dict[str, type["ConstitutionPrinciple"]]] = {}
    _scorers: ClassVar[dict[str, type["SafetyScorer"]]] = {}
    _generators: ClassVar[dict[str, type["TrainingSignalGenerator"]]] = {}
    _command_parsers: ClassVar[dict[str, type["CommandParser"]]] = {}

    @classmethod
    def register_principle(cls, name: str):
        """装饰器：注册自定义宪法原则。

        Args:
            name: 原则名称（唯一标识符）

        Example:
            >>> @ConstitutionRegistry.register_principle("no_collision")
            ... class NoCollisionPrinciple(ConstitutionPrinciple):
            ...     pass
        """

        def decorator(principle_cls: type["ConstitutionPrinciple"]):
            cls._principles[name] = principle_cls
            return principle_cls

        return decorator

    @classmethod
    def register_scorer(cls, name: str):
        """装饰器：注册自定义打分器。

        Args:
            name: 打分器名称

        Example:
            >>> @ConstitutionRegistry.register_scorer("weighted")
            ... class WeightedScorer(SafetyScorer):
            ...     pass
        """

        def decorator(scorer_cls: type["SafetyScorer"]):
            cls._scorers[name] = scorer_cls
            return scorer_cls

        return decorator

    @classmethod
    def register_generator(cls, name: str):
        """装饰器：注册自定义训练信号生成器。

        Args:
            name: 生成器名称

        Example:
            >>> @ConstitutionRegistry.register_generator("tfrecord")
            ... class TFRecordGenerator(TrainingSignalGenerator):
            ...     pass
        """

        def decorator(generator_cls: type["TrainingSignalGenerator"]):
            cls._generators[name] = generator_cls
            return generator_cls

        return decorator

    @classmethod
    def register_command_parser(cls, name: str):
        """装饰器：注册自定义指令解析器。

        Args:
            name: 解析器名称

        Example:
            >>> @ConstitutionRegistry.register_command_parser("ros")
            ... class ROSCommandParser(CommandParser):
            ...     pass
        """

        def decorator(parser_cls: type["CommandParser"]):
            cls._command_parsers[name] = parser_cls
            return parser_cls

        return decorator

    @classmethod
    def get_principle(cls, name: str) -> type["ConstitutionPrinciple"] | None:
        """获取已注册的原则类。"""
        return cls._principles.get(name)

    @classmethod
    def get_scorer(cls, name: str) -> type["SafetyScorer"] | None:
        """获取已注册的打分器类。"""
        return cls._scorers.get(name)

    @classmethod
    def get_generator(cls, name: str) -> type["TrainingSignalGenerator"] | None:
        """获取已注册的生成器类。"""
        return cls._generators.get(name)

    @classmethod
    def get_command_parser(cls, name: str) -> type["CommandParser"] | None:
        """获取已注册的指令解析器类。"""
        return cls._command_parsers.get(name)

    @classmethod
    def list_principles(cls) -> list[str]:
        """列出所有已注册的原则名称。"""
        return list(cls._principles.keys())

    @classmethod
    def list_scorers(cls) -> list[str]:
        """列出所有已注册的打分器名称。"""
        return list(cls._scorers.keys())

    @classmethod
    def list_generators(cls) -> list[str]:
        """列出所有已注册的生成器名称。"""
        return list(cls._generators.keys())

    @classmethod
    def list_command_parsers(cls) -> list[str]:
        """列出所有已注册的指令解析器名称。"""
        return list(cls._command_parsers.keys())

    @classmethod
    def create_principle(cls, name: str, **kwargs) -> "ConstitutionPrinciple":
        """创建原则实例。

        Args:
            name: 原则名称
            **kwargs: 传递给原则构造函数的参数

        Returns:
            原则实例

        Raises:
            KeyError: 如果原则未注册
        """
        principle_cls = cls._principles.get(name)
        if principle_cls is None:
            raise KeyError(f"原则 '{name}' 未注册。已注册: {cls.list_principles()}")
        return principle_cls(**kwargs)

    @classmethod
    def create_scorer(cls, name: str, **kwargs) -> "SafetyScorer":
        """创建打分器实例。"""
        scorer_cls = cls._scorers.get(name)
        if scorer_cls is None:
            raise KeyError(f"打分器 '{name}' 未注册。已注册: {cls.list_scorers()}")
        return scorer_cls(**kwargs)

    @classmethod
    def create_generator(cls, name: str, **kwargs) -> "TrainingSignalGenerator":
        """创建生成器实例。"""
        generator_cls = cls._generators.get(name)
        if generator_cls is None:
            raise KeyError(f"生成器 '{name}' 未注册。已注册: {cls.list_generators()}")
        return generator_cls(**kwargs)

    @classmethod
    def clear(cls) -> None:
        """清除所有注册（主要用于测试）。"""
        cls._principles.clear()
        cls._scorers.clear()
        cls._generators.clear()
        cls._command_parsers.clear()
