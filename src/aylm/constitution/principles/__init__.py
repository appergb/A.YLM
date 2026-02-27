"""内置宪法原则实现。

提供常用的宪法原则参考实现，用户可以直接使用或作为自定义实现的参考。
"""

from .collision import NoCollisionPrinciple
from .ttc import TTCSafetyPrinciple

__all__ = [
    "NoCollisionPrinciple",
    "TTCSafetyPrinciple",
]
