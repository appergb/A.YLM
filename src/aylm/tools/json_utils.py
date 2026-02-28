"""numpy 安全 JSON 序列化工具。

解决 numpy 类型（np.bool_, np.float64, np.ndarray 等）
无法被 json.dump 序列化的问题。
"""

import json
from typing import IO, Any

import numpy as np


def _numpy_default(obj: Any) -> Any:
    """json.dump 的 default 回调，处理 numpy 类型。"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def numpy_safe_dump(data: Any, f: IO[str], **kwargs: Any) -> None:
    """json.dump 的 numpy 安全版本。

    自动将 np.bool_, np.integer, np.floating, np.ndarray
    转换为 Python 原生类型后再序列化。

    Args:
        data: 要序列化的数据
        f: 文件对象
        **kwargs: 传递给 json.dump 的额外参数
    """
    kwargs.setdefault("default", _numpy_default)
    json.dump(data, f, **kwargs)


def numpy_safe_dumps(data: Any, **kwargs: Any) -> str:
    """json.dumps 的 numpy 安全版本。

    Args:
        data: 要序列化的数据
        **kwargs: 传递给 json.dumps 的额外参数

    Returns:
        JSON 字符串
    """
    kwargs.setdefault("default", _numpy_default)
    return json.dumps(data, **kwargs)
