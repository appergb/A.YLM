"""A-YLM 宪法评估 API 模块。

提供两种访问方式：
1. Python API — 直接导入 ConstitutionSession 使用
2. HTTP API — 通过 FastAPI 服务 (aylm serve)

Example (Python):
    >>> from aylm.api import ConstitutionSession
    >>> session = ConstitutionSession(ego_speed=10.0)
    >>> result = session.evaluate(obstacles=[...])
    >>> print(result["approved"])

Example (HTTP):
    $ aylm serve --port 8000
    $ curl -X POST http://localhost:8000/api/v1/evaluate \\
        -H 'Content-Type: application/json' \\
        -d '{"ego_speed": 10.0, "obstacles": [...]}'
"""

from .session import ConstitutionSession

__all__ = [
    "ConstitutionSession",
]
