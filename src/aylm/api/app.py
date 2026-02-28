"""FastAPI HTTP 服务 — 宪法评估 API。

提供 RESTful + WebSocket 接口：
- POST /api/v1/evaluate        — 单帧评估（接收速度/障碍物/指令）
- POST /api/v1/evaluate/batch  — 批量时序评估
- WS   /api/v1/session         — 实时流式评估（WebSocket）
- GET  /api/v1/config          — 查看当前配置
- PUT  /api/v1/ego             — 动态修改自车参数
- GET  /api/v1/summary         — 会话统计摘要
- GET  /api/v1/health          — 健康检查

启动方式:
    $ aylm serve --port 8000
    $ curl http://localhost:8000/api/v1/health
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_app(
    ego_speed: float = 0.0,
    ego_heading: float = 0.0,
    approval_threshold: float = 0.6,
    config_path: str | None = None,
) -> Any:
    """创建 FastAPI 应用实例。

    Args:
        ego_speed: 初始自车速度 m/s
        ego_heading: 初始自车航向弧度
        approval_threshold: 安全分批准阈值
        config_path: 宪法配置文件路径

    Returns:
        FastAPI 应用实例

    Raises:
        ImportError: 缺少 fastapi/uvicorn 依赖
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError(
            "API 服务需要 fastapi 和 uvicorn。\n"
            "安装: pip install 'aylm[api]'  或  pip install fastapi uvicorn"
        ) from exc

    from .session import ConstitutionSession

    # 创建会话
    session = ConstitutionSession(
        ego_speed=ego_speed,
        ego_heading=ego_heading,
        approval_threshold=approval_threshold,
        config_path=config_path,
    )

    # 创建 FastAPI 应用
    app = FastAPI(
        title="A-YLM Constitution API",
        description="几何宪法式 AI 安全评估 API — 为自动驾驶/机器人提供实时安全验证",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 保存 session 到 app state
    app.state.session = session

    # ── 健康检查 ──────────────────────────────────────────

    @app.get("/api/v1/health")
    async def health():
        return {
            "status": "ok",
            "evaluator_available": session.is_available,
            "frame_count": session.frame_count,
            "ego_speed": session.ego_speed,
        }

    # ── 单帧评估 ──────────────────────────────────────────

    @app.post("/api/v1/evaluate")
    async def evaluate(body: dict[str, Any]):
        """单帧安全评估。

        Request body:
            {
                "obstacles": [{"center_robot": [x,y,z], ...}],
                "command": "turn left" | {"type": "trajectory", ...},
                "ego_speed": 10.0,     // 可选，临时覆盖
                "ego_heading": 0.0     // 可选，临时覆盖
            }
        """
        # 临时覆盖 ego 参数
        prev_speed = session.ego_speed
        prev_heading = session.ego_heading

        if "ego_speed" in body:
            session.update_ego(speed=body["ego_speed"])
        if "ego_heading" in body:
            session.update_ego(heading=body["ego_heading"])

        try:
            result = session.evaluate(
                obstacles=body.get("obstacles"),
                command=body.get("command"),
                timestamp=body.get("timestamp"),
            )
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("评估异常: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
        finally:
            # 恢复原始参数（除非调用者明确想要持久化变更）
            if body.get("persist_ego") is not True:
                session.update_ego(speed=prev_speed, heading=prev_heading)

    # ── 批量时序评估 ──────────────────────────────────────

    @app.post("/api/v1/evaluate/batch")
    async def evaluate_batch(body: dict[str, Any]):
        """批量按时序评估多帧。

        Request body:
            {
                "frames": [
                    {
                        "obstacles": [...],
                        "command": "...",
                        "timestamp": 0.0,
                        "ego_speed": 10.0,
                        "ego_heading": 0.0
                    },
                    ...
                ]
            }
        """
        frames = body.get("frames", [])
        if not frames:
            return JSONResponse(
                status_code=400,
                content={"error": "frames 列表不能为空"},
            )

        try:
            results = session.evaluate_batch(frames)
            return JSONResponse(
                content={
                    "results": results,
                    "summary": session.summary,
                }
            )
        except Exception as e:
            logger.error("批量评估异常: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    # ── 动态修改 ego 参数 ─────────────────────────────────

    @app.put("/api/v1/ego")
    async def update_ego(body: dict[str, Any]):
        """动态修改自车参数。

        Request body:
            {"speed": 15.0, "heading": 0.1}
        """
        session.update_ego(
            speed=body.get("speed"),
            heading=body.get("heading"),
        )
        return {
            "ego_speed": session.ego_speed,
            "ego_heading": session.ego_heading,
        }

    # ── 会话统计 ──────────────────────────────────────────

    @app.get("/api/v1/summary")
    async def summary():
        return session.summary

    # ── 查看配置 ──────────────────────────────────────────

    @app.get("/api/v1/config")
    async def get_config():
        if session._validator and hasattr(session._validator, "config"):
            return {
                "approval_threshold": session._approval_threshold,
                "ego_speed": session.ego_speed,
                "ego_heading": session.ego_heading,
                "config": (
                    session._validator.config.to_dict()
                    if hasattr(session._validator.config, "to_dict")
                    else str(session._validator.config)
                ),
            }
        return {
            "approval_threshold": session._approval_threshold,
            "ego_speed": session.ego_speed,
            "ego_heading": session.ego_heading,
        }

    # ── WebSocket 实时流式评估 ────────────────────────────

    @app.websocket("/api/v1/session")
    async def websocket_session(ws: WebSocket):
        """实时流式评估 WebSocket。

        客户端发送 JSON 帧，服务端实时返回评估结果。

        客户端发送格式:
            {"obstacles": [...], "command": "...", "ego_speed": 10.0, "timestamp": 0.0}

        控制消息:
            {"action": "update_ego", "speed": 15.0, "heading": 0.1}
            {"action": "reset"}
            {"action": "summary"}

        服务端返回:
            评估结果 JSON（同 /evaluate 返回格式）
        """
        await ws.accept()
        logger.info("WebSocket 客户端已连接")

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"error": "无效的 JSON 格式"})
                    continue

                # 处理控制消息
                action = data.get("action")
                if action == "update_ego":
                    session.update_ego(
                        speed=data.get("speed"),
                        heading=data.get("heading"),
                    )
                    await ws.send_json(
                        {
                            "action": "ego_updated",
                            "ego_speed": session.ego_speed,
                            "ego_heading": session.ego_heading,
                        }
                    )
                    continue

                if action == "reset":
                    session.reset()
                    await ws.send_json({"action": "session_reset"})
                    continue

                if action == "summary":
                    await ws.send_json({"action": "summary", "data": session.summary})
                    continue

                # 常规评估
                if "ego_speed" in data:
                    session.update_ego(speed=data["ego_speed"])
                if "ego_heading" in data:
                    session.update_ego(heading=data["ego_heading"])

                result = session.evaluate(
                    obstacles=data.get("obstacles"),
                    command=data.get("command"),
                    timestamp=data.get("timestamp"),
                )
                await ws.send_json(result)

        except WebSocketDisconnect:
            logger.info("WebSocket 客户端已断开")
        except Exception as e:
            logger.error("WebSocket 异常: %s", e)
            with _suppress_ws_close():
                await ws.close(code=1011, reason=str(e))


def _suppress_ws_close():
    """安全关闭 WebSocket。"""
    import contextlib

    return contextlib.suppress(Exception)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    ego_speed: float = 0.0,
    ego_heading: float = 0.0,
    approval_threshold: float = 0.6,
    config_path: str | None = None,
    reload: bool = False,
) -> None:
    """启动 API 服务。

    Args:
        host: 监听地址
        port: 监听端口
        ego_speed: 初始自车速度
        ego_heading: 初始自车航向
        approval_threshold: 安全分阈值
        config_path: 宪法配置文件路径
        reload: 是否启用热重载（开发用）
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "API 服务需要 uvicorn。\n"
            "安装: pip install 'aylm[api]'  或  pip install uvicorn"
        ) from exc

    app = create_app(
        ego_speed=ego_speed,
        ego_heading=ego_heading,
        approval_threshold=approval_threshold,
        config_path=config_path,
    )

    print("\n  A-YLM Constitution API 启动中...")
    print(f"  地址: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs")
    print(f"  自车速度: {ego_speed} m/s")
    print(f"  安全阈值: {approval_threshold}\n")

    uvicorn.run(app, host=host, port=port, reload=reload)
