"""宪法评估演示模块。

模拟 4 帧逐渐逼近场景，调用真实的 ConstitutionEvaluator，
展示距离判断、碰撞时间预测和碰撞检测输出。

不依赖 SHARP/YOLO/Open3D，只需 numpy。
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..constitution import (
    ConstitutionConfig,
    ConstitutionEvaluator,
    ConstitutionObstacle,
    ObstacleMotion,
    RecommendedAction,
)
from ..constitution.config import PrincipleConfig
from ..constitution.types import AIDecision, EgoState, SceneState, TrajectoryPoint

# ── ANSI 颜色 ──────────────────────────────────────────────


class _C:
    """终端颜色常量。"""

    R = "\033[0;31m"
    G = "\033[0;32m"
    Y = "\033[1;33m"
    B = "\033[0;34m"
    CY = "\033[0;36m"
    BD = "\033[1m"
    DM = "\033[2m"
    NC = "\033[0m"


def _color_on() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ── 场景数据 ───────────────────────────────────────────────

# 4 帧逐渐逼近：ego 10 m/s，前方车辆逐步减速
FRAMES: list[dict] = [
    {
        "frame_id": 1,
        "timestamp": 0.0,
        "ego_speed": 10.0,
        "obs_distance": 30.0,
        "obs_speed": 8.0,
    },
    {
        "frame_id": 2,
        "timestamp": 1.0,
        "ego_speed": 10.0,
        "obs_distance": 20.0,
        "obs_speed": 5.0,
    },
    {
        "frame_id": 3,
        "timestamp": 2.0,
        "ego_speed": 10.0,
        "obs_distance": 8.0,
        "obs_speed": 2.0,
    },
    {
        "frame_id": 4,
        "timestamp": 3.0,
        "ego_speed": 10.0,
        "obs_distance": 1.5,
        "obs_speed": 0.0,
    },
]


# ── 场景构建 ───────────────────────────────────────────────


def _build_scene(frame: dict) -> tuple[SceneState, AIDecision]:
    """从帧数据构建 SceneState 和 AIDecision。"""
    ego_speed = frame["ego_speed"]
    obs_distance = frame["obs_distance"]
    obs_speed = frame["obs_speed"]

    # 自车状态：位于原点，沿 X 轴前进（机器人坐标系）
    ego_state = EgoState(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([ego_speed, 0.0, 0.0], dtype=np.float32),
        heading=0.0,
        speed=ego_speed,
    )

    # 前方障碍物（典型轿车尺寸 4.5×1.8×1.5m）
    # is_stationary=False 即使 speed=0，使 TTC/following 原则仍可评估
    obstacle = ConstitutionObstacle(
        center_robot=np.array([obs_distance, 0.0, 0.0], dtype=np.float32),
        dimensions=np.array([4.5, 1.8, 1.5], dtype=np.float32),
        label="car",
        confidence=0.95,
        track_id=1,
        motion=ObstacleMotion(
            velocity_robot=np.array([obs_speed, 0.0, 0.0], dtype=np.float32),
            speed=obs_speed,
            heading=0.0,
            is_stationary=False,
        ),
    )

    scene = SceneState(
        frame_id=frame["frame_id"],
        timestamp=frame["timestamp"],
        ego_state=ego_state,
        obstacles=[obstacle],
    )

    # AI 决策：保持当前速度直行（0.5s 短轨迹）
    trajectory = [
        TrajectoryPoint(
            position=np.array([ego_speed * t, 0.0, 0.0], dtype=np.float32),
            velocity=np.array([ego_speed, 0.0, 0.0], dtype=np.float32),
            timestamp=t,
        )
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]

    decision = AIDecision(
        decision_type="trajectory",
        trajectory=trajectory,
        target_speed=ego_speed,
        confidence=1.0,
    )

    return scene, decision


# ── 格式化辅助 ─────────────────────────────────────────────


def _mark(passed: bool, color: bool) -> str:
    if passed:
        return f"{_C.G if color else ''}✓{_C.NC if color else ''}"
    return f"{_C.R if color else ''}✗{_C.NC if color else ''}"


def _action_str(action: RecommendedAction, color: bool) -> str:
    table = {
        RecommendedAction.SAFE: ("✓ 安全", _C.G),
        RecommendedAction.CAUTION: ("⚠ 注意", _C.Y),
        RecommendedAction.WARNING: ("⚠ 警告", _C.Y),
        RecommendedAction.INTERVENTION: ("✗ 需要干预", _C.R),
        RecommendedAction.EMERGENCY_STOP: ("✗ 紧急停车", _C.R),
    }
    text, clr = table.get(action, ("?", ""))
    if color:
        return f"{clr}{text}{_C.NC}"
    return text


def _score_color(score: float, color: bool) -> str:
    txt = f"{score:.2f}"
    if not color:
        return txt
    if score >= 0.8:
        return f"{_C.G}{txt}{_C.NC}"
    if score >= 0.4:
        return f"{_C.Y}{txt}{_C.NC}"
    return f"{_C.R}{txt}{_C.NC}"


# ── 主入口 ─────────────────────────────────────────────────


def run_demo(verbose: bool = False) -> int:
    """运行宪法评估演示。

    Args:
        verbose: 显示完整 JSON 评估结果

    Returns:
        0 表示成功
    """
    c = _color_on()
    cy = _C.CY if c else ""
    bd = _C.BD if c else ""
    dm = _C.DM if c else ""
    nc = _C.NC if c else ""
    r = _C.R if c else ""
    y = _C.Y if c else ""

    # ── Banner ──
    print(f"\n{cy}{'=' * 60}{nc}")
    print(f"{cy}  A.YLM v2 - 宪法式 AI 安全评估演示{nc}")
    print(f"{cy}{'=' * 60}{nc}")

    # ── 评估器配置（3 个核心原则） ──
    config = ConstitutionConfig(
        principles=[
            PrincipleConfig(
                name="no_collision",
                severity="critical",
                params={"safety_margin": 0.5, "prediction_horizon": 2.0},
            ),
            PrincipleConfig(
                name="safe_following",
                severity="high",
                params={"time_gap": 2.0, "min_gap": 2.0},
            ),
            PrincipleConfig(
                name="ttc_safety",
                severity="high",
                params={
                    "warning_threshold": 3.0,
                    "critical_threshold": 1.5,
                    "min_safe_distance": 2.0,
                },
            ),
        ],
        generate_positive_signals=True,
    )

    evaluator = ConstitutionEvaluator(config)
    ego_speed = FRAMES[0]["ego_speed"]

    print(
        f"\n{bd}[场景]{nc} 自车 {ego_speed:.1f} m/s "
        f"({ego_speed * 3.6:.0f} km/h)，前方车辆逐渐减速\n"
    )
    print(f"{dm}  原则: {', '.join(evaluator.active_principles)}{nc}")
    print(f"{dm}  安全边距: 0.5m | TTC 警告: 3.0s | TTC 关键: 1.5s{nc}")

    # ── 逐帧评估 ──
    frame_results = []
    training_signals = []

    for frame in FRAMES:
        scene, decision = _build_scene(frame)
        result = evaluator.evaluate(scene, decision)
        frame_results.append((frame, result))
        if result.training_signal:
            training_signals.append(result.training_signal)

        fid = frame["frame_id"]
        ts = frame["timestamp"]
        obs_dist = frame["obs_distance"]
        obs_spd = frame["obs_speed"]

        # 帧头
        print(f"\n{bd}━━━ 第 {fid} 帧 (t={ts:.1f}s) ━━━{nc}")

        # 障碍物
        spd_txt = "已停车" if obs_spd < 0.1 else f"速度 {obs_spd:.1f} m/s"
        print(f"  [障碍物] 车辆 @ {obs_dist:.1f}m | {spd_txt}")

        # ── 跟车距离 ──
        following = result.principle_results.get("safe_following")
        if following:
            fm = following.metrics
            actual = fm.get("min_distance")
            required = fm.get("required_distance")
            if actual is not None and required is not None:
                mk = _mark(not following.violated, c)
                print(f"  [跟车距离] {actual:.1f}m / 需要 {required:.1f}m {mk}")
            else:
                mk = _mark(not following.violated, c)
                print(f"  [跟车距离] {following.description} {mk}")

        # ── TTC ──
        ttc_result = result.principle_results.get("ttc_safety")
        if ttc_result:
            tm = ttc_result.metrics
            min_ttc = tm.get("min_ttc", -1)
            if min_ttc > 0:
                threshold = 3.0
                mk = _mark(not ttc_result.violated, c)
                cmp = ">" if min_ttc > threshold else "<"
                print(f"  [TTC] {min_ttc:.1f}s {cmp} {threshold:.1f}s {mk}")
                if ttc_result.violated:
                    level = "关键" if min_ttc < 1.5 else "警告"
                    clr = r if min_ttc < 1.5 else y
                    print(f"         {clr}{level}: {ttc_result.description}{nc}")
            elif min_ttc < 0:
                # 负 TTC = 已在碰撞区域
                mk = _mark(False, c)
                print(f"  [TTC] {r}TTC < 0 — 已进入碰撞区域{nc} {mk}")
            else:
                mk = _mark(not ttc_result.violated, c)
                print(f"  [TTC] {ttc_result.description} {mk}")

        # ── 碰撞 ──
        collision = result.principle_results.get("no_collision")
        if collision:
            cm = collision.metrics
            min_dist = cm.get("min_distance", float("inf"))
            mk = _mark(not collision.violated, c)
            if collision.violated:
                print(
                    f"  {r}[碰撞] 最小距离 {min_dist:.1f}m "
                    f"< 安全边距 {mk} 碰撞风险！{nc}"
                )
            else:
                print(f"  [碰撞] 最小距离 {min_dist:.1f}m {mk}")

        # ── 总分 ──
        score = result.safety_score.overall
        action = _action_str(result.safety_score.recommended_action, c)
        sc = _score_color(score, c)
        print(f"  {bd}安全评分: {sc} | 动作: {action}")

        # ── Verbose JSON ──
        if verbose:
            print(f"\n  {dm}--- 详细评估结果 ---{nc}")
            formatted = json.dumps(
                result.to_dict(), indent=4, ensure_ascii=False, default=str
            )
            for line in formatted.splitlines():
                print(f"  {line}")

    # ── 汇总 ──
    print(f"\n{bd}{'=' * 10} 汇总 {'=' * 10}{nc}")
    _, first = frame_results[0]
    last_frame, last = frame_results[-1]
    first_sc = _score_color(first.safety_score.overall, c)
    last_sc = _score_color(last.safety_score.overall, c)
    first_act = _action_str(first.safety_score.recommended_action, c)
    last_act = _action_str(last.safety_score.recommended_action, c)

    print(f"  帧 1: {first_act} ({first_sc})  →  帧 4: {last_act} ({last_sc})")

    # 碰撞时间预估
    last_dist = last_frame["obs_distance"]
    if ego_speed > 0:
        time_to_crash = last_dist / ego_speed
        print(f"  如继续行驶，预计 {y}{time_to_crash:.2f}s{nc} 后碰撞")

    # 训练信号统计
    if training_signals:
        counts: dict[str, int] = {}
        for sig in training_signals:
            key = sig.signal_type.value.upper()
            counts[key] = counts.get(key, 0) + 1
        parts = [f"{v} {k}" for k, v in counts.items()]
        print(f"  训练信号: {len(training_signals)} 条 ({' + '.join(parts)})")

    print()
    return 0
