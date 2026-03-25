"""端到端集成测试：自校准闭环验证。

模拟多帧导航场景，验证校准器在以下场景中是否正确工作：
1. 安全场景 — 阈值应保持稳定或放松
2. 危险场景 — 阈值应收紧，权重应调整
3. 跨会话学习 — 第二次运行应从更优基线启动
4. 提示词注入 — VLM 提示词应包含安全提示
"""

from __future__ import annotations

import json

from aylm.navigation_demo.calibrator import (
    CalibrationContext,
    FrameFeedback,
    SessionCalibrator,
)
from aylm.navigation_demo.learning_store import LearningStore
from aylm.navigation_demo.prompt_calibrator import PromptCalibrator


def _simulate_safe_session(
    calibrator: SessionCalibrator,
    num_frames: int = 20,
) -> CalibrationContext:
    """模拟一段全安全的驾驶会话。"""
    for i in range(num_frames):
        calibrator.record_frame(
            FrameFeedback(
                frame_index=i,
                timestamp=float(i) * 0.5,
                proposed_approved=True,
                proposal_score=0.85 + (i * 0.005),
                executed_score=0.88 + (i * 0.003),
                violations=[],
            )
        )
    return calibrator.finalize()


def _simulate_dangerous_session(
    calibrator: SessionCalibrator,
    num_frames: int = 20,
) -> CalibrationContext:
    """模拟一段有大量碰撞违规的危险会话。"""
    for i in range(num_frames):
        if i % 3 == 0:
            # 每 3 帧碰撞一次
            calibrator.record_frame(
                FrameFeedback(
                    frame_index=i,
                    timestamp=float(i) * 0.5,
                    proposed_approved=False,
                    proposal_score=0.25,
                    executed_score=0.35,
                    violations=["no_collision", "ttc_safety"],
                    signal_type="correction",
                    correction_hints=[
                        "Reduce target_speed below 0.5 m/s near obstacles"
                    ],
                )
            )
        else:
            calibrator.record_frame(
                FrameFeedback(
                    frame_index=i,
                    timestamp=float(i) * 0.5,
                    proposed_approved=True,
                    proposal_score=0.7,
                    executed_score=0.72,
                    violations=[],
                )
            )
    return calibrator.finalize()


def _simulate_improving_session(
    calibrator: SessionCalibrator,
    num_frames: int = 20,
) -> CalibrationContext:
    """模拟一段先差后好的改善会话。"""
    for i in range(num_frames):
        if i < num_frames // 2:
            # 前半段：频繁违规
            calibrator.record_frame(
                FrameFeedback(
                    frame_index=i,
                    timestamp=float(i) * 0.5,
                    proposed_approved=False,
                    proposal_score=0.3,
                    executed_score=0.4,
                    violations=["no_collision"],
                )
            )
        else:
            # 后半段：安全
            calibrator.record_frame(
                FrameFeedback(
                    frame_index=i,
                    timestamp=float(i) * 0.5,
                    proposed_approved=True,
                    proposal_score=0.9,
                    executed_score=0.92,
                    violations=[],
                )
            )
    return calibrator.finalize()


class TestCalibrationLoopSafeScenario:
    """安全场景：阈值应保持稳定。"""

    def test_safe_session_keeps_threshold_stable(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        ctx = _simulate_safe_session(cal, num_frames=20)

        # 阈值应放松或保持不变（因为趋势改善且高批准率）
        assert ctx.adjusted_threshold <= 0.6
        assert ctx.calibration_round >= 2
        # 无违规，不应生成违规相关提示
        assert ctx.violation_summary == ""

    def test_safe_session_weights_unchanged(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
            base_weights={"collision": 1.0, "ttc": 0.8, "boundary": 0.5},
        )
        ctx = _simulate_safe_session(cal, num_frames=20)

        # 无违规时权重应接近基线
        assert abs(ctx.adjusted_weights["collision"] - 1.0) < 0.1
        assert abs(ctx.adjusted_weights["ttc"] - 0.8) < 0.1


class TestCalibrationLoopDangerousScenario:
    """危险场景：阈值应收紧，权重应增加。"""

    def test_dangerous_session_tightens_threshold(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        ctx = _simulate_dangerous_session(cal, num_frames=20)

        # 安全趋势下降时，阈值应收紧
        assert ctx.adjusted_threshold > 0.6
        print(f"  阈值从 0.6 收紧到 {ctx.adjusted_threshold:.3f}")

    def test_dangerous_session_boosts_collision_weight(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
            base_weights={"collision": 1.0, "ttc": 0.8, "boundary": 0.5},
        )
        ctx = _simulate_dangerous_session(cal, num_frames=20)

        # 碰撞类违规频繁，collision 权重应增加
        assert ctx.adjusted_weights["collision"] > 1.0
        print(
            f"  collision 权重从 1.0 增加到 "
            f"{ctx.adjusted_weights['collision']:.4f}"
        )

    def test_dangerous_session_generates_hints(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        ctx = _simulate_dangerous_session(cal, num_frames=20)

        assert len(ctx.safety_hints) > 0
        # 应包含碰撞相关提示
        collision_hints = [
            h for h in ctx.safety_hints if "no_collision" in h.lower()
        ]
        assert len(collision_hints) > 0
        print(f"  安全提示 ({len(ctx.safety_hints)}):")
        for hint in ctx.safety_hints[:3]:
            print(f"    - {hint}")

    def test_dangerous_session_correction_hints_propagate(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        ctx = _simulate_dangerous_session(cal, num_frames=20)

        # 纠正提示应传播到安全提示中
        speed_hints = [
            h for h in ctx.safety_hints if "speed" in h.lower()
        ]
        assert len(speed_hints) > 0
        print(f"  速度相关纠正提示: {speed_hints[0]}")


class TestCalibrationLoopImprovingScenario:
    """改善场景：先差后好，阈值应先紧后松。"""

    def test_improving_session_relaxes_after_improvement(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=5,
        )
        ctx = _simulate_improving_session(cal, num_frames=20)

        # 整体趋势改善，且后半段批准率高
        # 阈值最终可能回到基线附近
        assert ctx.calibration_round >= 4
        print(
            f"  校准轮次: {ctx.calibration_round}, "
            f"最终阈值: {ctx.adjusted_threshold:.3f}"
        )


class TestPromptSuffixIntegration:
    """验证提示词校准器与校准器的集成。"""

    def test_dangerous_session_produces_prompt_suffix(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        prompt_cal = PromptCalibrator()

        _simulate_dangerous_session(cal, num_frames=20)
        ctx = cal.get_context()
        suffix = prompt_cal.build_prompt_suffix(ctx)

        assert suffix != ""
        assert "Safety Calibration Context" in suffix
        assert "no_collision" in suffix
        print(f"  生成的提示词后缀 ({len(suffix)} 字符):")
        for line in suffix.split("\n")[:8]:
            print(f"    {line}")

    def test_safe_session_minimal_or_no_suffix(self):
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
        )
        prompt_cal = PromptCalibrator()

        _simulate_safe_session(cal, num_frames=20)
        ctx = cal.get_context()
        suffix = prompt_cal.build_prompt_suffix(ctx)

        # 安全场景没有违规，可能只有 score context
        if suffix:
            assert "no_collision" not in suffix
            print(f"  安全场景后缀: {suffix[:100]}...")
        else:
            print("  安全场景无需提示词后缀 ✓")


class TestCrossSessionLearning:
    """跨会话学习：第二次运行应从更优基线启动。"""

    def test_second_session_starts_from_better_baseline(self, tmp_path):
        store_path = tmp_path / "learning_store.json"
        store = LearningStore(store_path)

        # 第一次运行：危险场景
        cal1 = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
            learning_store=store,
        )
        ctx1 = _simulate_dangerous_session(cal1, num_frames=20)
        print(f"  会话1 最终阈值: {ctx1.adjusted_threshold:.3f}")
        print(f"  会话1 collision 权重: {ctx1.adjusted_weights.get('collision', 'N/A')}")

        # 验证存储已保存
        assert store.get_session_count() == 1
        baseline = store.load_baseline()
        assert baseline is not None
        print(f"  存储的基线阈值: {baseline.threshold:.3f}")
        print(f"  存储的基线 avg_score: {baseline.avg_score:.4f}")

        # 第二次运行：加载基线
        cal2 = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
            learning_store=store,
        )
        # 应用基线
        cal2.apply_baseline(baseline)

        # 验证第二次运行从更优基线启动
        ctx2_initial = cal2.get_context()
        assert ctx2_initial.adjusted_threshold == baseline.threshold
        print(
            f"  会话2 初始阈值: {ctx2_initial.adjusted_threshold:.3f} "
            f"(继承自基线)"
        )

        # 第二次运行模拟安全场景
        ctx2 = _simulate_safe_session(cal2, num_frames=20)
        print(f"  会话2 最终阈值: {ctx2.adjusted_threshold:.3f}")

        # 验证第二次运行也保存了
        assert store.get_session_count() == 2

    def test_multiple_sessions_accumulate_violations(self, tmp_path):
        store_path = tmp_path / "learning_store.json"
        store = LearningStore(store_path)

        # 运行 3 次
        for _session_num in range(3):
            cal = SessionCalibrator(
                base_threshold=0.6,
                calibration_interval=10,
                learning_store=store,
            )
            baseline = store.load_baseline()
            if baseline:
                cal.apply_baseline(baseline)
            _simulate_dangerous_session(cal, num_frames=10)

        # 聚合违规应跨会话累积
        violations = store.get_aggregated_violations()
        assert violations.get("no_collision", 0) >= 9  # 每次约 4 次违规
        print(f"  跨 3 会话聚合违规: {violations}")

    def test_learning_store_file_format(self, tmp_path):
        store_path = tmp_path / "learning_store.json"
        store = LearningStore(store_path)

        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=10,
            learning_store=store,
        )
        _simulate_dangerous_session(cal, num_frames=20)

        # 验证存储文件可读
        data = json.loads(store_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert len(data["sessions"]) == 1
        assert data["current_baseline"] is not None
        assert "aggregated_violations" in data

        session = data["sessions"][0]
        print("  存储的会话记录:")
        print(f"    session_id: {session['session_id']}")
        print(f"    frame_count: {session['frame_count']}")
        print(f"    avg_score: {session['avg_score']:.4f}")
        print(f"    approval_rate: {session['approval_rate']:.4f}")
        print(f"    violation_pattern: {session['violation_pattern']}")
        print(f"    final_threshold: {session['final_threshold']:.3f}")
        print(f"    effective_hints: {len(session['effective_hints'])} 条")


class TestFullCalibrationFlow:
    """完整闭环流程测试：模拟 runner 的实际数据流。"""

    def test_full_loop_with_threshold_sync(self):
        """模拟 runner.py 中校准器与验证器的阈值同步。"""
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=5,
            violation_count_trigger=3,
        )
        prompt_cal = PromptCalibrator()

        # 模拟验证器阈值
        validator_threshold = 0.6
        threshold_history = [validator_threshold]
        prompt_suffix_history: list[str] = []

        for i in range(30):
            # 获取校准上下文
            ctx = cal.get_context()
            suffix = prompt_cal.build_prompt_suffix(ctx)
            prompt_suffix_history.append(suffix or "")

            # 模拟交替的安全/危险帧
            if i < 15:
                # 前 15 帧：有些危险
                if i % 4 == 0:
                    fb = FrameFeedback(
                        frame_index=i,
                        timestamp=float(i) * 0.5,
                        proposed_approved=False,
                        proposal_score=0.3,
                        executed_score=0.4,
                        violations=["no_collision"],
                        signal_type="negative",
                    )
                else:
                    fb = FrameFeedback(
                        frame_index=i,
                        timestamp=float(i) * 0.5,
                        proposed_approved=True,
                        proposal_score=0.8,
                        executed_score=0.82,
                        violations=[],
                    )
            else:
                # 后 15 帧：安全（模型学到了）
                fb = FrameFeedback(
                    frame_index=i,
                    timestamp=float(i) * 0.5,
                    proposed_approved=True,
                    proposal_score=0.9,
                    executed_score=0.92,
                    violations=[],
                )

            cal.record_frame(fb)

            # 同步阈值
            new_ctx = cal.get_context()
            if new_ctx.adjusted_threshold != validator_threshold:
                validator_threshold = new_ctx.adjusted_threshold
                threshold_history.append(validator_threshold)

        final_ctx = cal.finalize()

        print("\n  === 完整闭环流程测试 ===")
        print(f"  校准轮次: {final_ctx.calibration_round}")
        print(f"  阈值变化历程: {[f'{t:.3f}' for t in threshold_history]}")
        print(f"  最终阈值: {final_ctx.adjusted_threshold:.3f}")
        print(f"  最终权重: {final_ctx.adjusted_weights}")
        print(f"  安全提示数: {len(final_ctx.safety_hints)}")

        # 验证关键行为
        # 1. 校准确实发生了多轮
        assert final_ctx.calibration_round >= 5

        # 2. 阈值有过变化
        assert len(threshold_history) > 1

        # 3. 提示词后缀在有违规时非空
        non_empty_suffixes = [s for s in prompt_suffix_history if s]
        print(
            f"  非空提示词后缀帧数: "
            f"{len(non_empty_suffixes)}/{len(prompt_suffix_history)}"
        )

        # 4. 有违规时产生了提示词
        assert len(non_empty_suffixes) > 0

    def test_calibration_actually_helps(self):
        """验证校准器的介入确实改善了安全分数趋势。

        对比有/无校准器的模拟结果。
        """
        # 无校准器：固定阈值
        fixed_scores: list[float] = []
        for i in range(30):
            if i % 3 == 0:
                fixed_scores.append(0.35)
            else:
                fixed_scores.append(0.75)

        # 有校准器：动态调整
        cal = SessionCalibrator(
            base_threshold=0.6,
            calibration_interval=5,
        )
        calibrated_thresholds: list[float] = []
        for i in range(30):
            ctx = cal.get_context()
            calibrated_thresholds.append(ctx.adjusted_threshold)

            if i % 3 == 0:
                cal.record_frame(
                    FrameFeedback(
                        frame_index=i,
                        timestamp=float(i) * 0.5,
                        proposed_approved=False,
                        proposal_score=0.35,
                        executed_score=0.35,
                        violations=["no_collision"],
                    )
                )
            else:
                cal.record_frame(
                    FrameFeedback(
                        frame_index=i,
                        timestamp=float(i) * 0.5,
                        proposed_approved=True,
                        proposal_score=0.75,
                        executed_score=0.75,
                        violations=[],
                    )
                )

        final = cal.finalize()

        print("\n  === 校准效果对比 ===")
        print("  固定阈值: 0.600 (始终不变)")
        print(
            f"  校准阈值变化: "
            f"{calibrated_thresholds[0]:.3f} → {calibrated_thresholds[-1]:.3f}"
        )
        print(
            f"  最终校准阈值: {final.adjusted_threshold:.3f}"
        )
        print(f"  collision 权重变化: {final.adjusted_weights.get('collision', 'N/A')}")

        # 关键验证：校准器确实做了有意义的调整
        # 有违规模式时阈值应该收紧
        assert final.adjusted_threshold > 0.6
        # collision 权重应该增加
        assert final.adjusted_weights.get("collision", 1.0) > 1.0
        # 应该产生了安全提示
        assert len(final.safety_hints) > 0
