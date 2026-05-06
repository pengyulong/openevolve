"""
P0/P1/P2 三级诊断引擎

从 factor_evolution 评估器中提取的通用诊断系统。
可用于任何需要结构化问题诊断和 LLM 反馈的场景。

核心理念：
- P0 (必须修复): 致命问题，如方向错误、严重过拟合
- P1 (应该优化): 性能不足，如信号弱、不稳定
- P2 (可以尝试): 锦上添花，如覆盖率、胜率

每个诊断附带结构化的参数调整建议和代码示例，
可直接注入 LLM 提示词指导改进方向。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class Severity(str, Enum):
    """问题严重级别"""
    P0 = "P0"       # 必须修复 - 致命问题
    P1 = "P1"       # 应该优化 - 性能不足
    P2 = "P2"       # 可以尝试 - 锦上添花
    OK = "OK"       # 无问题


@dataclass
class DiagnosticIssue:
    """单个诊断问题"""
    code: str               # 问题码，如 "IC_WEAK", "AUTO_FLIP"
    issue: str              # 问题描述
    severity: Severity      # 严重级别
    reason: str = ""        # 根因说明


@dataclass
class ParameterSuggestion:
    """参数调整建议"""
    priority: Severity      # 优先级
    target: str             # 调整目标（如 "窗口期/平滑"）
    current: str            # 当前状态
    suggestion: str         # 调整建议
    example: str = ""       # 代码示例
    options: List[str] = field(default_factory=list)  # 备选方案


@dataclass
class DiagnosticsResult:
    """诊断结果"""
    priority_issues: List[DiagnosticIssue] = field(default_factory=list)
    parameter_suggestions: List[ParameterSuggestion] = field(default_factory=list)
    ic_trend: Optional[Dict[str, Any]] = None


class DiagnosticsEngine:
    """
    通用诊断引擎

    使用方式：
    1. 定义诊断规则（通过继承或注册回调）
    2. 调用 diagnose(metrics) 获取结构化诊断结果
    3. 调用 format_for_prompt(result) 生成 LLM 友好的文本

    可扩展：通过 register_rule() 添加自定义诊断规则。
    """

    def __init__(self):
        self._rules: List[callable] = []

    def register_rule(self, rule_func: callable):
        """
        注册自定义诊断规则

        Args:
            rule_func: 函数签名为 (metrics: Dict) -> List[DiagnosticIssue]
        """
        self._rules.append(rule_func)

    def diagnose(self, metrics: Dict[str, Any]) -> DiagnosticsResult:
        """
        执行完整诊断

        Args:
            metrics: 评估指标字典，至少包含 score 相关字段

        Returns:
            DiagnosticsResult 包含分级问题和建议
        """
        issues = []
        suggestions = []

        # 执行内置规则
        builtin_issues, builtin_suggestions = self._run_builtin_rules(metrics)
        issues.extend(builtin_issues)
        suggestions.extend(builtin_suggestions)

        # 执行自定义规则
        for rule in self._rules:
            try:
                result = rule(metrics)
                if isinstance(result, tuple) and len(result) == 2:
                    issues.extend(result[0])
                    suggestions.extend(result[1])
                elif isinstance(result, list):
                    issues.extend(result)
            except Exception:
                continue

        # 如果没有问题，返回 ALL_GOOD
        if not issues:
            issues.append(DiagnosticIssue(
                code="ALL_GOOD",
                issue="表现良好，无明显问题",
                severity=Severity.OK,
            ))

        return DiagnosticsResult(
            priority_issues=issues,
            parameter_suggestions=suggestions,
            ic_trend=self._extract_trend(metrics),
        )

    def _run_builtin_rules(
        self, metrics: Dict[str, Any]
    ) -> tuple:
        """执行内置诊断规则"""
        issues = []
        suggestions = []

        # 提取关键指标（兼容多种命名方式）
        t_ic = metrics.get("train_rank_ic_mean", metrics.get("train_ic_mean", 0))
        t_ir = metrics.get("train_rank_ic_ir", metrics.get("train_ir", 0))
        v_ic = metrics.get("val_rank_ic_mean", metrics.get("val_ic_mean", 0))
        v_ir = metrics.get("val_rank_ic_ir", metrics.get("val_ir", 0))
        t_wr = metrics.get("train_ic_win_rate", metrics.get("train_win_rate", 0))
        coverage = metrics.get("factor_coverage", metrics.get("coverage", 1.0))
        auto_flipped = metrics.get("auto_flipped", False)
        parent_corr = metrics.get("parent_corr", 0)
        direction_consistent = metrics.get("direction_consistent", True)

        # ═══ P0: 致命问题 ═══

        if auto_flipped:
            issues.append(DiagnosticIssue(
                code="AUTO_FLIP",
                issue="原始因子IC为负，已自动翻转",
                severity=Severity.P0,
                reason="依赖自动翻转说明因子逻辑方向错误",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P0,
                target="因子构造逻辑",
                current="使用负IC逻辑",
                suggestion="在因子公式中直接取负，不要依赖评估器翻转",
                example="bp = 1/data['pb']  # PB本身就是正IC，不需要取负",
            ))

        if not direction_consistent and abs(t_ic) > 0.02:
            issues.append(DiagnosticIssue(
                code="DIR_MISMATCH",
                issue=f"训练集IC={t_ic:+.4f}与验证集IC={v_ic:+.4f}方向相反",
                severity=Severity.P0,
                reason="严重过拟合，验证集泛化失败",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P0,
                target="因子复杂度",
                current="多子因子组合",
                suggestion="简化因子：减少子因子数量至2-3个，使用更长窗口(30-60日)",
                options=["考虑使用单一稳健因子（BP/EP/低波动）"],
            ))

        # ═══ P1: 性能不足 ═══

        if abs(t_ic) < 0.02:
            issues.append(DiagnosticIssue(
                code="IC_WEAK",
                issue=f"训练集IC={t_ic:.4f}，信号极弱（<0.02）",
                severity=Severity.P1,
                reason="当前因子逻辑可能本身无效",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P1,
                target="因子类型",
                current="当前因子逻辑",
                suggestion="尝试完全不同的因子类型",
                options=[
                    "价值类：BP(1/PB)或EP(1/PE)",
                    "低波动：-rolling_std(20)",
                    "反转类：-pct_change(5)",
                    "换手率类：-turnover_rate_rolling",
                ],
            ))
        elif abs(t_ic) < 0.04:
            issues.append(DiagnosticIssue(
                code="IC_MODERATE",
                issue=f"训练集IC={t_ic:.4f}，信号偏弱（0.02-0.04）",
                severity=Severity.P1,
                reason="需要增强信号或组合其他因子",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P1,
                target="因子组合",
                current="单因子或弱组合",
                suggestion="组合2-3个正交弱因子增强信号",
                example="factor = 0.4*reversal + 0.3*bp + 0.3*low_vol",
            ))

        if abs(t_ir) < 0.3:
            issues.append(DiagnosticIssue(
                code="IR_LOW",
                issue=f"IC_IR={t_ir:.3f}，因子不稳定（<0.3）",
                severity=Severity.P1,
                reason="IC波动大，因子信号不稳健",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P1,
                target="窗口期/平滑",
                current="窗口可能过短或不均匀",
                suggestion="增加滚动窗口至30-60日，或使用指数加权ewm(span=30)",
                example="vol = -data['close'].pct_change(1).rolling(30).std()",
            ))

        if abs(v_ic) < 0.01 and abs(t_ic) > 0.02:
            issues.append(DiagnosticIssue(
                code="VAL_IC_WEAK",
                issue=f"验证集IC={v_ic:.4f}，样本外几乎无信号",
                severity=Severity.P1,
                reason="因子验证集失效或过拟合",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P1,
                target="因子类型",
                current="可能使用了短期技术指标",
                suggestion="使用更稳健的基本面因子（BP/EP），简化公式",
                options=["考虑在因子中加入市值中性化处理"],
            ))

        # ═══ P2: 锦上添花 ═══

        if parent_corr != 0 and abs(parent_corr) < 0.3:
            issues.append(DiagnosticIssue(
                code="PARENT_LOW_CORR",
                issue=f"与最优因子相关性={parent_corr:.2f}，变异幅度过大",
                severity=Severity.P2,
                reason="完全重写而非渐进改进，可能丢失已有优势",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P2,
                target="改进方式",
                current="大幅重写",
                suggestion="在现有因子基础上做渐进式改进",
                example="调整现有因子的权重/窗口，而非完全重写",
            ))

        if coverage < 0.8:
            issues.append(DiagnosticIssue(
                code="COVERAGE_LOW",
                issue=f"因子覆盖率={coverage:.0%}，部分股票缺失",
                severity=Severity.P2,
                reason="缺失数据可能导致截面计算偏差",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P2,
                target="缺失值处理",
                current="存在NaN",
                suggestion="对缺失值做合理填充（如用行业均值/中位数填充）",
                example="factor = factor.fillna(factor.median())",
            ))

        if t_wr < 0.55 and abs(t_ic) > 0.02:
            issues.append(DiagnosticIssue(
                code="WIN_RATE_LOW",
                issue=f"IC胜率={t_wr:.1%}，低于55%基准",
                severity=Severity.P2,
                reason="因子正确方向的天数不足",
            ))
            suggestions.append(ParameterSuggestion(
                priority=Severity.P2,
                target="因子平滑度",
                current="可能使用了过多短期波动",
                suggestion="使用更长窗口或指数加权提高胜率",
                example="使用ewm(span=30)替代rolling(20)",
            ))

        return issues, suggestions

    def _extract_trend(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """提取IC趋势信息"""
        return {
            "train_ic": metrics.get("train_rank_ic_mean", metrics.get("train_ic_mean", 0)),
            "val_ic": metrics.get("val_rank_ic_mean", metrics.get("val_ic_mean", 0)),
            "train_ir": metrics.get("train_rank_ic_ir", metrics.get("train_ir", 0)),
            "val_ir": metrics.get("val_rank_ic_ir", metrics.get("val_ir", 0)),
        }

    def format_for_prompt(
        self,
        result: DiagnosticsResult,
        prev_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        将诊断结果格式化为 LLM 提示词文本

        Args:
            result: 诊断结果
            prev_metrics: 上一次评估的指标（用于计算趋势）

        Returns:
            Dict with keys: diagnostics_prio, parameter_suggestions, ic_trend_table
        """
        # 1. 优先级问题列表
        priority_lines = []
        for issue in result.priority_issues:
            if issue.severity == Severity.OK:
                priority_lines.append(f"[✓] {issue.issue}")
            elif issue.severity == Severity.P0:
                priority_lines.append(f"[P0] 🚨 {issue.issue}")
                if issue.reason:
                    priority_lines.append(f"    → {issue.reason}")
            elif issue.severity == Severity.P1:
                priority_lines.append(f"[P1] ⚠️ {issue.issue}")
                if issue.reason:
                    priority_lines.append(f"    → {issue.reason}")
            elif issue.severity == Severity.P2:
                priority_lines.append(f"[P2] 💡 {issue.issue}")
                if issue.reason:
                    priority_lines.append(f"    → {issue.reason}")

        # 2. 参数调整建议
        param_lines = []
        for sug in result.parameter_suggestions:
            param_lines.append(f"**[{sug.priority.value}] {sug.target}**: {sug.suggestion}")
            if sug.example:
                param_lines.append(f"    例如: `{sug.example}`")
            for opt in sug.options:
                param_lines.append(f"    - {opt}")

        # 3. IC趋势表格
        trend_lines = []
        if result.ic_trend:
            t_ic = result.ic_trend["train_ic"]
            v_ic = result.ic_trend["val_ic"]
            t_ir = result.ic_trend["train_ir"]
            v_ir = result.ic_trend["val_ir"]

            if prev_metrics:
                prev_t_ic = prev_metrics.get("train_rank_ic_mean", prev_metrics.get("train_ic_mean", 0))
                prev_v_ic = prev_metrics.get("val_rank_ic_mean", prev_metrics.get("val_ic_mean", 0))
                prev_t_ir = prev_metrics.get("train_rank_ic_ir", prev_metrics.get("train_ir", 0))
                prev_v_ir = prev_metrics.get("val_rank_ic_ir", prev_metrics.get("val_ir", 0))

                def trend_arrow(cur, prev):
                    if abs(cur - prev) < 0.001:
                        return "→"
                    return "↑" if cur > prev else "↓"

                def pct_change(cur, prev):
                    if abs(prev) < 0.001:
                        return "N/A"
                    return f"{((cur - prev) / abs(prev) * 100):+.1f}%"

                trend_lines.append("| 指标 | 当前值 | 上次值 | 趋势 | 变化率 |")
                trend_lines.append("|------|--------|--------|------|--------|")
                trend_lines.append(
                    f"| 训练集IC | {t_ic:+.4f} | {prev_t_ic:+.4f} "
                    f"| {trend_arrow(t_ic, prev_t_ic)} | {pct_change(t_ic, prev_t_ic)} |"
                )
                trend_lines.append(
                    f"| 验证集IC | {v_ic:+.4f} | {prev_v_ic:+.4f} "
                    f"| {trend_arrow(v_ic, prev_v_ic)} | {pct_change(v_ic, prev_v_ic)} |"
                )
                trend_lines.append(
                    f"| 训练集IR | {t_ir:.3f} | {prev_t_ir:.3f} "
                    f"| {trend_arrow(t_ir, prev_t_ir)} | {pct_change(t_ir, prev_t_ir)} |"
                )
                trend_lines.append(
                    f"| 验证集IR | {v_ir:.3f} | {prev_v_ir:.3f} "
                    f"| {trend_arrow(v_ir, prev_v_ir)} | {pct_change(v_ir, prev_v_ir)} |"
                )
            else:
                trend_lines.append("（首次评估，无历史趋势）")

        return {
            "diagnostics_prio": "\n".join(priority_lines),
            "parameter_suggestions": "\n".join(param_lines) if param_lines else "无具体建议，继续当前方向优化",
            "ic_trend_table": "\n".join(trend_lines) if trend_lines else "",
        }

    def extract_problem_codes(self, metrics: Dict[str, Any]) -> List[str]:
        """从指标中提取问题码列表（用于知识库检索）"""
        result = self.diagnose(metrics)
        codes = []
        for issue in result.priority_issues:
            if issue.code and issue.code != "ALL_GOOD":
                codes.append(issue.code)
        return codes
