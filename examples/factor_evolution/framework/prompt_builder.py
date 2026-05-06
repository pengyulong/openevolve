"""
提示词构建器

提供模板化的提示词构建系统，支持：
- 从文件加载模板
- 变量替换
- 知识库上下文注入
- 诊断信息集成
- 演化历史追踪

使用方式：
    builder = PromptBuilder(template_dir="prompts/")
    system_msg = builder.build_system_message()
    user_msg = builder.build_user_message(
        current_code=code,
        metrics=metrics,
        kb_context=kb_text,
        diagnostics=diag_text,
    )
"""

import os
import logging
from typing import Dict, Optional, Any, List
from string import Template

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    提示词构建器

    管理提示词模板的加载、变量填充和上下文注入。
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        """
        Args:
            template_dir: 模板文件目录（包含 system_message.txt, diff_user.txt 等）
            system_template: 直接指定 system 模板文本（优先于文件）
            user_template: 直接指定 user 模板文本（优先于文件）
        """
        self.template_dir = template_dir
        self._system_template = system_template
        self._user_template = user_template

    @property
    def system_template(self) -> str:
        """获取 system 模板内容"""
        if self._system_template:
            return self._system_template
        if self.template_dir:
            path = os.path.join(self.template_dir, "system_message.txt")
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
        return ""

    @property
    def user_template(self) -> str:
        """获取 user 模板内容"""
        if self._user_template:
            return self._user_template
        if self.template_dir:
            # 优先使用 diff 格式
            for name in ["diff_user.txt", "full_rewrite_user.txt", "user_message.txt"]:
                path = os.path.join(self.template_dir, name)
                if os.path.exists(path):
                    with open(path, "r") as f:
                        return f.read()
        return ""

    def set_system_template(self, text: str):
        """直接设置 system 模板文本"""
        self._system_template = text

    def set_user_template(self, text: str):
        """直接设置 user 模板文本"""
        self._user_template = text

    def build_system_message(self, **extra_vars) -> str:
        """
        构建 system message

        Args:
            **extra_vars: 额外的模板变量

        Returns:
            填充后的 system message
        """
        template = self.system_template
        if not template:
            return ""
        return self._fill_template(template, extra_vars)

    def build_user_message(
        self,
        current_code: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        kb_context: str = "",
        diagnostics_prio: str = "",
        parameter_suggestions: str = "",
        ic_trend_table: str = "",
        evolution_history: str = "",
        **extra_vars,
    ) -> str:
        """
        构建 user message

        支持以下标准变量（自动从 metrics 中提取）：
        - {current_program}: 当前代码
        - {train_ic_current}, {train_ir_current}, {val_ic_current}: 当前指标
        - {diagnostics_prio}: 格式化诊断优先级列表
        - {parameter_suggestions}: 参数调整建议
        - {ic_trend_table}: IC趋势表格
        - {knowledge_base_context}: 知识库检索结果
        - {evolution_history}: 演化历史

        Args:
            current_code: 当前程序代码
            metrics: 评估指标字典
            kb_context: 知识库检索结果（Markdown格式）
            diagnostics_prio: 诊断优先级文本
            parameter_suggestions: 参数建议文本
            ic_trend_table: IC趋势表格文本
            evolution_history: 演化历史文本
            **extra_vars: 额外的模板变量

        Returns:
            填充后的 user message
        """
        template = self.user_template
        if not template:
            return ""

        # 从 metrics 提取标准变量
        if metrics is None:
            metrics = {}

        vars_dict = dict(extra_vars)

        # 当前程序代码
        vars_dict.setdefault("current_program", current_code)

        # 训练集指标
        vars_dict.setdefault("train_ic_current", self._fmt(
            metrics.get("train_rank_ic_mean", metrics.get("train_ic_mean", 0))))
        vars_dict.setdefault("train_ir_current", self._fmt(
            metrics.get("train_rank_ic_ir", metrics.get("train_ic_ir", 0)), dec=3))
        vars_dict.setdefault("train_win_rate_current", self._fmt(
            metrics.get("train_ic_win_rate", metrics.get("train_win_rate", 0)), dec=1, as_pct=True))

        # 验证集指标
        vars_dict.setdefault("val_ic_current", self._fmt(
            metrics.get("val_rank_ic_mean", metrics.get("val_ic_mean", 0))))
        vars_dict.setdefault("val_ir_current", self._fmt(
            metrics.get("val_rank_ic_ir", metrics.get("val_ic_ir", 0)), dec=3))
        vars_dict.setdefault("val_win_rate_current", self._fmt(
            metrics.get("val_ic_win_rate", metrics.get("val_win_rate", 0)), dec=1, as_pct=True))

        # 综合指标
        vars_dict.setdefault("combined_score", self._fmt(
            metrics.get("combined_score", 0), dec=4))
        vars_dict.setdefault("factor_coverage", self._fmt(
            metrics.get("factor_coverage", metrics.get("coverage", 1.0)), dec=1, as_pct=True))
        vars_dict.setdefault("auto_flipped", str(metrics.get("auto_flipped", False)))
        vars_dict.setdefault("parent_corr", self._fmt(
            metrics.get("parent_corr", 0), dec=3))

        # 诊断和知识库
        vars_dict.setdefault("diagnostics_prio", diagnostics_prio)
        vars_dict.setdefault("parameter_suggestions", parameter_suggestions)
        vars_dict.setdefault("ic_trend_table", ic_trend_table)
        vars_dict.setdefault("knowledge_base_context", kb_context)
        vars_dict.setdefault("evolution_history", evolution_history)

        return self._fill_template(template, vars_dict)

    def _fill_template(self, template: str, vars_dict: Dict[str, Any]) -> str:
        """安全地填充模板变量"""
        # 使用 safe_substitute 避免缺失变量报错
        result = template
        for key, value in vars_dict.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    @staticmethod
    def _fmt(value: float, dec: int = 4, as_pct: bool = False) -> str:
        """格式化数值"""
        if as_pct:
            return f"{value * 100:.{dec}f}%"
        if dec == 3:
            return f"{value:.3f}"
        return f"{value:.4f}"

    @staticmethod
    def build_metrics_table(
        metrics: Dict[str, Any],
        train_prefix: str = "train",
        val_prefix: str = "val",
    ) -> str:
        """
        构建指标表格（Markdown格式）

        Args:
            metrics: 指标字典
            train_prefix: 训练集指标前缀
            val_prefix: 验证集指标前缀

        Returns:
            Markdown表格文本
        """
        lines = []
        lines.append("| 指标 | 训练集 | 验证集 |")
        lines.append("|------|--------|--------|")

        rows = [
            ("Rank IC Mean", f"{train_prefix}_rank_ic_mean", f"{val_prefix}_rank_ic_mean", ".4f"),
            ("Rank IC IR", f"{train_prefix}_rank_ic_ir", f"{val_prefix}_rank_ic_ir", ".3f"),
            ("Pearson IC", f"{train_prefix}_pearson_ic_mean", f"{val_prefix}_pearson_ic_mean", ".4f"),
            ("IC Win Rate", f"{train_prefix}_ic_win_rate", f"{val_prefix}_ic_win_rate", ".1%"),
            ("IC Count", f"{train_prefix}_ic_count", f"{val_prefix}_ic_count", "d"),
        ]

        for name, train_key, val_key, fmt in rows:
            tv = metrics.get(train_key, 0)
            vv = metrics.get(val_key, 0)
            if fmt == ".1%":
                lines.append(f"| {name} | {tv:.1%} | {vv:.1%} |")
            elif fmt == "d":
                lines.append(f"| {name} | {int(tv)} | {int(vv)} |")
            else:
                lines.append(f"| {name} | {tv:{fmt}} | {vv:{fmt}} |")

        # 综合评分
        lines.append(f"| **Combined Score** | **{metrics.get('combined_score', 0):.4f}** | |")
        lines.append(f"| Coverage | {metrics.get('factor_coverage', 1.0):.1%} | |")

        return "\n".join(lines)
