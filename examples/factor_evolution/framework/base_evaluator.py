"""
抽象评估器基类

提供：
- 标准化的评分配置（ScoringConfig）
- 诊断引擎集成
- 指标计算工具方法
- OpenEvolve evaluate() 入口的标准模式

继承方式：
    class MyEvaluator(BaseEvaluator):
        def evaluate_program(self, program_path: str) -> Dict:
            # 实现自己的评估逻辑
            # 调用 self.diagnostics.diagnose(metrics) 获取诊断
            # 调用 self.format_diagnostics(metrics) 格式化输出
            return metrics
"""

import os
import sys
import importlib.util
import traceback
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from .diagnostics import DiagnosticsEngine, DiagnosticsResult, Severity

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """
    综合评分配置

    评分公式（可自定义权重）：
        score = sum(weight_i * min(metric_i / target_i, 1.0)) - penalties

    示例（截面IC因子）：
        ScoringConfig(
            metrics={
                "train_ic":  {"weight": 0.20, "target": 0.10},
                "train_ir":  {"weight": 0.25, "target": 2.00},
                "train_wr":  {"weight": 0.10, "target": 1.00},
                "val_ic":    {"weight": 0.25, "target": 0.10},
                "val_ir":    {"weight": 0.20, "target": 2.00},
            },
            penalties=[
                {"name": "direction_mismatch", "weight": 0.30,
                 "condition": "train_ic > 0.01 and val_ic > 0.005 and sign(train_ic) != sign(val_ic)"},
            ]
        )
    """
    metrics: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "train_ic":  {"weight": 0.20, "target": 0.10},
        "train_ir":  {"weight": 0.25, "target": 2.00},
        "train_wr":  {"weight": 0.10, "target": 1.00},
        "val_ic":    {"weight": 0.25, "target": 0.10},
        "val_ir":    {"weight": 0.20, "target": 2.00},
    })
    penalties: List[Dict[str, Any]] = field(default_factory=list)


class BaseEvaluator(ABC):
    """
    抽象评估器基类

    子类需要实现：
        evaluate_program(program_path) -> Dict[str, Any]

    基类提供：
        - 诊断引擎集成
        - 评分计算
        - 指标统计工具
        - OpenEvolve evaluate() 入口
    """

    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        diagnostics: Optional[DiagnosticsEngine] = None,
    ):
        self.scoring = scoring or ScoringConfig()
        self.diagnostics = diagnostics or DiagnosticsEngine()

        # 缓存：最佳因子面板、上一次指标
        self._best_score: float = 0.0
        self._prev_metrics: Optional[Dict[str, float]] = None

    @abstractmethod
    def evaluate_program(self, program_path: str) -> Dict[str, Any]:
        """
        评估单个程序

        Args:
            program_path: 程序文件路径

        Returns:
            评估结果字典，必须包含 "combined_score"
        """
        ...

    # ── 工具方法 ──

    def load_program(self, program_path: str) -> Any:
        """
        动态加载 Python 程序模块

        Args:
            program_path: .py 文件路径

        Returns:
            加载的模块对象
        """
        spec = importlib.util.spec_from_file_location("program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def calc_ic_stats(
        self, ic_series: pd.Series, prefix: str = ""
    ) -> Dict[str, float]:
        """
        计算IC统计指标

        Args:
            ic_series: IC时间序列
            prefix: 指标名前缀（如 "train_", "val_"）

        Returns:
            {mean, std, ir, win_rate, count} 字典
        """
        if len(ic_series) == 0:
            return {
                f"{prefix}ic_mean": 0.0,
                f"{prefix}ic_std": 0.0,
                f"{prefix}ic_ir": 0.0,
                f"{prefix}ic_win_rate": 0.0,
                f"{prefix}ic_count": 0,
            }

        mean = ic_series.mean()
        std = ic_series.std()
        ir = mean / std if std > 0 else 0.0
        win_rate = (ic_series > 0).sum() / len(ic_series)

        return {
            f"{prefix}ic_mean": float(mean),
            f"{prefix}ic_std": float(std),
            f"{prefix}ic_ir": float(ir),
            f"{prefix}ic_win_rate": float(win_rate),
            f"{prefix}ic_count": int(len(ic_series)),
        }

    def calc_combined_score(self, metrics: Dict[str, float]) -> float:
        """
        根据 ScoringConfig 计算综合评分

        Args:
            metrics: 指标字典

        Returns:
            综合评分 [0, 1]
        """
        score = 0.0

        for metric_name, config in self.scoring.metrics.items():
            value = abs(metrics.get(metric_name, 0))
            weight = config["weight"]
            target = config["target"]
            score += weight * min(value / target, 1.0)

        # 应用罚分
        for penalty in self.scoring.penalties:
            if self._eval_penalty_condition(penalty["condition"], metrics):
                # 罚分基于相关指标
                related_metric = penalty.get("based_on", list(self.scoring.metrics.keys())[0])
                base_val = abs(metrics.get(related_metric, 0))
                base_target = self.scoring.metrics[related_metric]["target"]
                score -= penalty["weight"] * min(base_val / base_target, 1.0)

        return float(min(max(score, 0), 1))

    def _eval_penalty_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """评估罚分条件（安全的简化表达式求值）"""
        try:
            # 构建安全的求值环境
            env = {k: v for k, v in metrics.items()}
            env["abs"] = abs
            env["sign"] = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            return bool(eval(condition, {"__builtins__": {}}, env))
        except Exception:
            return False

    def calc_turnover(self, factor_panel: pd.DataFrame) -> float:
        """
        计算因子换手率（截面排名日间变化率的均值）

        Args:
            factor_panel: 因子截面矩阵 (date × stock)

        Returns:
            换手率 [0, 1]
        """
        try:
            ranks = factor_panel.rank(axis=1, pct=True)
            rank_diff = ranks.diff().abs()
            turnover = rank_diff.mean(axis=1).mean()
            return float(turnover) if not np.isnan(turnover) else 0.5
        except Exception:
            return 0.5

    def calc_cross_sectional_corr(
        self, panel_a: pd.DataFrame, panel_b: pd.DataFrame, sample_step: int = 10
    ) -> float:
        """
        计算两个因子面板的平均截面相关性（Spearman）

        用于亲子相关性检测——判断子代是否保留了父代的有效信号。

        Args:
            panel_a: 因子面板A (date × stock)
            panel_b: 因子面板B (date × stock)
            sample_step: 采样步长（每N天计算一次，避免计算量过大）

        Returns:
            平均截面Spearman相关系数 [-1, 1]
        """
        from scipy import stats

        common_dates = panel_a.index.intersection(panel_b.index)
        common_stocks = panel_a.columns.intersection(panel_b.columns)

        if len(common_dates) == 0 or len(common_stocks) == 0:
            return 0.0

        a = panel_a.loc[common_dates, common_stocks]
        b = panel_b.loc[common_dates, common_stocks]

        corrs = []
        sampled_dates = common_dates[::sample_step]
        for date in sampled_dates:
            a_row = a.loc[date].dropna()
            b_row = b.loc[date].dropna()
            valid = a_row.index.intersection(b_row.index)
            if len(valid) < 20:
                continue
            corr, _ = stats.spearmanr(a_row[valid].values, b_row[valid].values)
            if not np.isnan(corr):
                corrs.append(corr)

        return float(np.mean(corrs)) if corrs else 0.0

    # ── 诊断集成 ──

    def run_diagnostics(self, metrics: Dict[str, Any]) -> DiagnosticsResult:
        """
        执行诊断分析

        Args:
            metrics: 评估指标字典

        Returns:
            结构化诊断结果
        """
        return self.diagnostics.diagnose(metrics)

    def format_diagnostics(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        格式化诊断信息为 LLM 提示词文本

        Args:
            metrics: 评估指标字典

        Returns:
            {diagnostics_prio, parameter_suggestions, ic_trend_table}
        """
        result = self.run_diagnostics(metrics)
        return self.diagnostics.format_for_prompt(result, self._prev_metrics)

    def update_best(self, score: float):
        """更新最佳分数缓存"""
        if score > self._best_score:
            self._best_score = score

    def update_history(self, metrics: Dict[str, float]):
        """更新上一次指标缓存（用于趋势计算）"""
        self._prev_metrics = {
            k: metrics.get(k, 0)
            for k in ["train_ic_mean", "val_ic_mean", "train_ic_ir", "val_ic_ir",
                       "train_rank_ic_mean", "val_rank_ic_mean",
                       "train_rank_ic_ir", "val_rank_ic_ir",
                       "train_ic_win_rate", "train_win_rate"]
        }

    @staticmethod
    def empty_result(error: str = "") -> Dict[str, Any]:
        """返回空结果（评估失败时）"""
        return {
            "combined_score": 0.0,
            "train_ic_mean": 0.0,
            "train_ic_ir": 0.0,
            "train_ic_win_rate": 0.0,
            "val_ic_mean": 0.0,
            "val_ic_ir": 0.0,
            "ic_ir": 0.0,
            "abs_ic_mean": 0.0,
            "ic_stability": 0.0,
            "factor_turnover": 0.5,
            "_error": error,
        }


# ── OpenEvolve evaluate() 入口的通用实现模式 ──

def create_evaluate_function(
    evaluator_factory: callable,
) -> callable:
    """
    创建标准的 OpenEvolve evaluate() 函数

    Args:
        evaluator_factory: 无参数函数，返回 BaseEvaluator 实例

    Returns:
        evaluate(program_path) -> Dict 函数

    使用示例:
        def make_evaluator():
            return MyEvaluator(data=load_data())

        evaluate = create_evaluate_function(make_evaluator)
    """
    _evaluator: Optional[BaseEvaluator] = None

    def get_evaluator() -> BaseEvaluator:
        nonlocal _evaluator
        if _evaluator is None:
            _evaluator = evaluator_factory()
        return _evaluator

    def evaluate(program_path: str) -> Dict[str, Any]:
        try:
            evaluator = get_evaluator()
            result = evaluator.evaluate_program(program_path)

            # 打印关键指标
            score = result.get("combined_score", 0)
            ic = result.get("train_rank_ic_mean", result.get("train_ic_mean", 0))
            ir = result.get("train_rank_ic_ir", result.get("train_ic_ir", 0))
            val_ic = result.get("val_rank_ic_mean", result.get("val_ic_mean", 0))
            auto_flip = result.get("auto_flipped", False)
            flip_tag = " [flipped]" if auto_flip else ""
            print(f"  IC={ic:+.4f}{flip_tag}  IR={ir:+.3f}  Val_IC={val_ic:+.4f}  Score={score:.3f}")

            return result

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"评估失败: {e}\n{tb}")
            result = BaseEvaluator.empty_result(str(e))
            result["_traceback"] = tb
            return result

    return evaluate
