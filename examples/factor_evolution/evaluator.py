"""
截面 IC 因子评估器 v3

核心改进：
1. IC方向自动翻转：检测到负IC时自动取反，LLM无需关心方向
2. 最佳因子面板缓存 + 亲子相关性：防止变异破坏原有信号
3. 诊断反馈：输出结构化诊断信息，帮助LLM定位问题

核心逻辑：
1. 对股票池每只股票计算因子值
2. 每个交易日计算截面 IC（因子值 vs 未来N日收益的截面相关系数）
3. 对时间序列 IC 计算 均值、标准差、IR、胜率
4. 分训练集/验证集防过拟合
"""

import os
import sys
import importlib.util
import inspect
import re
import traceback
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


class CrossSectionalICEvaluator:
    """截面 IC 因子评估器"""

    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        forward_period: int = 5,
        train_end_date: str = "20231231",
        min_stocks_per_day: int = 30,
    ):
        self.stock_data = stock_data
        self.forward_period = forward_period
        self.train_end_date = pd.Timestamp(train_end_date)
        self.min_stocks_per_day = min_stocks_per_day
        self.stock_codes = list(stock_data.keys())

        # 预计算未来收益矩阵（只需一次）
        logger.info("构建未来收益矩阵...")
        self.forward_returns = self._build_forward_returns()
        logger.info(f"收益矩阵: {self.forward_returns.shape}")

        # 缓存当前最佳因子面板（用于计算亲子相关性）
        self._best_factor_panel: Optional[pd.DataFrame] = None
        self._best_score: float = 0.0

        # 上一次评估的指标（用于计算趋势）
        self._prev_metrics: Optional[Dict[str, float]] = None

        # P0-1: 代码历史 — 用于同质化检测
        self._code_history: List[str] = []  # 存储已评估因子的归一化代码
        self._max_code_history: int = 50   # 最多保留 50 个历史代码

        # P0-2: 市场状态检测所需数据
        self._market_volatility: float = 0.0  # 最近的市场波动率

    def _build_forward_returns(self) -> pd.DataFrame:
        """构建未来 N 日收益率截面矩阵 (date x stock)"""
        series_dict = {}
        for code, df in self.stock_data.items():
            if "close" in df.columns:
                fwd = df["close"].pct_change(self.forward_period).shift(-self.forward_period)
                series_dict[code] = fwd

        panel = pd.DataFrame(series_dict).sort_index()
        return panel

    def compute_factor_panel(self, compute_func) -> pd.DataFrame:
        """对全股票池计算因子值，返回截面矩阵 (date x stock)"""
        factor_dict = {}
        for code, df in self.stock_data.items():
            try:
                factor_values = compute_func(df)
                if factor_values is not None and isinstance(factor_values, pd.Series):
                    factor_dict[code] = factor_values
            except Exception:
                continue

        if not factor_dict:
            return pd.DataFrame()

        panel = pd.DataFrame(factor_dict).sort_index()
        return panel

    def calculate_cross_sectional_ic(
        self,
        factor_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.Series:
        """计算逐日截面 IC"""
        common_dates = factor_panel.index.intersection(returns_panel.index)
        common_stocks = factor_panel.columns.intersection(returns_panel.columns)

        if len(common_dates) == 0 or len(common_stocks) == 0:
            return pd.Series(dtype=float)

        factor_aligned = factor_panel.loc[common_dates, common_stocks]
        returns_aligned = returns_panel.loc[common_dates, common_stocks]

        ic_series = {}
        for date in common_dates:
            f_row = factor_aligned.loc[date].dropna()
            r_row = returns_aligned.loc[date].dropna()

            valid_stocks = f_row.index.intersection(r_row.index)
            if len(valid_stocks) < self.min_stocks_per_day:
                continue

            f_vals = f_row[valid_stocks].values
            r_vals = r_row[valid_stocks].values

            # 去极端值（winsorize 1%~99%）
            f_vals = np.clip(f_vals, np.percentile(f_vals, 1), np.percentile(f_vals, 99))

            if method == "spearman":
                corr, _ = stats.spearmanr(f_vals, r_vals)
            else:
                corr, _ = stats.pearsonr(f_vals, r_vals)

            if not np.isnan(corr):
                ic_series[date] = corr

        return pd.Series(ic_series).sort_index()

    def _calc_cross_sectional_corr(
        self, panel_a: pd.DataFrame, panel_b: pd.DataFrame
    ) -> float:
        """计算两个因子面板的平均截面Spearman相关性"""
        common_dates = panel_a.index.intersection(panel_b.index)
        common_stocks = panel_a.columns.intersection(panel_b.columns)

        if len(common_dates) == 0 or len(common_stocks) == 0:
            return 0.0

        a = panel_a.loc[common_dates, common_stocks]
        b = panel_b.loc[common_dates, common_stocks]

        corrs = []
        # 每10天采样一次，避免计算量过大
        sampled_dates = common_dates[::10]
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

    def evaluate_factor(self, compute_func) -> Dict[str, Any]:
        """完整评估因子（含自动翻转、亲子相关性、诊断反馈）"""
        # 1. 计算因子面板
        factor_panel = self.compute_factor_panel(compute_func)
        if factor_panel.empty or len(factor_panel) < 60:
            return self._empty_result("因子面板为空或数据不足")

        # 检查因子有效性
        valid_ratio = factor_panel.notna().sum().sum() / (factor_panel.shape[0] * factor_panel.shape[1])
        if valid_ratio < 0.3:
            return self._empty_result(f"因子有效值比例过低: {valid_ratio:.2%}")

        # 检查因子方差
        factor_std = factor_panel.stack().std()
        if factor_std == 0 or np.isnan(factor_std):
            return self._empty_result("因子无方差（常数因子）")

        # 2. 计算截面 IC（Rank IC）—— 先算一次判断方向
        ic_series = self.calculate_cross_sectional_ic(
            factor_panel, self.forward_returns, method="spearman"
        )

        if len(ic_series) < 20:
            return self._empty_result(f"IC 序列太短: {len(ic_series)}")

        # ═══ 改进1：IC方向自动翻转 ═══
        # 如果训练集IC为负，自动对因子取反，LLM无需关心方向
        train_ic_pre = ic_series[ic_series.index <= self.train_end_date]
        auto_flipped = False
        if len(train_ic_pre) > 0 and train_ic_pre.mean() < 0:
            factor_panel = -factor_panel
            ic_series = -ic_series  # IC也翻转
            auto_flipped = True

        # 计算 Pearson IC（用翻转后的因子）
        pearson_ic_series = self.calculate_cross_sectional_ic(
            factor_panel, self.forward_returns, method="pearson"
        )

        # 3. 分训练集/验证集
        train_ic = ic_series[ic_series.index <= self.train_end_date]
        val_ic = ic_series[ic_series.index > self.train_end_date]
        train_pearson = pearson_ic_series[pearson_ic_series.index <= self.train_end_date]
        val_pearson = pearson_ic_series[pearson_ic_series.index > self.train_end_date]

        # 4. 计算指标
        train_metrics = self._calc_ic_stats(train_ic, train_pearson, prefix="train")
        val_metrics = self._calc_ic_stats(val_ic, val_pearson, prefix="val")

        # 5. 市场状态检测 → 动态权重调整系数
        self._regime_mod = self._detect_market_regime(train_ic, val_ic)

        # 6. 综合评分（含动态权重 + 同质化惩罚）
        # 获取源码并检测同质化
        try:
            factor_code = inspect.getsource(compute_func)
        except Exception:
            factor_code = ""
        code_similarity = self._calc_code_similarity(factor_code)
        self._add_to_code_history(factor_code)

        combined_score = self._calc_combined_score(
            train_metrics, val_metrics, code_similarity
        )

        # ═══ Issue-4: 按季度IC分布分析 ═══
        quarterly_ic_train = self._calc_quarterly_ic_breakdown(train_ic)
        quarterly_ic_val = self._calc_quarterly_ic_breakdown(val_ic)
        worst_q_ic = self._calc_worst_quarter_ic(ic_series)  # 全时段最差季度

        # ═══ 高分区额外奖励：鼓励稳定性和单调性 ═══
        if combined_score > 0.5:
            bonus = 0.0
            # 最差季度 IC 奖励
            train_worst_q = self._calc_worst_quarter_ic(train_ic)
            if train_worst_q > 0.02:
                bonus += 0.03
            elif train_worst_q > 0.01:
                bonus += 0.015
            # 分组单调性奖励
            monotonicity = self._calc_monotonicity(factor_panel, self.forward_returns)
            if monotonicity > 0.7:
                bonus += 0.03
            elif monotonicity > 0.5:
                bonus += 0.015
            combined_score = min(combined_score + bonus, 1.0)

        # ═══ Issue-4: 最差季度IC惩罚（全时段，包括验证集） ═══
        if worst_q_ic < -0.02:
            combined_score -= 0.04
        elif worst_q_ic < 0:
            combined_score -= 0.02
        combined_score = max(combined_score, 0.0)

        # ═══ Issue-7: 经济学逻辑验证 ═══
        size_group_stability = self._calc_size_group_ic_stability(
            factor_panel, self.forward_returns
        )
        incremental_alpha = self._calc_incremental_alpha(
            factor_panel, self.forward_returns
        )

        # 增量alpha奖励：如果残差IC高，说明因子提供独立预测能力
        if combined_score > 0.3 and incremental_alpha > 0.03:
            combined_score += min(0.03, incremental_alpha * 0.3)
            combined_score = min(combined_score, 1.0)

        # ═══ 改进2：亲子截面相关性 ═══
        parent_corr = 0.0
        if self._best_factor_panel is not None:
            parent_corr = self._calc_cross_sectional_corr(
                factor_panel, self._best_factor_panel
            )

        # 更新最佳因子面板缓存
        if combined_score > self._best_score:
            self._best_score = combined_score
            self._best_factor_panel = factor_panel.copy()

        # 6. 合并结果
        result = {**train_metrics, **val_metrics}
        result["combined_score"] = combined_score
        result["ic_count"] = len(ic_series)
        result["factor_coverage"] = valid_ratio
        result["auto_flipped"] = auto_flipped
        result["code_similarity"] = code_similarity  # P0-1: 同质化程度
        result["market_regime"] = self._regime_mod.get("regime", "normal")  # P0-2: 市场状态

        # IC方向标记（翻转后应始终为正或接近零）
        t_ic_raw = train_metrics.get("train_rank_ic_mean", 0)
        v_ic_raw = val_metrics.get("val_rank_ic_mean", 0)
        result["ic_direction"] = "positive" if t_ic_raw > 0 else "negative"
        result["direction_consistent"] = (t_ic_raw > 0) == (v_ic_raw > 0) if abs(v_ic_raw) > 0.005 else True

        # 亲子相关性
        result["parent_corr"] = parent_corr

        # Issue-4: 季度IC分布（诊断用）
        result["worst_quarter_ic"] = worst_q_ic
        result["quarterly_ic_train"] = quarterly_ic_train
        result["quarterly_ic_val"] = quarterly_ic_val

        # Issue-7: 经济学逻辑验证指标
        result["size_group_stability"] = size_group_stability
        result["incremental_alpha"] = incremental_alpha

        # MAP-Elites 特征维度
        result["abs_ic_mean"] = abs(train_metrics.get("train_rank_ic_mean", 0))
        result["ic_ir"] = abs(train_metrics.get("train_rank_ic_ir", 0))
        result["ic_stability"] = train_metrics.get("train_ic_win_rate", 0)
        result["factor_turnover"] = self._calc_factor_turnover(factor_panel)
        try:
            import inspect
            source_code = inspect.getsource(compute_func)
        except (OSError, TypeError):
            source_code = ""
        result["factor_type_code"] = self._calc_factor_type_code(source_code)

        # ═══ 改进3：结构化诊断反馈 ═══
        # 生成用于提示词的结构化诊断信息
        formatted_diag = self._format_diagnostics_for_prompt(
            train_metrics, val_metrics, valid_ratio, factor_std,
            auto_flipped, parent_corr, self._prev_metrics
        )

        # 保存原始诊断字典（用于调试）
        result["_diagnostics_raw"] = self._generate_diagnostics(
            train_metrics, val_metrics, valid_ratio, factor_std,
            auto_flipped, parent_corr
        )

        # 保存格式化后的诊断信息（用于提示词）
        result["_diagnostics_prio"] = formatted_diag["diagnostics_prio"]
        result["_parameter_suggestions"] = formatted_diag["parameter_suggestions"]
        result["_ic_trend_table"] = formatted_diag["ic_trend_table"]

        # 更新上一次的指标（用于下次趋势计算）
        self._prev_metrics = {
            "train_rank_ic_mean": t_ic_raw,
            "val_rank_ic_mean": v_ic_raw,
            "train_rank_ic_ir": train_metrics.get("train_rank_ic_ir", 0),
            "val_rank_ic_ir": val_metrics.get("val_rank_ic_ir", 0),
            "train_ic_win_rate": train_metrics.get("train_ic_win_rate", 0),
        }

        # artifacts
        result["_ic_series_head"] = {
            str(k): float(v) for k, v in ic_series.head(50).items()
        }
        result["_factor_stats"] = {
            "mean": float(factor_panel.stack().mean()),
            "std": float(factor_std),
            "coverage": float(valid_ratio),
        }

        return result

    def _generate_diagnostics(
        self,
        train: Dict[str, float],
        val: Dict[str, float],
        coverage: float,
        factor_std: float,
        auto_flipped: bool,
        parent_corr: float,
    ) -> Dict[str, Any]:
        """
        生成结构化诊断反馈，帮助LLM理解问题并改进

        Returns:
            Dict with keys: priority_issues, parameter_suggestions, ic_trend
        """
        p0_issues = []  # 必须修复的问题
        p1_issues = []  # 应该优化的问题
        p2_issues = []  # 可以尝试的改进
        suggestions = []

        t_ic = train.get("train_rank_ic_mean", 0)
        t_ir = train.get("train_rank_ic_ir", 0)
        v_ic = val.get("val_rank_ic_mean", 0)
        v_ir = val.get("val_rank_ic_ir", 0)
        t_wr = train.get("train_ic_win_rate", 0)

        # ═══ P0: 方向/过拟合问题（必须立即修复）═══

        # 方向翻转提示
        if auto_flipped:
            p0_issues.append({
                "code": "AUTO_FLIP",
                "issue": "原始因子IC为负，已自动翻转",
                "severity": "P0",
                "reason": "依赖自动翻转说明因子逻辑方向错误"
            })
            suggestions.append({
                "priority": "P0",
                "target": "因子构造逻辑",
                "current": "使用负IC逻辑",
                "suggestion": "在因子公式中直接取负（如 reversal = -pct_change(5)），不要依赖评估器翻转",
                "example": "bp = 1/data['pb']  # PB本身就是正IC，不需要取负"
            })

        # 训练/验证方向不一致 = 严重过拟合
        if abs(t_ic) > 0.02 and abs(v_ic) > 0.005:
            if (t_ic > 0) != (v_ic > 0):
                p0_issues.append({
                    "code": "DIR_MISMATCH",
                    "issue": f"训练集IC={t_ic:+.4f}与验证集IC={v_ic:+.4f}方向相反",
                    "severity": "P0",
                    "reason": "严重过拟合，验证集泛化失败"
                })
                suggestions.append({
                    "priority": "P0",
                    "target": "因子复杂度",
                    "current": "多子因子组合",
                    "suggestion": "简化因子：减少子因子数量至2-3个，使用更长窗口(30-60日)",
                    "alternative": "考虑使用单一稳健因子（BP/EP/低波动）"
                })

        # ═══ P1: IC强度/稳定性问题（应该优化）═══

        # IC绝对值过弱
        if abs(t_ic) < 0.02:
            p1_issues.append({
                "code": "IC_WEAK",
                "issue": f"训练集IC={t_ic:.4f}，信号极弱（<0.02）",
                "severity": "P1",
                "reason": "当前因子逻辑可能本身无效"
            })
            suggestions.append({
                "priority": "P1",
                "target": "因子类型",
                "current": "当前因子逻辑",
                "suggestion": "尝试完全不同的因子类型",
                "options": [
                    "价值类：BP(1/PB)或EP(1/PE_ttm)",
                    "低波动：-rolling_std(20)",
                    "反转类：-pct_change(5)",
                    "换手率类：-turnover_rate_rolling"
                ]
            })
        elif abs(t_ic) < 0.04:
            p1_issues.append({
                "code": "IC_MODERATE",
                "issue": f"训练集IC={t_ic:.4f}，信号偏弱（0.02-0.04）",
                "severity": "P1",
                "reason": "需要增强信号或组合其他因子"
            })
            suggestions.append({
                "priority": "P1",
                "target": "因子组合",
                "current": "单因子或弱组合",
                "suggestion": "组合2-3个正交弱因子增强信号",
                "example": "factor = 0.4*ts_z(reversal) + 0.3*ts_z(bp) + 0.3*ts_z(low_vol)"
            })

        # IC_IR不稳定
        if abs(t_ir) < 0.3:
            p1_issues.append({
                "code": "IR_LOW",
                "issue": f"IC_IR={t_ir:.3f}，因子不稳定（<0.3）",
                "severity": "P1",
                "reason": "IC波动大，因子信号不稳健"
            })
            suggestions.append({
                "priority": "P1",
                "target": "窗口期/平滑",
                "current": f"窗口可能过短或不均匀",
                "suggestion": "增加滚动窗口至30-60日，或使用指数加权ewm(span=30)",
                "example": "vol = -data['close'].pct_change(1).rolling(30).std()  # 延长窗口"
            })

        # 验证集表现弱
        if abs(v_ic) < 0.01 and abs(t_ic) > 0.02:
            p1_issues.append({
                "code": "VAL_IC_WEAK",
                "issue": f"验证集IC={v_ic:.4f}，样本外几乎无信号",
                "severity": "P1",
                "reason": "因子在2024年失效或过拟合"
            })
            suggestions.append({
                "priority": "P1",
                "target": "因子类型",
                "current": "可能使用了短期技术指标",
                "suggestion": "使用更稳健的基本面因子（BP/EP），简化公式",
                "alternative": "考虑在因子中加入市值中性化处理"
            })

        # ═══ P2: 锦上添花的问题（可以尝试）═══

        # 亲子相关性过低
        if parent_corr != 0 and abs(parent_corr) < 0.3:
            p2_issues.append({
                "code": "PARENT_LOW_CORR",
                "issue": f"与最优因子截面相关性={parent_corr:.2f}，变异幅度过大",
                "severity": "P2",
                "reason": "完全重写而非渐进改进，可能丢失已有优势"
            })
            suggestions.append({
                "priority": "P2",
                "target": "改进方式",
                "current": "大幅重写",
                "suggestion": "在现有因子基础上做渐进式改进",
                "example": "调整现有因子的权重/窗口，而非完全重写"
            })

        # 覆盖率问题
        if coverage < 0.8:
            p2_issues.append({
                "code": "COVERAGE_LOW",
                "issue": f"因子覆盖率={coverage:.0%}，部分股票缺失",
                "severity": "P2",
                "reason": "缺失数据可能导致截面计算偏差"
            })
            suggestions.append({
                "priority": "P2",
                "target": "缺失值处理",
                "current": "存在NaN",
                "suggestion": "对缺失值做合理填充（如用行业均值/中位数填充）",
                "example": "factor = factor.fillna(factor.median())"
            })

        # IC胜率偏低
        if t_wr < 0.55 and abs(t_ic) > 0.02:
            p2_issues.append({
                "code": "WIN_RATE_LOW",
                "issue": f"IC胜率={t_wr:.1%}，低于55%基准",
                "severity": "P2",
                "reason": "因子正确方向的天数不足"
            })
            suggestions.append({
                "priority": "P2",
                "target": "因子平滑度",
                "current": "可能使用了过多短期波动",
                "suggestion": "使用更长窗口或指数加权提高胜率",
                "example": "使用ewm(span=30)替代rolling(20)"
            })

        # 组装结构化结果
        all_issues = p0_issues + p1_issues + p2_issues

        if not all_issues:
            return {
                "priority_issues": [{
                    "code": "ALL_GOOD",
                    "issue": "因子表现良好，无明显问题",
                    "severity": "OK",
                    "reason": ""
                }],
                "parameter_suggestions": [],
                "ic_trend": None
            }

        return {
            "priority_issues": all_issues,
            "parameter_suggestions": suggestions,
            "ic_trend": {
                "train_ic": t_ic,
                "val_ic": v_ic,
                "train_ir": t_ir,
                "val_ir": v_ir,
                "auto_flipped": auto_flipped
            }
        }

    def _format_diagnostics_for_prompt(
        self,
        train: Dict[str, float],
        val: Dict[str, float],
        coverage: float,
        factor_std: float,
        auto_flipped: bool,
        parent_corr: float,
        prev_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        生成用于提示词的结构化诊断信息

        Args:
            prev_metrics: 上一次评估的指标（用于计算趋势）

        Returns:
            Dict with formatted strings for prompt
        """
        diagnostics = self._generate_diagnostics(
            train, val, coverage, factor_std, auto_flipped, parent_corr
        )

        # ═══ 1. 生成优先级问题列表 ═══
        priority_lines = []
        for issue in diagnostics["priority_issues"]:
            severity = issue.get("severity", "")
            code = issue.get("code", "")
            issue_text = issue.get("issue", "")
            reason = issue.get("reason", "")

            if severity == "OK":
                priority_lines.append(f"[✓] {issue_text}")
            elif severity == "P0":
                priority_lines.append(f"[P0] 🚨 {issue_text}")
                if reason:
                    priority_lines.append(f"    → {reason}")
            elif severity == "P1":
                priority_lines.append(f"[P1] ⚠️ {issue_text}")
                if reason:
                    priority_lines.append(f"    → {reason}")
            elif severity == "P2":
                priority_lines.append(f"[P2] 💡 {issue_text}")
                if reason:
                    priority_lines.append(f"    → {reason}")

        # ═══ 2. 生成参数调整建议 ═══
        param_lines = []
        for sug in diagnostics["parameter_suggestions"]:
            priority = sug.get("priority", "")
            target = sug.get("target", "")
            current = sug.get("current", "")
            suggestion = sug.get("suggestion", "")
            example = sug.get("example", "")
            options = sug.get("options", [])

            param_lines.append(f"**[{priority}] {target}**: {suggestion}")
            if example:
                param_lines.append(f"    例如: `{example}`")
            if options:
                for opt in options:
                    param_lines.append(f"    - {opt}")

        # ═══ 3. 生成IC趋势表格 ═══
        trend_lines = []
        if diagnostics["ic_trend"]:
            trend = diagnostics["ic_trend"]
            t_ic = trend["train_ic"]
            v_ic = trend["val_ic"]
            t_ir = trend["train_ir"]
            v_ir = trend["val_ir"]

            # 与上一次比较
            if prev_metrics:
                prev_t_ic = prev_metrics.get("train_rank_ic_mean", 0)
                prev_v_ic = prev_metrics.get("val_rank_ic_mean", 0)
                prev_t_ir = prev_metrics.get("train_rank_ic_ir", 0)
                prev_v_ir = prev_metrics.get("val_rank_ic_ir", 0)

                def trend_arrow(current, previous):
                    if abs(current - previous) < 0.001:
                        return "→"
                    return "↑" if current > previous else "↓"

                def pct_change(current, previous):
                    if abs(previous) < 0.001:
                        return "N/A"
                    return f"{((current - previous) / abs(previous) * 100):+.1f}%"

                trend_lines.append("| 指标 | 当前值 | 上次值 | 趋势 | 变化率 |")
                trend_lines.append("|------|--------|--------|------|--------|")
                trend_lines.append(f"| 训练集IC | {t_ic:+.4f} | {prev_t_ic:+.4f} | {trend_arrow(t_ic, prev_t_ic)} | {pct_change(t_ic, prev_t_ic)} |")
                trend_lines.append(f"| 验证集IC | {v_ic:+.4f} | {prev_v_ic:+.4f} | {trend_arrow(v_ic, prev_v_ic)} | {pct_change(v_ic, prev_v_ic)} |")
                trend_lines.append(f"| 训练集IR | {t_ir:.3f} | {prev_t_ir:.3f} | {trend_arrow(t_ir, prev_t_ir)} | {pct_change(t_ir, prev_t_ir)} |")
                trend_lines.append(f"| 验证集IR | {v_ir:.3f} | {prev_v_ir:.3f} | {trend_arrow(v_ir, prev_v_ir)} | {pct_change(v_ir, prev_v_ir)} |")
            else:
                # 首次评估，没有趋势
                trend_lines.append("（首次评估，无历史趋势）")

        return {
            "diagnostics_prio": "\n".join(priority_lines),
            "parameter_suggestions": "\n".join(param_lines) if param_lines else "无具体建议，继续当前方向优化",
            "ic_trend_table": "\n".join(trend_lines) if trend_lines else "",
        }

    def _calc_ic_stats(
        self, rank_ic: pd.Series, pearson_ic: pd.Series, prefix: str
    ) -> Dict[str, float]:
        """计算 IC 统计指标"""
        if len(rank_ic) == 0:
            return {
                f"{prefix}_rank_ic_mean": 0.0,
                f"{prefix}_rank_ic_std": 0.0,
                f"{prefix}_rank_ic_ir": 0.0,
                f"{prefix}_pearson_ic_mean": 0.0,
                f"{prefix}_ic_win_rate": 0.0,
                f"{prefix}_ic_count": 0,
            }

        rank_mean = rank_ic.mean()
        rank_std = rank_ic.std()
        rank_ir = rank_mean / rank_std if rank_std > 0 else 0.0

        # 翻转后IC应为正，胜率统一按正IC计算
        win_rate = (rank_ic > 0).sum() / len(rank_ic)

        pearson_mean = pearson_ic.mean() if len(pearson_ic) > 0 else 0.0

        return {
            f"{prefix}_rank_ic_mean": float(rank_mean),
            f"{prefix}_rank_ic_std": float(rank_std),
            f"{prefix}_rank_ic_ir": float(rank_ir),
            f"{prefix}_pearson_ic_mean": float(pearson_mean),
            f"{prefix}_ic_win_rate": float(win_rate),
            f"{prefix}_ic_count": int(len(rank_ic)),
        }

    def _calc_code_similarity(self, code: str) -> float:
        """
        P0-1: 计算新因子代码与历史种群中最高相似度

        使用关键字频率向量 + 归一化 n-gram 重叠度。
        返回 0~1 之间的值，1 表示完全同质化。
        """
        if not code or len(self._code_history) == 0:
            return 0.0

        # 归一化：去注释、去空行、统一变量名风格
        def normalize(c: str) -> str:
            c = re.sub(r'#.*', '', c)          # 去注释
            c = re.sub(r'\s+', ' ', c).strip() # 统一空白
            return c

        norm_code = normalize(code)
        if len(norm_code) < 50:
            return 0.0

        # 提取关键字token（操作符、函数调用模式）
        def extract_tokens(c: str) -> set:
            # 提取标识符和操作符模式
            tokens = set(re.findall(r'[a-zA-Z_]\w*|[+\-*/<>]=?|\.\w+', c))
            return tokens

        code_tokens = extract_tokens(norm_code)

        max_similarity = 0.0
        for hist_code in self._code_history[-20:]:  # 只和最接近的 20 个比
            hist_tokens = extract_tokens(hist_code)
            if not code_tokens or not hist_tokens:
                continue
            intersection = len(code_tokens & hist_tokens)
            union = len(code_tokens | hist_tokens)
            if union == 0:
                continue
            jaccard = intersection / union
            max_similarity = max(max_similarity, jaccard)

        return max_similarity

    def _add_to_code_history(self, code: str) -> None:
        """P0-1: 将归一化代码加入历史"""
        if not code:
            return
        norm = re.sub(r'#.*', '', code)
        norm = re.sub(r'\s+', ' ', norm).strip()
        if len(norm) > 50:
            self._code_history.append(norm)
            if len(self._code_history) > self._max_code_history:
                self._code_history = self._code_history[-self._max_code_history:]

    def _detect_market_regime(self, train_ic, val_ic) -> Dict[str, float]:
        """
        P0-2: 检测市场状态，返回动态权重调整系数

        根据 IC 序列的波动特征判断当前市场状态：
        - high_vol: IC 波动大 → 重 IR（稳定性），轻 IC 均值
        - low_vol:  IC 稳定 → 重 IC 均值，鼓励高信号
        - trend:    训验 IC 一致 → 标准权重
        - overfit:  训验 IC 背离 → 重验证集，惩罚过拟合
        """
        if len(train_ic) < 20:
            return {"ic_weight": 1.0, "ir_weight": 1.0, "val_weight": 1.0, "regime": "normal"}

        train_ic_vals = train_ic.dropna()
        val_ic_vals = val_ic.dropna()

        if len(train_ic_vals) < 20:
            return {"ic_weight": 1.0, "ir_weight": 1.0, "val_weight": 1.0, "regime": "normal"}

        # 计算 IC 波动率
        ic_vol = train_ic_vals.std()
        ic_mean = abs(train_ic_vals.mean())
        ic_ir = ic_mean / ic_vol if ic_vol > 0 else 0

        # 训练-验证一致性
        if len(val_ic_vals) > 5:
            train_val_corr = train_ic_vals[-len(val_ic_vals):].corr(
                pd.Series(val_ic_vals.values, index=train_ic_vals.index[-len(val_ic_vals):])
            ) if len(val_ic_vals) <= len(train_ic_vals) else 0
        else:
            train_val_corr = 0

        regime = "normal"
        ic_weight_mod = 1.0
        ir_weight_mod = 1.0
        val_weight_mod = 1.0

        if ic_vol > 0.15:  # 高波动市场
            regime = "high_vol"
            ic_weight_mod = 0.7    # 降 IC 均值权重
            ir_weight_mod = 1.4    # 升 IR 权重（稳定性更重要）
            val_weight_mod = 1.2   # 升验证集权重
        elif ic_ir > 0.8 and ic_vol < 0.08:  # 低波动高信号
            regime = "low_vol_strong"
            ic_weight_mod = 1.3    # 升 IC 权重（信号可靠）
            val_weight_mod = 0.9
        elif train_val_corr < -0.2:  # 训验背离
            regime = "overfit_warning"
            val_weight_mod = 1.5    # 重验证集
            ic_weight_mod = 0.6     # 降训练集权重
            ir_weight_mod = 0.8

        self._market_volatility = ic_vol

        return {
            "ic_weight": ic_weight_mod,
            "ir_weight": ir_weight_mod,
            "val_weight": val_weight_mod,
            "regime": regime,
        }

    @staticmethod
    def _score_ic(ic: float) -> float:
        """IC 评分：低分区线性，高分区对数拉伸"""
        if ic <= 0.08:
            return ic / 0.1
        else:
            import numpy as _np
            return 0.8 + 0.2 * _np.log1p((ic - 0.08) / 0.04) / _np.log1p(3.0)

    @staticmethod
    def _score_ir(ir: float) -> float:
        """IR 评分：低分区线性，高分区对数拉伸"""
        if ir <= 1.5:
            return ir / 2.0
        else:
            import numpy as _np
            return 0.75 + 0.25 * _np.log1p((ir - 1.5) / 0.5) / _np.log1p(3.0)

    def _calc_combined_score(
        self, train: Dict[str, float], val: Dict[str, float],
        code_similarity: float = 0.0,
    ) -> float:
        """
        综合评分 v4: 动态权重 + 同质化惩罚

        P0-1: 代码相似度 > 0.75 时施加同质化惩罚（最多 -0.2）
        P0-2: 根据 IC 序列特征动态调整 IC/IR/验证集权重
        """
        t_ic_raw = train.get("train_rank_ic_mean", 0)
        v_ic_raw = val.get("val_rank_ic_mean", 0)
        t_ic = abs(t_ic_raw)
        t_ir = abs(train.get("train_rank_ic_ir", 0))
        t_wr = train.get("train_ic_win_rate", 0)
        v_ic = abs(v_ic_raw)
        v_ir = abs(val.get("val_rank_ic_ir", 0))

        # P0-2: 动态权重 — 默认权重 × 市场状态调整系数
        # 默认权重通过环境变量或配置覆盖
        default_weights = {
            "train_ic": float(os.environ.get("EVAL_W_TRAIN_IC", "0.20")),
            "train_ir": float(os.environ.get("EVAL_W_TRAIN_IR", "0.25")),
            "train_wr": float(os.environ.get("EVAL_W_TRAIN_WR", "0.10")),
            "val_ic":   float(os.environ.get("EVAL_W_VAL_IC", "0.25")),
            "val_ir":   float(os.environ.get("EVAL_W_VAL_IR", "0.20")),
        }

        # 读取市场状态调整系数（由 evaluate_factor 事先设置）
        regime_mod = getattr(self, '_regime_mod', None)
        if regime_mod is None:
            regime_mod = {"ic_weight": 1.0, "ir_weight": 1.0, "val_weight": 1.0}

        ic_mod = regime_mod.get("ic_weight", 1.0)
        ir_mod = regime_mod.get("ir_weight", 1.0)
        val_mod = regime_mod.get("val_weight", 1.0)

        w_train_ic = default_weights["train_ic"] * ic_mod
        w_train_ir = default_weights["train_ir"] * ir_mod
        w_train_wr = default_weights["train_wr"]
        w_val_ic   = default_weights["val_ic"] * val_mod
        w_val_ir   = default_weights["val_ir"] * val_mod

        # 归一化权重
        total_w = w_train_ic + w_train_ir + w_train_wr + w_val_ic + w_val_ir
        if total_w > 0:
            w_train_ic /= total_w
            w_train_ir /= total_w
            w_train_wr /= total_w
            w_val_ic   /= total_w
            w_val_ir   /= total_w

        base_score = (
            w_train_ic * self._score_ic(t_ic)
            + w_train_ir * self._score_ir(t_ir)
            + w_train_wr * t_wr
            + w_val_ic * self._score_ic(v_ic)
            + w_val_ir * self._score_ir(v_ir)
        )

        # 方向惩罚
        direction_penalty = 0.0
        if t_ic > 0.01 and v_ic > 0.005:
            if (t_ic_raw > 0) != (v_ic_raw > 0):
                direction_penalty = 0.3 * min(t_ic / 0.1, 1.0)

        # P0-1: 同质化惩罚
        similarity_penalty = 0.0
        if code_similarity > 0.65:
            # 相似度 0.65~1.0 映射到惩罚 0~0.25
            similarity_penalty = (code_similarity - 0.65) / 0.35 * 0.25

        score = base_score - direction_penalty - similarity_penalty

        return float(min(max(score, 0), 1))

    @staticmethod
    def _calc_worst_quarter_ic(ic_series: "pd.Series") -> float:
        """计算最差季度的平均 IC"""
        try:
            quarterly = ic_series.resample("QE").mean()
            if len(quarterly) < 2:
                return float(ic_series.mean())
            return float(quarterly.min())
        except Exception:
            return 0.0

    @staticmethod
    def _calc_quarterly_ic_breakdown(ic_series: "pd.Series") -> Dict[str, float]:
        """按季度计算IC均值分布，返回 {quarter_label: ic_mean}"""
        try:
            if len(ic_series) < 5:
                return {}
            quarterly = ic_series.resample("QE").mean().dropna()
            return {f"{k.year}-Q{(k.month-1)//3+1}": float(v)
                    for k, v in quarterly.items()}
        except Exception:
            return {}

    def _calc_monotonicity(
        self, factor_panel: "pd.DataFrame", returns: "pd.DataFrame"
    ) -> float:
        """
        计算因子分组收益的单调性（Spearman rank correlation of group returns）

        将股票按因子值分5组，计算各组平均收益，检查收益是否单调递增。
        返回 0~1 之间的分数，1.0 表示完美单调。
        """
        try:
            common_dates = factor_panel.index.intersection(returns.index)
            if len(common_dates) < 20:
                return 0.5

            mono_scores = []
            for date in common_dates[-60:]:
                fv = factor_panel.loc[date].dropna()
                rv = returns.loc[date].dropna()
                common = fv.index.intersection(rv.index)
                if len(common) < 50:
                    continue

                fv_c = fv[common]
                rv_c = rv[common]

                # 分5组
                try:
                    groups = pd.qcut(fv_c, 5, labels=False, duplicates="drop")
                except Exception:
                    continue

                group_returns = rv_c.groupby(groups).mean()
                if len(group_returns) < 4:
                    continue

                # Spearman correlation between group rank and group return
                from scipy.stats import spearmanr
                corr, _ = spearmanr(range(len(group_returns)), group_returns.values)
                mono_scores.append(abs(corr))

            if not mono_scores:
                return 0.5
            return float(np.mean(mono_scores))
        except Exception:
            return 0.5

    def _calc_factor_turnover(self, factor_panel: pd.DataFrame) -> float:
        """计算因子换手率（截面排名变化率的均值）"""
        try:
            ranks = factor_panel.rank(axis=1, pct=True)
            rank_diff = ranks.diff().abs()
            turnover = rank_diff.mean(axis=1).mean()
            return float(turnover) if not np.isnan(turnover) else 0.5
        except Exception:
            return 0.5

    # ═══ Issue-7: 因子经济学逻辑验证 ═══

    def _calc_size_group_ic_stability(
        self, factor_panel: pd.DataFrame, returns: pd.DataFrame, n_groups: int = 3
    ) -> float:
        """
        按市值分组计算 IC，返回跨组 IC 的稳定性（1 - 变异系数）。
        稳定性高说明因子不依赖特定市值区间，泛化能力强。
        返回 0~1，1 表示各组 IC 完全一致。
        """
        try:
            mv_panel = pd.DataFrame(
                {code: df["total_mv"] for code, df in self.stock_data.items() if "total_mv" in df.columns}
            ).sort_index()

            common_dates = factor_panel.index.intersection(returns.index).intersection(mv_panel.index)
            common_stocks = factor_panel.columns.intersection(returns.columns).intersection(mv_panel.columns)
            if len(common_dates) < 30 or len(common_stocks) < 50:
                return 0.5

            fp = factor_panel.loc[common_dates, common_stocks]
            rp = returns.loc[common_dates, common_stocks]
            mp = mv_panel.loc[common_dates, common_stocks]

            group_ics = {g: [] for g in range(n_groups)}

            sampled_dates = common_dates[::5]
            for date in sampled_dates:
                mv_row = mp.loc[date].dropna()
                f_row = fp.loc[date].dropna()
                r_row = rp.loc[date].dropna()
                valid = mv_row.index.intersection(f_row.index).intersection(r_row.index)
                if len(valid) < 60:
                    continue

                try:
                    mv_groups = pd.qcut(mv_row[valid], n_groups, labels=False, duplicates="drop")
                except Exception:
                    continue

                for g in range(n_groups):
                    stocks_in_g = mv_groups[mv_groups == g].index
                    if len(stocks_in_g) < 15:
                        continue
                    corr, _ = stats.spearmanr(f_row[stocks_in_g].values, r_row[stocks_in_g].values)
                    if not np.isnan(corr):
                        group_ics[g].append(corr)

            mean_ics = [np.mean(v) for v in group_ics.values() if len(v) > 5]
            if len(mean_ics) < 2:
                return 0.5

            ic_std = np.std(mean_ics)
            ic_mean = np.mean(np.abs(mean_ics))
            if ic_mean < 0.001:
                return 0.5

            cv = ic_std / ic_mean
            stability = max(0.0, min(1.0, 1.0 - cv))
            return float(stability)
        except Exception:
            return 0.5

    def _calc_incremental_alpha(
        self, factor_panel: pd.DataFrame, returns: pd.DataFrame
    ) -> float:
        """
        增量 alpha 检验：回归掉 市值因子 + BP因子 后的残差 IC。
        衡量因子提供的独立预测能力。
        返回残差 IC 均值（越高说明增量 alpha 越强）。
        """
        try:
            mv_panel = pd.DataFrame(
                {code: np.log(df["total_mv"]) for code, df in self.stock_data.items() if "total_mv" in df.columns}
            ).sort_index()
            bp_panel = pd.DataFrame(
                {code: 1.0 / df["pb"].replace(0, np.nan) for code, df in self.stock_data.items() if "pb" in df.columns}
            ).sort_index()

            common_dates = (factor_panel.index
                           .intersection(returns.index)
                           .intersection(mv_panel.index)
                           .intersection(bp_panel.index))
            common_stocks = (factor_panel.columns
                            .intersection(returns.columns)
                            .intersection(mv_panel.columns)
                            .intersection(bp_panel.columns))

            if len(common_dates) < 30 or len(common_stocks) < 50:
                return 0.0

            fp = factor_panel.loc[common_dates, common_stocks]
            rp = returns.loc[common_dates, common_stocks]
            mp = mv_panel.loc[common_dates, common_stocks]
            bp = bp_panel.loc[common_dates, common_stocks]

            residual_ics = []
            sampled_dates = common_dates[::5]

            for date in sampled_dates:
                f_row = fp.loc[date].dropna()
                r_row = rp.loc[date].dropna()
                mv_row = mp.loc[date].dropna()
                bp_row = bp.loc[date].dropna()

                valid = f_row.index.intersection(r_row.index).intersection(mv_row.index).intersection(bp_row.index)
                if len(valid) < 50:
                    continue

                y = f_row[valid].values
                X = np.column_stack([mv_row[valid].values, bp_row[valid].values])

                # 标准化
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0)
                X_std[X_std == 0] = 1
                X = (X - X_mean) / X_std

                # OLS 回归取残差
                X_aug = np.column_stack([np.ones(len(valid)), X])
                try:
                    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                    residual = y - X_aug @ beta
                except Exception:
                    continue

                # 残差与收益的 rank IC
                corr, _ = stats.spearmanr(residual, r_row[valid].values)
                if not np.isnan(corr):
                    residual_ics.append(corr)

            if not residual_ics:
                return 0.0
            return float(np.mean(residual_ics))
        except Exception:
            return 0.0

    @staticmethod
    def _calc_factor_type_code(code: str) -> float:
        """
        将因子类型映射为连续值，用于 MAP-Elites 多样性维度。
        混合类型返回加权均值。
        """
        code_lower = code.lower()
        categories = {
            0.1: ["1/pb", "1/pe", "pe_ttm", "pb", "ps_ttm", "dv_ratio", "ep"],
            0.3: ["pct_change", "pct_chg", "returns_", "momentum", "roc"],
            0.5: ["std()", "volatility", "rolling(", ".std(", "amplitude"],
            0.7: ["turnover", "amount", "volume", "vol_ratio", "vol_ma"],
            0.9: ["macd", "rsi", "kdj", "boll", "bb_upper", "ema"],
        }

        weights = {}
        for val, keywords in categories.items():
            count = sum(1 for kw in keywords if kw in code_lower)
            if count > 0:
                weights[val] = count

        if not weights:
            return 0.5

        total = sum(weights.values())
        return sum(v * c for v, c in weights.items()) / total

    @staticmethod
    def _empty_result(error: str) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "combined_score": 0.0,
            "train_rank_ic_mean": 0.0,
            "train_rank_ic_ir": 0.0,
            "train_ic_win_rate": 0.0,
            "val_rank_ic_mean": 0.0,
            "val_rank_ic_ir": 0.0,
            "abs_ic_mean": 0.0,
            "ic_ir": 0.0,
            "ic_stability": 0.0,
            "factor_turnover": 0.5,
            "factor_type_code": 0.5,
            "_error": error,
        }


# ═══════════════════════════════════════════════════════════
# OpenEvolve 评估器入口（evaluate 函数）
# ═══════════════════════════════════════════════════════════

# 全局评估器实例（避免每次评估都重新加载数据）
_evaluator_instance: Optional[CrossSectionalICEvaluator] = None


def _get_evaluator() -> CrossSectionalICEvaluator:
    """获取或创建评估器实例"""
    global _evaluator_instance

    if _evaluator_instance is not None:
        return _evaluator_instance

    from data_loader import TushareDataLoader
    import yaml

    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    token = config.get("tushare_token", "")
    start_date = config.get("data_start_date", "20210101")
    end_date = config.get("data_end_date", "20250101")
    max_stocks = config.get("max_stocks", 100)
    forward_period = config.get("forward_period", 5)
    train_end = config.get("train_end_date", "20231231")

    # 尝试从缓存加载
    cache_file = os.path.join(os.path.dirname(__file__), "data_cache", "stock_data_all.pkl")
    if os.path.exists(cache_file):
        print("从缓存加载股票数据...")
        with open(cache_file, "rb") as f:
            stock_data = pickle.load(f)
        print(f"缓存加载完成: {len(stock_data)} 只股票")
    else:
        print("首次运行，下载股票数据...")
        loader = TushareDataLoader(token=token)
        codes = loader.get_stock_pool("hs300")[:max_stocks]
        stock_data = loader.load_stock_pool_data(codes, start_date, end_date)

        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump(stock_data, f)
        print(f"数据已缓存到 {cache_file}")

    _evaluator_instance = CrossSectionalICEvaluator(
        stock_data=stock_data,
        forward_period=forward_period,
        train_end_date=train_end,
    )

    return _evaluator_instance


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    OpenEvolve 评估入口函数

    Args:
        program_path: 被评估因子程序的文件路径

    Returns:
        Dict[str, float]: 评估指标字典，必须包含 'combined_score'
    """
    try:
        # 动态加载因子程序
        spec = importlib.util.spec_from_file_location("factor_program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "compute_factor"):
            return {"combined_score": 0.0, "_error": "Missing compute_factor function"}

        compute_func = module.compute_factor

        # 获取评估器
        evaluator = _get_evaluator()

        # 评估因子
        result = evaluator.evaluate_factor(compute_func)

        # 打印关键指标
        rank_ic = result.get("train_rank_ic_mean", 0)
        ic_ir = result.get("train_rank_ic_ir", 0)
        val_ic = result.get("val_rank_ic_mean", 0)
        score = result.get("combined_score", 0)
        flipped = result.get("auto_flipped", False)
        p_corr = result.get("parent_corr", 0)
        flip_tag = " [flipped]" if flipped else ""
        print(f"  IC={rank_ic:+.4f}{flip_tag}  IR={ic_ir:+.3f}  Val_IC={val_ic:+.4f}  ParentCorr={p_corr:.2f}  Score={score:.3f}")

        # 打印诊断信息
        diag = result.get("_diagnostics", "")
        if diag and "无明显问题" not in diag:
            for line in diag.split("\n"):
                print(f"    {line}")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"评估失败: {e}\n{tb}")
        return {
            "combined_score": 0.0,
            "train_rank_ic_mean": 0.0,
            "train_rank_ic_ir": 0.0,
            "abs_ic_mean": 0.0,
            "ic_ir": 0.0,
            "ic_stability": 0.0,
            "factor_turnover": 0.5,
            "_error": str(e),
            "_traceback": tb,
        }


if __name__ == "__main__":
    # 测试评估器
    base_dir = os.path.dirname(os.path.abspath(__file__))
    initial_factor_path = os.path.join(base_dir, "initial_factor.py")

    print("=== 评估初始因子 ===")
    result = evaluate(initial_factor_path)

    print("\n=== 评估结果 ===")
    for k, v in sorted(result.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
