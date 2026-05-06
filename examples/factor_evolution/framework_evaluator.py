"""
基于 framework 的因子评估器

使用 framework.BaseEvaluator + DiagnosticsEngine 替代原评估器的手动诊断逻辑。
核心能力复用 CrossSectionalICEvaluator，诊断和评分使用框架组件。

这是迁移的标准模式：继承 BaseEvaluator → 实现 evaluate_program() → create_evaluate_function()。
"""

import os
import sys
import importlib.util
import traceback
import logging
import pickle
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from framework.base_evaluator import BaseEvaluator, ScoringConfig, create_evaluate_function
from framework.diagnostics import DiagnosticsEngine

logger = logging.getLogger(__name__)


class FrameworkEvaluator(BaseEvaluator):
    """
    基于框架的因子评估器

    继承 BaseEvaluator，获得：
    - 结构化诊断（P0/P1/P2）
    - 评分配置化
    - LLM 友好的格式化输出
    - MAP-Elites 特征维度计算
    """

    def __init__(self, stock_data, forward_period=5, train_end_date="20231231"):
        # 配置评分（使用默认的 IC 因子评分权重）
        scoring = ScoringConfig(
            metrics={
                "train_ic": {"weight": 0.20, "target": 0.10},
                "train_ir": {"weight": 0.25, "target": 2.00},
                "train_wr": {"weight": 0.10, "target": 1.00},
                "val_ic":   {"weight": 0.25, "target": 0.10},
                "val_ir":   {"weight": 0.20, "target": 2.00},
            },
            penalties=[
                {
                    "name": "direction_mismatch",
                    "weight": 0.30,
                    "condition": "train_ic > 0.01 and val_ic > 0.005 and sign(train_ic) != sign(val_ic)",
                },
            ],
        )

        # 初始化框架诊断引擎（使用内置规则）
        diagnostics = DiagnosticsEngine()

        super().__init__(scoring=scoring, diagnostics=diagnostics)

        # 核心计算引擎（复用 CrossSectionalICEvaluator）
        from evaluator import CrossSectionalICEvaluator
        self._engine = CrossSectionalICEvaluator(
            stock_data=stock_data,
            forward_period=forward_period,
            train_end_date=train_end_date,
        )

    def evaluate_program(self, program_path: str) -> Dict[str, Any]:
        """
        评估单个因子程序

        Args:
            program_path: 因子程序 .py 文件路径

        Returns:
            Dict with combined_score, IC metrics, diagnostics, MAP-Elites features
        """
        # 1. 动态加载因子程序
        try:
            module = self.load_program(program_path)
        except Exception as e:
            return self.empty_result(f"Failed to load program: {e}")

        if not hasattr(module, "compute_factor"):
            return self.empty_result("Missing compute_factor function")

        # 2. 使用 CrossSectionalICEvaluator 计算核心指标
        engine_result = self._engine.evaluate_factor(module.compute_factor)

        # 检查评估是否有效
        if engine_result.get("combined_score", 0) == 0 and "_error" not in engine_result:
            pass  # 可能是真的0分，继续处理

        # 3. 提取指标（兼容框架命名）
        train_ic = engine_result.get("train_rank_ic_mean", 0)
        train_ir = engine_result.get("train_rank_ic_ir", 0)
        val_ic = engine_result.get("val_rank_ic_mean", 0)
        val_ir = engine_result.get("val_rank_ic_ir", 0)
        train_wr = engine_result.get("train_ic_win_rate", 0)
        coverage = engine_result.get("factor_coverage", 1.0)
        auto_flipped = engine_result.get("auto_flipped", False)
        direction_consistent = engine_result.get("direction_consistent", True)
        parent_corr = engine_result.get("parent_corr", 0)
        factor_turnover = engine_result.get("factor_turnover", 0.5)

        # 4. 构建框架指标
        metrics = {
            "train_ic": abs(train_ic),
            "train_ir": abs(train_ir),
            "train_wr": train_wr,
            "val_ic": abs(val_ic),
            "val_ir": abs(val_ir),
            # 保留原始正负号用于诊断
            "train_ic_signed": train_ic,
            "val_ic_signed": val_ic,
        }

        # 5. 使用框架计算综合评分
        combined_score = self.calc_combined_score(metrics)

        # 方向一致性罚分（框架评分公式已处理，这里做二次确认）
        if train_ic > 0.01 and val_ic > 0.005 and (train_ic > 0) != (val_ic > 0):
            # 已在 scoring.penalties 中处理
            pass

        # 6. 使用框架诊断引擎
        diagnostics_input = {
            "train_rank_ic_mean": train_ic,
            "train_rank_ic_ir": train_ir,
            "val_rank_ic_mean": val_ic,
            "val_rank_ic_ir": val_ir,
            "train_ic_win_rate": train_wr,
            "factor_coverage": coverage,
            "auto_flipped": auto_flipped,
            "direction_consistent": direction_consistent,
            "parent_corr": parent_corr,
        }
        formatted_diag = self.format_diagnostics(diagnostics_input)

        # 7. 组装最终结果
        result = {**engine_result}  # 保留原始引擎的所有字段

        # 覆盖/新增框架计算的字段
        result["combined_score"] = combined_score
        result["_diagnostics_prio"] = formatted_diag["diagnostics_prio"]
        result["_parameter_suggestions"] = formatted_diag["parameter_suggestions"]
        result["_ic_trend_table"] = formatted_diag["ic_trend_table"]
        result["_diagnostics_raw"] = {
            "priority_issues": [
                {"code": i.code, "issue": i.issue, "severity": i.severity.value, "reason": i.reason}
                for i in self.diagnostics.diagnose(diagnostics_input).priority_issues
            ],
            "parameter_suggestions": [],
        }
        result["_framework_version"] = "1.0.0"

        # MAP-Elites 特征维度
        result["abs_ic_mean"] = abs(train_ic)
        result["ic_ir"] = abs(train_ir)
        result["ic_stability"] = train_wr
        result["factor_turnover"] = factor_turnover

        # 更新历史（用于趋势计算）
        self.update_history({
            "train_rank_ic_mean": train_ic,
            "val_rank_ic_mean": val_ic,
            "train_rank_ic_ir": train_ir,
            "val_rank_ic_ir": val_ir,
            "train_ic_win_rate": train_wr,
        })
        self.update_best(combined_score)

        return result


# ═══════════════════════════════════════════════════════════
# 创建 OpenEvolve evaluate() 入口
# ═══════════════════════════════════════════════════════════

_evaluator: Optional[FrameworkEvaluator] = None


def _make_evaluator() -> FrameworkEvaluator:
    """工厂函数：创建 FrameworkEvaluator 实例"""
    global _evaluator

    if _evaluator is not None:
        return _evaluator

    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    token = config.get("tushare_token", "")
    start_date = config.get("data_start_date", "20210101")
    end_date = config.get("data_end_date", "20250101")
    max_stocks = config.get("max_stocks", 100)
    forward_period = config.get("forward_period", 5)
    train_end = config.get("train_end_date", "20231231")

    # 从缓存加载
    cache_file = os.path.join(os.path.dirname(__file__), "data_cache", "stock_data_all.pkl")
    if os.path.exists(cache_file):
        print("[FrameworkEvaluator] 从缓存加载股票数据...")
        with open(cache_file, "rb") as f:
            stock_data = pickle.load(f)
        print(f"[FrameworkEvaluator] 缓存加载完成: {len(stock_data)} 只股票")
    else:
        print("[FrameworkEvaluator] 首次运行，下载股票数据...")
        from data_loader import TushareDataLoader
        loader = TushareDataLoader(token=token)
        codes = loader.get_stock_pool("hs300")[:max_stocks]
        stock_data = loader.load_stock_pool_data(codes, start_date, end_date)
        with open(cache_file, "wb") as f:
            pickle.dump(stock_data, f)

    _evaluator = FrameworkEvaluator(
        stock_data=stock_data,
        forward_period=forward_period,
        train_end_date=train_end,
    )
    return _evaluator


# 这是 OpenEvolve 调用的入口函数
evaluate = create_evaluate_function(_make_evaluator)


if __name__ == "__main__":
    # 快速自测
    base_dir = os.path.dirname(os.path.abspath(__file__))
    initial_factor = os.path.join(base_dir, "initial_factor.py")

    print("=== FrameworkEvaluator 自测 ===")
    result = evaluate(initial_factor)

    print("\n=== 评估结果 ===")
    for k, v in sorted(result.items()):
        if k.startswith("_"):
            if k in ("_diagnostics_prio", "_parameter_suggestions", "_ic_trend_table"):
                print(f"\n--- {k} ---")
                print(v)
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, str) and len(v) > 100:
            print(f"  {k}: {v[:100]}...")
