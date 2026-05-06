"""
OpenEvolve 量化因子演化通用框架 v1.0

提供可复用的核心组件：
- DiagnosticsEngine: P0/P1/P2 三级诊断系统
- BaseEvaluator: 抽象评估器基类
- VectorizedBacktester: 向量化截面回测
- PromptBuilder: 模板化提示词构建（含知识库注入）

使用方式：
    1. 复制 framework/ 和 knowledge_base/ 目录到新示例项目
    2. 继承 BaseEvaluator 实现自己的评估器
    3. 配置 config.yaml 中的 knowledge_base 部分
    4. 参考 MIGRATION_GUIDE.md 完成迁移
"""

from .diagnostics import DiagnosticsEngine, DiagnosticIssue, Severity
from .base_evaluator import BaseEvaluator, ScoringConfig
from .backtest import VectorizedBacktester
from .prompt_builder import PromptBuilder

__version__ = "1.0.0"

__all__ = [
    "DiagnosticsEngine",
    "DiagnosticIssue",
    "Severity",
    "BaseEvaluator",
    "ScoringConfig",
    "VectorizedBacktester",
    "PromptBuilder",
]
