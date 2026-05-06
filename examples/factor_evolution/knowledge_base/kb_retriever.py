"""
混合检索器

实现规则过滤 + 向量相似度排序的两阶段检索策略。
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .kb_manager import KnowledgeBase
from .kb_embedder import KBEmbedder

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    混合检索器

    检索流程：
    1. 规则过滤：根据 evaluator 诊断码和因子类别缩小候选集
    2. 向量相似度：用 embedding 对候选集排序
    3. 返回 Top-K 结果
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: KBEmbedder,
        top_k: int = 5,
        rule_filter_min_candidates: int = 10,
    ):
        """
        Args:
            kb: 知识库实例
            embedder: 嵌入向量生成器
            top_k: 返回的知识条目数量
            rule_filter_min_candidates: 规则过滤阶段的最小候选数
        """
        self.kb = kb
        self.embedder = embedder
        self.top_k = top_k
        self.rule_filter_min_candidates = rule_filter_min_candidates

    def retrieve(
        self,
        program_metrics: Dict[str, Any],
        program_code: str = "",
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        Args:
            program_metrics: 因子评估指标字典（包含 _diagnostics_raw 等）
            program_code: 当前因子代码（用于辅助判断因子类别）

        Returns:
            检索到的知识条目列表（按相似度排序）
        """
        # Step 0: 从 metrics 提取诊断信息
        problem_codes = self._extract_problem_codes(program_metrics)
        factor_category = self._detect_factor_category(program_code)
        tags = self._extract_tags(program_metrics, program_code)

        # Step 1: 规则过滤
        candidates = self.kb.search_by_tags(
            problem_codes=problem_codes,
            factor_category=factor_category,
            tags=tags,
            limit=50,
        )

        # 如果规则过滤结果太少，放宽条件
        if len(candidates) < self.rule_filter_min_candidates:
            candidates = self.kb.search_by_tags(
                problem_codes=problem_codes,
                limit=50,
            )

        # 如果还是太少，取所有活跃条目
        if len(candidates) < self.top_k:
            candidates = self.kb.get_all_active_entries()

        logger.debug(
            f"Rule filter: {len(candidates)} candidates "
            f"(problem_codes={problem_codes}, category={factor_category})"
        )

        if not candidates:
            return []

        # Step 2: 构建检索查询文本
        query_text = self._build_query_text(program_metrics, problem_codes, factor_category)

        # Step 3: 向量相似度排序
        candidate_ids = [c["id"] for c in candidates]

        # 获取候选条目的嵌入向量
        candidate_embeddings, valid_ids = self.kb.get_embeddings_by_ids(candidate_ids)

        if len(candidate_embeddings) == 0:
            # 没有嵌入向量缓存，只用规则过滤结果
            logger.debug("No embeddings available, returning rule-filtered results")
            return candidates[: self.top_k]

        # 生成查询嵌入
        query_embedding = self.embedder.embed_text(query_text)

        # 计算余弦相似度
        similarities = self._cosine_similarity(query_embedding, candidate_embeddings)

        # 按相似度排序
        ranked_indices = np.argsort(similarities)[::-1]
        top_indices = ranked_indices[: self.top_k]

        # 获取结果
        top_ids = [valid_ids[i] for i in top_indices]
        results = self.kb.get_entries_by_ids(top_ids)

        # 更新使用计数
        for entry_id in top_ids:
            self.kb.update_usage(entry_id)

        logger.info(
            f"Retrieved {len(results)} knowledge entries "
            f"(similarities: {[f'{similarities[i]:.3f}' for i in top_indices]})"
        )

        return results

    def retrieve_and_format(
        self,
        program_metrics: Dict[str, Any],
        program_code: str = "",
    ) -> str:
        """
        检索并格式化为可直接注入 prompt 的文本

        Args:
            program_metrics: 因子评估指标
            program_code: 当前因子代码

        Returns:
            格式化的知识库参考文本
        """
        results = self.retrieve(program_metrics, program_code)

        if not results:
            return "（暂无匹配的历史改进经验）"

        lines = []
        lines.append(f"以下是从知识库中检索到的 {len(results)} 条与当前因子问题最相关的改进案例：\n")

        for i, entry in enumerate(results):
            lines.append(f"### 案例 {i + 1} [相关度: ★★★★★]")
            lines.append(f"**问题场景**：{entry['context_before']}")
            lines.append(f"**改进方法**：{entry['improvement_action']}")
            lines.append(f"**改进效果**：{entry['improvement_result']}")

            if entry.get("code_example"):
                lines.append("")
                lines.append("**参考代码**：")
                lines.append("```python")
                lines.append(entry["code_example"])
                lines.append("```")

            tags_str = ", ".join(entry.get("tags", [])[:5])
            if tags_str:
                lines.append(f"*相关标签: {tags_str}*")
            lines.append("")

        return "\n".join(lines)

    def _extract_problem_codes(self, metrics: Dict[str, Any]) -> List[str]:
        """从评估指标中提取问题类型码"""
        codes = []

        # 从 _diagnostics_raw 提取
        diagnostics = metrics.get("_diagnostics_raw", {})
        if isinstance(diagnostics, dict):
            for issue in diagnostics.get("priority_issues", []):
                code = issue.get("code", "")
                if code and code != "ALL_GOOD":
                    codes.append(code)

        # 如果没有结构化诊断，根据指标数值生成
        if not codes:
            train_ic = abs(metrics.get("train_rank_ic_mean", 0))
            train_ir = abs(metrics.get("train_rank_ic_ir", 0))
            val_ic = abs(metrics.get("val_rank_ic_mean", 0))
            win_rate = metrics.get("train_ic_win_rate", 0)
            coverage = metrics.get("factor_coverage", 1.0)
            auto_flipped = metrics.get("auto_flipped", False)

            if auto_flipped:
                codes.append("AUTO_FLIP")
            if train_ic < 0.02:
                codes.append("IC_WEAK")
            elif train_ic < 0.04:
                codes.append("IC_MODERATE")
            if train_ir < 0.3:
                codes.append("IR_LOW")
            if val_ic < 0.01 and abs(metrics.get("train_rank_ic_mean", 0)) > 0.02:
                codes.append("VAL_IC_WEAK")
            if win_rate < 0.55:
                codes.append("WIN_RATE_LOW")
            if coverage < 0.8:
                codes.append("COVERAGE_LOW")

        return codes

    def _detect_factor_category(self, program_code: str) -> str:
        """从因子代码中检测因子类别"""
        code_lower = program_code.lower()

        # 关键词检测
        categories = {
            "value": ["pb", "pe_ttm", "pe ", "bp", "ep", "book", "earnings", "dividend", "股息"],
            "momentum": ["pct_change", "momentum", "roc", "ret_", "return", "收益", "涨跌幅"],
            "volatility": ["std", "volatility", "vol", "波动", "vix"],
            "quality": ["roe", "roa", "gross_margin", "profit", "quality", "毛利", "净利润"],
            "technical": ["macd", "rsi", "kdj", "boll", "ma_", "sma_", "均线"],
            "composite": ["ts_z", "factor", "composite", "combination", "组合", "合成"],
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in code_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)

        return "unknown"

    def _extract_tags(
        self, metrics: Dict[str, Any], program_code: str
    ) -> List[str]:
        """从指标和代码中提取检索标签"""
        tags = []

        code_lower = program_code.lower()

        # 从代码中检测具体因子
        if "1/pb" in code_lower or "1 / pb" in code_lower or "1/data['pb']" in code_lower:
            tags.append("BP")
        if "1/pe" in code_lower or "1 / pe" in code_lower:
            tags.append("EP")
        if "roe" in code_lower:
            tags.append("ROE")
        if "pct_change(5)" in code_lower or "pct_change(10)" in code_lower:
            tags.append("反转")
        if "pct_change(20)" in code_lower or "pct_change(60)" in code_lower:
            tags.append("动量")
        if "ts_z" in code_lower:
            tags.append("时序标准化")
        if "rank" in code_lower:
            tags.append("截面排名")
        if "+" in code_lower and ("*" in code_lower or "0." in code_lower):
            tags.append("多因子组合")
        if "ewm" in code_lower:
            tags.append("指数加权")
        if "rolling" in code_lower:
            tags.append("滚动窗口")

        # 从指标状态提取标签
        if abs(metrics.get("train_rank_ic_mean", 0)) < 0.03:
            tags.append("IC偏弱")
        if abs(metrics.get("val_rank_ic_mean", 0)) < 0.01:
            tags.append("验证集失效")
        if metrics.get("auto_flipped", False):
            tags.append("方向错误")

        return tags[:8]  # 限制标签数量

    def _build_query_text(
        self,
        metrics: Dict[str, Any],
        problem_codes: List[str],
        factor_category: str,
    ) -> str:
        """构建用于向量检索的查询文本"""
        parts = []

        # 问题描述
        train_ic = metrics.get("train_rank_ic_mean", 0)
        val_ic = metrics.get("val_rank_ic_mean", 0)
        train_ir = metrics.get("train_rank_ic_ir", 0)

        parts.append(f"因子问题: {'、'.join(problem_codes) if problem_codes else '无明显问题'}")
        parts.append(
            f"当前指标: train_IC={train_ic:+.4f}, val_IC={val_ic:+.4f}, train_IR={train_ir:.3f}"
        )

        if factor_category != "unknown":
            parts.append(f"因子类型: {factor_category}")

        # 加入诊断摘要
        diagnostics = metrics.get("_diagnostics_raw", {})
        if isinstance(diagnostics, dict):
            for issue in diagnostics.get("priority_issues", [])[:3]:
                parts.append(issue.get("issue", ""))

        return " ".join(parts)

    @staticmethod
    def _cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
        return np.dot(candidates_norm, query_norm)
