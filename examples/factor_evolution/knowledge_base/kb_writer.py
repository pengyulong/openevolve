"""
知识回写器

在演化过程中检测显著改进，用轻量 LLM 总结改进经验，并写入知识库。
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np

from .kb_manager import KnowledgeBase
from .kb_embedder import KBEmbedder

logger = logging.getLogger(__name__)


class KnowledgeWriter:
    """
    知识回写器

    触发条件：子代 combined_score 比父代提升超过阈值
    处理流程：
    1. 提取父代问题、子代改进内容
    2. 用轻量 LLM 总结改进经验
    3. 生成结构化知识条目并写入 KB
    4. 生成嵌入向量
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: KBEmbedder,
        writeback_threshold: float = 0.15,
        use_llm_summary: bool = True,
        llm_client: Optional[Any] = None,
    ):
        """
        Args:
            kb: 知识库实例
            embedder: 嵌入向量生成器
            writeback_threshold: 回写阈值（combined_score 相对提升比例）
            use_llm_summary: 是否使用 LLM 总结改进经验
            llm_client: LLM 客户端（OpenAI 兼容接口）
        """
        self.kb = kb
        self.embedder = embedder
        self.writeback_threshold = writeback_threshold
        self.use_llm_summary = use_llm_summary
        self.llm_client = llm_client

    def should_writeback(
        self,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
    ) -> bool:
        """
        判断是否应该触发知识回写

        Args:
            parent_metrics: 父代指标
            child_metrics: 子代指标

        Returns:
            是否触发回写
        """
        parent_score = parent_metrics.get("combined_score", 0)
        child_score = child_metrics.get("combined_score", 0)

        if parent_score <= 0 or child_score <= 0:
            return False

        improvement = (child_score - parent_score) / parent_score
        return improvement > self.writeback_threshold

    def extract_knowledge(
        self,
        parent_code: str,
        child_code: str,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
        problem_codes: list,
    ) -> Optional[Dict[str, Any]]:
        """
        从父代-子代对比中提取改进知识

        Args:
            parent_code: 父代因子代码
            child_code: 子代因子代码
            parent_metrics: 父代评估指标
            child_metrics: 子代评估指标
            problem_codes: 父代的问题码列表

        Returns:
            结构化的知识条目，或 None（如果不满足提取条件）
        """
        # 计算改进幅度
        parent_score = parent_metrics.get("combined_score", 0)
        child_score = child_metrics.get("combined_score", 0)

        improvement_ratio = (
            (child_score - parent_score) / parent_score if parent_score > 0 else 0
        )

        # 如果使用 LLM 总结
        if self.use_llm_summary and self.llm_client is not None:
            return self._llm_extract_knowledge(
                parent_code, child_code, parent_metrics, child_metrics,
                problem_codes, improvement_ratio
            )
        else:
            return self._template_extract_knowledge(
                parent_code, child_code, parent_metrics, child_metrics,
                problem_codes, improvement_ratio
            )

    def _template_extract_knowledge(
        self,
        parent_code: str,
        child_code: str,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
        problem_codes: list,
        improvement_ratio: float,
    ) -> Dict[str, Any]:
        """用模板方式提取知识（不需要 LLM）"""
        parent_ic = parent_metrics.get("train_rank_ic_mean", 0)
        child_ic = child_metrics.get("train_rank_ic_mean", 0)
        parent_ir = parent_metrics.get("train_rank_ic_ir", 0)
        child_ir = child_metrics.get("train_rank_ic_ir", 0)
        parent_val_ic = parent_metrics.get("val_rank_ic_mean", 0)
        child_val_ic = child_metrics.get("val_rank_ic_mean", 0)

        # 检测因子类别
        factor_category = self._detect_category(child_code)

        # 提取代码差异的关键行
        child_diff = self._extract_significant_lines(parent_code, child_code)

        # 构建知识内容
        context_before = (
            f"因子train_IC={parent_ic:+.4f}, val_IC={parent_val_ic:+.4f}, "
            f"IR={parent_ir:.3f}, combined_score={parent_metrics.get('combined_score', 0):.3f}"
        )

        improvement_action = f"修改因子计算逻辑，新增/调整以下关键部分：\n{child_diff}"

        improvement_result = (
            f"combined_score {parent_metrics.get('combined_score', 0):.3f} → "
            f"{child_metrics.get('combined_score', 0):.3f} "
            f"(+{improvement_ratio:.0%})。"
            f"train_IC: {parent_ic:+.4f}→{child_ic:+.4f}, "
            f"train_IR: {parent_ir:.3f}→{child_ir:.3f}, "
            f"val_IC: {parent_val_ic:+.4f}→{child_val_ic:+.4f}"
        )

        # 提取 EVOLVE-BLOCK 中的代码作为示例
        code_example = self._extract_evolve_block(child_code)

        # 构建标签
        tags = list(set(self._extract_tags(child_code) + problem_codes))

        # 构建搜索文本
        search_text = (
            f"{' '.join(problem_codes)} {factor_category} {context_before} "
            f"{improvement_action} {improvement_result} {' '.join(tags)}"
        )

        return {
            "problem_codes": problem_codes,
            "factor_category": factor_category,
            "tags": tags,
            "context_before": context_before,
            "improvement_action": improvement_action,
            "improvement_result": improvement_result,
            "code_example": code_example,
            "search_text": search_text[:2000],
            "success_rating": min(improvement_ratio * 5, 1.0),
            "source": "evolution",
            "status": "active",
        }

    def _llm_extract_knowledge(
        self,
        parent_code: str,
        child_code: str,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
        problem_codes: list,
        improvement_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        """
        用轻量 LLM 总结改进经验

        用结构化 prompt 让 LLM 提取：
        1. 父代因子存在什么问题？
        2. 子代做了哪些关键改进？
        3. 改进效果如何量化？
        """
        prompt = f"""你是一个量化因子知识提取助手。请分析以下因子演化中的改进，提取结构化知识。

## 父代因子代码（改进前）
```python
{parent_code[:1500]}
```

## 子代因子代码（改进后）
```python
{child_code[:1500]}
```

## 父代评估指标
- train_rank_ic_mean: {parent_metrics.get('train_rank_ic_mean', 0):+.4f}
- train_rank_ic_ir: {parent_metrics.get('train_rank_ic_ir', 0):.3f}
- val_rank_ic_mean: {parent_metrics.get('val_rank_ic_mean', 0):+.4f}
- combined_score: {parent_metrics.get('combined_score', 0):.4f}
- 问题码: {', '.join(problem_codes) if problem_codes else '无'}

## 子代评估指标
- train_rank_ic_mean: {child_metrics.get('train_rank_ic_mean', 0):+.4f}
- train_rank_ic_ir: {child_metrics.get('train_rank_ic_ir', 0):.3f}
- val_rank_ic_mean: {child_metrics.get('val_rank_ic_mean', 0):+.4f}
- combined_score: {child_metrics.get('combined_score', 0):.4f}
- 改进幅度: +{improvement_ratio:.1%}

请输出 JSON 格式的结构化知识（不要输出其他内容）：

```json
{{
  "context_before": "简短描述改进前的问题和指标（1-2句话）",
  "improvement_action": "描述核心改进思路和方法（2-3句话）",
  "improvement_result": "描述改进效果，包含量化指标变化（1-2句话）",
  "factor_category": "value/momentum/volatility/quality/composite/technical 中的一个",
  "tags": ["标签1", "标签2", "标签3", "标签4"],
  "code_example_max3lines": "最关键的3行改进代码"
}}
```"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": "你是一个精准的量化因子知识提取助手。只输出JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            content = response.choices[0].message.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                extracted = json.loads(json_match.group())

                return {
                    "problem_codes": problem_codes,
                    "factor_category": extracted.get("factor_category", "unknown"),
                    "tags": extracted.get("tags", []),
                    "context_before": extracted.get("context_before", ""),
                    "improvement_action": extracted.get("improvement_action", ""),
                    "improvement_result": extracted.get("improvement_result", ""),
                    "code_example": extracted.get("code_example_max3lines", ""),
                    "search_text": f"{' '.join(problem_codes)} {extracted.get('context_before', '')} {extracted.get('improvement_action', '')}",
                    "success_rating": min(improvement_ratio * 5, 1.0),
                    "source": "evolution",
                    "status": "active",
                }

        except Exception as e:
            logger.warning(f"LLM knowledge extraction failed: {e}, falling back to template")

        return self._template_extract_knowledge(
            parent_code, child_code, parent_metrics, child_metrics,
            problem_codes, improvement_ratio
        )

    def writeback(
        self,
        parent_code: str,
        child_code: str,
        parent_metrics: Dict[str, float],
        child_metrics: Dict[str, float],
        problem_codes: Optional[list] = None,
    ) -> Optional[str]:
        """
        执行完整的知识回写流程

        Args:
            parent_code: 父代代码
            child_code: 子代代码
            parent_metrics: 父代指标
            child_metrics: 子代指标
            problem_codes: 问题码

        Returns:
            新知识的 ID，或 None（如果未触发回写）
        """
        if not self.should_writeback(parent_metrics, child_metrics):
            return None

        parent_score = parent_metrics.get("combined_score", 0)
        child_score = child_metrics.get("combined_score", 0)
        improvement_ratio = (child_score - parent_score) / parent_score if parent_score > 0 else 0

        if problem_codes is None:
            problem_codes = []

        # 先加入回写队列
        queue_id = self.kb.add_to_writeback_queue({
            "parent_code": parent_code,
            "child_code": child_code,
            "parent_metrics": parent_metrics,
            "child_metrics": child_metrics,
            "problem_codes": problem_codes,
            "improvement_ratio": improvement_ratio,
        })

        # 提取知识
        knowledge = self.extract_knowledge(
            parent_code, child_code, parent_metrics, child_metrics, problem_codes
        )

        if knowledge is None:
            self.kb.update_writeback_status(queue_id, "skipped")
            return None

        # 写入知识库
        entry_id = self.kb.add_entry(knowledge)

        # 生成嵌入向量并缓存
        try:
            embedding = self.embedder.embed_text(knowledge.get("search_text", ""))
            self.kb.update_embeddings(entry_id, embedding)
            self.kb._save_embedding_cache()
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {entry_id}: {e}")

        # 更新队列状态
        self.kb.update_writeback_status(
            queue_id, "stored",
            knowledge_json=json.dumps(knowledge, ensure_ascii=False)
        )

        logger.info(
            f"Knowledge writeback: {entry_id} "
            f"(improvement: +{improvement_ratio:.1%}, "
            f"score: {parent_score:.3f} → {child_score:.3f})"
        )

        return entry_id

    # ── 辅助方法 ──

    def _detect_category(self, code: str) -> str:
        """从代码中检测因子类别"""
        code_lower = code.lower()
        if any(kw in code_lower for kw in ["pb", "pe_ttm", "pe ", "bp", "ep"]):
            return "value"
        if any(kw in code_lower for kw in ["pct_change", "momentum", "ret_"]):
            return "momentum"
        if any(kw in code_lower for kw in ["std", "volatility", "vol"]):
            return "volatility"
        if any(kw in code_lower for kw in ["roe", "roa", "gross_margin", "profit"]):
            return "quality"
        if any(kw in code_lower for kw in ["macd", "rsi", "kdj", "boll"]):
            return "technical"
        if "ts_z" in code_lower and "+" in code_lower:
            return "composite"
        return "unknown"

    def _extract_significant_lines(self, parent_code: str, child_code: str) -> str:
        """提取父代到子代的关键变化行"""
        parent_lines = parent_code.strip().split("\n")
        child_lines = child_code.strip().split("\n")

        # 简单 diff：找新增/变化行
        parent_set = set(line.strip() for line in parent_lines if line.strip())
        child_set = set(line.strip() for line in child_lines if line.strip())

        new_lines = child_set - parent_set

        # 过滤：只保留有意义的代码行
        significant = [
            line for line in new_lines
            if not line.startswith("#") and len(line) > 10
        ]

        return "\n".join(significant[:10])

    def _extract_evolve_block(self, code: str) -> str:
        """从代码中提取 EVOLVE-BLOCK 部分"""
        in_block = False
        block_lines = []

        for line in code.split("\n"):
            if "EVOLVE-BLOCK-END" in line:
                break
            if in_block:
                block_lines.append(line)
            if "EVOLVE-BLOCK-START" in line:
                in_block = True

        if block_lines:
            return "\n".join(block_lines[:15])
        return ""

    def _extract_tags(self, code: str) -> list:
        """从代码中提取标签"""
        tags = []
        code_lower = code.lower()

        tag_map = {
            "1/pb": "BP", "1 / pb": "BP", "pb": "PB",
            "pe_ttm": "PE", "ep": "EP",
            "roe": "ROE",
            "pct_change(5)": "短期反转", "pct_change(10)": "短期反转",
            "pct_change(20)": "中期动量", "pct_change(60)": "长期动量",
            "ts_z": "时序标准化",
            "rank": "截面排名",
            "ewm": "指数加权",
            "rolling": "滚动窗口",
            "std()": "波动率",
            "volume": "成交量",
        }

        for kw, tag in tag_map.items():
            if kw in code_lower:
                tags.append(tag)

        return list(set(tags))[:6]
