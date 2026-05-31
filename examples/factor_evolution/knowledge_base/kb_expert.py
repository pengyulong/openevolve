"""
Expert Advisor — 专家 LLM 知识注入模块

通过"专家 LLM"角色调用搜索工具获取外部知识（学术论文、市场动态、新因子思路），
将策略建议转化为知识库条目，打破演化的局部最优。

支持两种触发模式：
1. 独立运行（run_expert_advisor.py）：手动触发知识注入
2. 嵌入演化流程：在知识修剪后或演化停滞时自动触发
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .kb_manager import KnowledgeBase
from .kb_embedder import KBEmbedder
from .expert_tools import (
    AcademicPaperTool,
    ExpertToolCache,
    MarketRegimeTool,
    TushareFactorTool,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


class ExpertAdvisor:
    """
    专家 LLM 知识注入器

    主流程：
    1. 分析当前演化状态（best_score, 停滞轮数, 因子类型分布）
    2. Expert LLM 决定搜索策略（市场环境变了？需要新因子类型？瓶颈是什么？）
    3. 调用工具获取外部信息
    4. Expert LLM 将搜索结果转化为结构化知识条目
    5. 写入知识库
    """

    def __init__(
        self,
        config: Dict[str, Any],
        kb: KnowledgeBase,
        embedder: KBEmbedder,
    ):
        """
        Args:
            config: expert_advisor 配置段
            kb: 知识库实例
            embedder: 嵌入向量生成器
        """
        self.config = config
        self.kb = kb
        self.embedder = embedder
        self.max_entries_per_run = config.get("max_entries_per_run", 5)

        # 初始化 LLM 客户端
        self.llm_client = self._init_llm_client(config.get("llm", {}))

        # 初始化缓存
        cache_config = config.get("cache", {})
        cache_db_path = cache_config.get("db_path", "./knowledge_base/expert_cache.db")
        cache_ttl = cache_config.get("ttl_seconds", 604800)
        self.cache = ExpertToolCache(db_path=cache_db_path, default_ttl=cache_ttl)

        # 初始化搜索工具
        self.tools = self._init_tools(config.get("tools", {}))

        # 运行统计
        self._last_run_time: Optional[str] = None
        self._total_entries_injected = 0

    def _init_llm_client(self, llm_config: Dict) -> Optional[Any]:
        try:
            from openai import OpenAI

            api_key = llm_config.get("api_key", "")
            api_base = llm_config.get("api_base", "")
            model = llm_config.get("model", "")

            if not api_key or not api_base:
                logger.warning("Expert LLM not configured (missing api_key or api_base)")
                return None

            client = OpenAI(api_key=api_key, base_url=api_base)
            client.model = model
            return client

        except ImportError:
            logger.warning("openai package not installed")
            return None

    def _init_tools(self, tools_config: Dict) -> Dict[str, Any]:
        tools = {}

        if tools_config.get("market_regime", {}).get("enabled", True):
            tools["market_regime"] = MarketRegimeTool(cache=self.cache)

        if tools_config.get("academic_paper", {}).get("enabled", True):
            max_results = tools_config.get("academic_paper", {}).get("max_results", 5)
            tools["academic_paper"] = AcademicPaperTool(cache=self.cache, max_results=max_results)

        if tools_config.get("web_search", {}).get("enabled", True):
            max_results = tools_config.get("web_search", {}).get("max_results", 3)
            tools["web_search"] = WebSearchTool(cache=self.cache, max_results=max_results)

        if tools_config.get("tushare_factor", {}).get("enabled", True):
            tools["tushare_factor"] = TushareFactorTool(cache=self.cache)

        logger.info(f"Expert Advisor initialized with tools: {list(tools.keys())}")
        return tools

    def run(self, context: Dict[str, Any]) -> List[Dict]:
        """
        执行专家知识注入

        Args:
            context: 演化上下文，包含：
                - best_score: 当前最优分数
                - stagnation_rounds: 连续无改进轮数
                - current_iteration: 当前迭代数
                - best_program_code: 当前最优因子代码
                - factor_categories: 种群中因子类别分布
                - kb_stats: 知识库统计信息
                - topic: (可选) 指定搜索主题

        Returns:
            新注入的知识条目列表
        """
        logger.info("Expert Advisor: starting knowledge injection run")
        start_time = time.time()

        # Step 1: 分析演化状态
        state_analysis = self._analyze_evolution_state(context)
        logger.info(f"State analysis: {json.dumps(state_analysis, ensure_ascii=False)[:200]}")

        # Step 2: 制定搜索策略
        search_plan = self._plan_search_strategy(state_analysis, context)
        logger.info(f"Search plan: {len(search_plan)} searches planned")

        if not search_plan:
            logger.info("Expert Advisor: no searches needed")
            return []

        # Step 3: 执行搜索
        search_results = self._execute_searches(search_plan)
        logger.info(f"Search results: {sum(len(r['results']) for r in search_results)} items total")

        if not search_results or all(len(r["results"]) == 0 for r in search_results):
            logger.info("Expert Advisor: no search results found")
            return []

        # Step 4: 合成知识条目
        entries = self._synthesize_knowledge(search_results, state_analysis)
        logger.info(f"Synthesized {len(entries)} knowledge entries")

        # Step 5: 写入知识库
        written_count = self._write_to_kb(entries)

        elapsed = time.time() - start_time
        self._last_run_time = datetime.now().isoformat()
        self._total_entries_injected += written_count

        logger.info(
            f"Expert Advisor completed: {written_count} entries written "
            f"in {elapsed:.1f}s (total injected: {self._total_entries_injected})"
        )

        return entries[:written_count]

    def _analyze_evolution_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        best_score = context.get("best_score", 0)
        stagnation_rounds = context.get("stagnation_rounds", 0)
        current_iteration = context.get("current_iteration", 0)
        factor_categories = context.get("factor_categories", {})
        kb_stats = context.get("kb_stats", {})

        # 判断停滞程度
        if stagnation_rounds >= 50:
            stagnation_level = "severe"
        elif stagnation_rounds >= 20:
            stagnation_level = "moderate"
        elif stagnation_rounds >= 10:
            stagnation_level = "mild"
        else:
            stagnation_level = "none"

        # 判断因子多样性
        total_factors = sum(factor_categories.values()) if factor_categories else 0
        dominant_category = max(factor_categories, key=factor_categories.get) if factor_categories else "unknown"
        diversity_ratio = len(factor_categories) / max(total_factors, 1) if factor_categories else 0

        # 判断知识库健康度
        kb_active = kb_stats.get("total_entries", 0)
        kb_deprecated = kb_stats.get("deprecated_entries", 0)

        return {
            "best_score": best_score,
            "stagnation_level": stagnation_level,
            "stagnation_rounds": stagnation_rounds,
            "current_iteration": current_iteration,
            "dominant_category": dominant_category,
            "factor_diversity": round(diversity_ratio, 3),
            "factor_categories": factor_categories,
            "kb_active_entries": kb_active,
            "kb_deprecated_entries": kb_deprecated,
            "needs_exploration": stagnation_level in ("moderate", "severe"),
            "needs_diversity": diversity_ratio < 0.3,
        }

    def _plan_search_strategy(
        self, state_analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        # 如果用户指定了主题，直接构造搜索计划
        topic = context.get("topic")
        if topic:
            return self._plan_from_topic(topic)

        # 使用 Expert LLM 制定搜索策略
        if self.llm_client:
            return self._llm_plan_search(state_analysis, context)

        # Fallback: 基于规则的搜索策略
        return self._rule_based_plan(state_analysis)

    def _plan_from_topic(self, topic: str) -> List[Dict[str, str]]:
        plans = []
        plans.append({"tool": "academic_paper", "query": f"quantitative factor {topic} stock market"})
        plans.append({"tool": "web_search", "query": f"量化因子 {topic} A股 策略"})
        plans.append({"tool": "tushare_factor", "query": topic})
        return plans

    def _llm_plan_search(
        self, state_analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        best_code_snippet = (context.get("best_program_code") or "")[:800]

        prompt = f"""你是一个量化因子研究专家。基于当前演化系统的状态，制定搜索策略以注入新知识。

## 当前演化状态
- 最优分数: {state_analysis['best_score']:.3f}
- 停滞程度: {state_analysis['stagnation_level']}（连续{state_analysis['stagnation_rounds']}轮无改进）
- 主导因子类别: {state_analysis['dominant_category']}
- 因子多样性: {state_analysis['factor_diversity']:.2f}
- 知识库活跃条目: {state_analysis['kb_active_entries']}

## 当前最优因子代码片段
```python
{best_code_snippet}
```

## 可用搜索工具
1. academic_paper: 学术论文检索（Semantic Scholar / arXiv）
2. web_search: 网页搜索（量化策略文章、研报）
3. tushare_factor: Tushare 数据接口探索
4. market_regime: 市场环境检测

## 任务
根据当前状态，制定 3-5 个搜索计划。目标是找到能打破当前局部最优的新思路。

请输出 JSON 数组格式（不要其他内容）：
```json
[
  {{"tool": "工具名", "query": "搜索词", "reason": "搜索理由"}},
  ...
]
```"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": "你是量化因子研究专家。只输出JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                plans = json.loads(json_match.group())
                valid_tools = set(self.tools.keys())
                return [
                    {"tool": p["tool"], "query": p["query"]}
                    for p in plans
                    if p.get("tool") in valid_tools and p.get("query")
                ][:5]

        except Exception as e:
            logger.warning(f"LLM search planning failed: {e}, using rule-based fallback")

        return self._rule_based_plan(state_analysis)

    def _rule_based_plan(self, state_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        plans = []

        # 总是检测市场环境
        plans.append({"tool": "market_regime", "query": "current"})

        # 根据停滞程度决定搜索方向
        if state_analysis["needs_exploration"]:
            dominant = state_analysis["dominant_category"]
            # 搜索与当前主导类别不同的因子
            alternative_categories = {
                "value": "momentum factor anomaly",
                "momentum": "quality factor fundamental",
                "volatility": "liquidity factor microstructure",
                "composite": "alternative data factor machine learning",
                "quality": "technical factor market microstructure",
            }
            alt_query = alternative_categories.get(dominant, "novel factor alpha")
            plans.append({"tool": "academic_paper", "query": alt_query})
            plans.append({"tool": "web_search", "query": f"量化因子 {alt_query} 新思路 A股"})

        if state_analysis["needs_diversity"]:
            plans.append({"tool": "tushare_factor", "query": "全部"})
            plans.append({"tool": "web_search", "query": "多因子选股 非线性 机器学习 因子合成"})

        # 默认搜索
        if len(plans) <= 1:
            plans.append({"tool": "academic_paper", "query": "cross-sectional stock return prediction factor"})
            plans.append({"tool": "tushare_factor", "query": "资金流向 筹码"})

        return plans[:5]

    def _execute_searches(self, search_plan: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []

        for plan in search_plan:
            tool_name = plan["tool"]
            query = plan["query"]

            if tool_name not in self.tools:
                logger.warning(f"Tool not available: {tool_name}")
                continue

            try:
                tool = self.tools[tool_name]
                search_results = tool.search(query)
                results.append({
                    "tool": tool_name,
                    "query": query,
                    "results": search_results,
                })
            except Exception as e:
                logger.warning(f"Search failed: {tool_name}/{query}: {e}")
                results.append({"tool": tool_name, "query": query, "results": []})

            time.sleep(0.5)

        return results

    def _synthesize_knowledge(
        self, search_results: List[Dict[str, Any]], state_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if self.llm_client:
            return self._llm_synthesize(search_results, state_analysis)
        return self._template_synthesize(search_results, state_analysis)

    def _llm_synthesize(
        self, search_results: List[Dict[str, Any]], state_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # 格式化搜索结果
        results_text = self._format_search_results(search_results)

        prompt = f"""你是一个量化因子研究专家。基于以下搜索结果，提取可操作的因子改进知识。

## 当前演化状态
- 主导因子类别: {state_analysis['dominant_category']}
- 停滞程度: {state_analysis['stagnation_level']}
- 需要探索新方向: {state_analysis['needs_exploration']}

## 搜索结果
{results_text}

## 任务
将搜索结果转化为 {self.max_entries_per_run} 条结构化知识条目。每条知识应包含：
1. 明确的因子改进思路
2. 可执行的代码方向建议
3. 适用的市场条件

请输出 JSON 数组（不要其他内容）：
```json
[
  {{
    "context_before": "问题场景描述：什么情况下应用这条知识",
    "improvement_action": "改进方法：具体的因子构造/修改思路",
    "improvement_result": "预期效果：基于文献或实践的预期改进",
    "factor_category": "value/momentum/volatility/quality/composite/technical",
    "problem_codes": ["IC_WEAK", "IR_LOW"],
    "tags": ["标签1", "标签2", "标签3"],
    "code_example": "关键代码片段（1-3行）"
  }}
]
```

重要：
- 每条知识必须是可操作的（能直接指导代码修改）
- 避免和现有知识重复（当前主导类别是 {state_analysis['dominant_category']}，尽量提供不同方向）
- 适用于 A 股市场环境"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": "你是量化因子研究专家。只输出JSON数组，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=3000,
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                entries = json.loads(json_match.group())
                return self._validate_entries(entries)

        except Exception as e:
            logger.warning(f"LLM knowledge synthesis failed: {e}, using template")

        return self._template_synthesize(search_results, state_analysis)

    def _template_synthesize(
        self, search_results: List[Dict[str, Any]], state_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """模板方式合成知识（无 LLM fallback）"""
        entries = []

        for result_group in search_results:
            tool = result_group["tool"]
            query = result_group["query"]

            for item in result_group["results"][:2]:
                if item.get("type") == "paper":
                    entries.append({
                        "context_before": f"学术研究表明（来源: {item.get('source', 'unknown')}）",
                        "improvement_action": f"[{item.get('title', '')}] {item.get('content', '')[:200]}",
                        "improvement_result": f"引用数: {item.get('citations', 0)}, 年份: {item.get('year', '')}",
                        "factor_category": self._guess_category_from_query(query),
                        "problem_codes": ["IC_WEAK"] if state_analysis["needs_exploration"] else [],
                        "tags": [tool, query.split()[0] if query else "research"],
                        "code_example": "",
                        "source_url": item.get("url", ""),
                    })
                elif item.get("type") == "tushare_api":
                    entries.append({
                        "context_before": f"可用数据源: {item.get('title', '')}",
                        "improvement_action": f"利用 Tushare {item.get('api_name', '')} 接口获取新数据列: {item.get('content', '')}",
                        "improvement_result": "引入新数据维度，增加因子构造的可能性",
                        "factor_category": self._guess_category_from_query(query),
                        "problem_codes": [],
                        "tags": ["数据源", "tushare", item.get("api_name", "")],
                        "code_example": "",
                    })
                elif item.get("type") == "market_style":
                    entries.append({
                        "context_before": f"市场环境: {item.get('title', '')}",
                        "improvement_action": f"当前市场状态 — {item.get('content', '')}。建议根据市场风格调整因子权重。",
                        "improvement_result": "适应当前市场环境，提高因子有效性",
                        "factor_category": "composite",
                        "problem_codes": [],
                        "tags": ["市场环境", "风格切换"],
                        "code_example": "",
                    })

        return entries[:self.max_entries_per_run]

    def _validate_entries(self, entries: List[Dict]) -> List[Dict[str, Any]]:
        valid = []
        valid_categories = {"value", "momentum", "volatility", "quality", "composite", "technical", "unknown"}

        for entry in entries:
            if not entry.get("improvement_action"):
                continue
            if entry.get("factor_category") not in valid_categories:
                entry["factor_category"] = "unknown"
            if not isinstance(entry.get("problem_codes"), list):
                entry["problem_codes"] = []
            if not isinstance(entry.get("tags"), list):
                entry["tags"] = []
            valid.append(entry)

        return valid[:self.max_entries_per_run]

    def _write_to_kb(self, entries: List[Dict[str, Any]]) -> int:
        written = 0

        for entry in entries:
            try:
                # 构建搜索文本
                search_text = (
                    f"{' '.join(entry.get('problem_codes', []))} "
                    f"{entry.get('factor_category', '')} "
                    f"{entry.get('context_before', '')} "
                    f"{entry.get('improvement_action', '')} "
                    f"{' '.join(entry.get('tags', []))}"
                )

                kb_entry = {
                    "problem_codes": entry.get("problem_codes", []),
                    "factor_category": entry.get("factor_category", "unknown"),
                    "tags": entry.get("tags", []),
                    "market_condition": "all",
                    "context_before": entry.get("context_before", ""),
                    "improvement_action": entry.get("improvement_action", ""),
                    "improvement_result": entry.get("improvement_result", ""),
                    "code_example": entry.get("code_example", ""),
                    "search_text": search_text[:2000],
                    "success_rating": 0.5,
                    "source": "expert_advisor",
                    "source_url": entry.get("source_url", ""),
                    "status": "active",
                }

                entry_id = self.kb.add_entry(kb_entry)

                # 生成嵌入向量
                try:
                    embedding = self.embedder.embed_text(search_text[:1000])
                    self.kb.update_embeddings(entry_id, embedding)
                except Exception as e:
                    logger.warning(f"Embedding generation failed for {entry_id}: {e}")

                written += 1

            except Exception as e:
                logger.warning(f"Failed to write knowledge entry: {e}")

        # 保存嵌入缓存
        if written > 0:
            self.kb._save_embedding_cache()

        return written

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        lines = []
        for group in search_results:
            lines.append(f"\n### 工具: {group['tool']} | 查询: {group['query']}")
            for i, item in enumerate(group["results"][:5]):
                lines.append(f"  {i+1}. [{item.get('type', '')}] {item.get('title', '')}")
                content = item.get("content", "")
                if content:
                    lines.append(f"     {content[:150]}")
        return "\n".join(lines)

    @staticmethod
    def _guess_category_from_query(query: str) -> str:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["value", "价值", "bp", "pe", "pb"]):
            return "value"
        if any(kw in query_lower for kw in ["momentum", "动量", "反转"]):
            return "momentum"
        if any(kw in query_lower for kw in ["volatility", "波动", "vol"]):
            return "volatility"
        if any(kw in query_lower for kw in ["quality", "质量", "roe", "fundamental"]):
            return "quality"
        if any(kw in query_lower for kw in ["technical", "技术", "macd"]):
            return "technical"
        return "composite"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "last_run_time": self._last_run_time,
            "total_entries_injected": self._total_entries_injected,
            "tools_available": list(self.tools.keys()),
            "llm_configured": self.llm_client is not None,
        }
