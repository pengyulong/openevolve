"""
Expert Advisor 搜索工具集

提供多种外部信息检索工具，每个工具统一实现 search(query) -> List[Dict] 接口。
所有搜索结果通过 ExpertToolCache 缓存到 SQLite，避免重复请求。
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExpertToolCache:
    """搜索结果 SQLite 缓存层"""

    def __init__(self, db_path: str, default_ttl: int = 604800):
        """
        Args:
            db_path: SQLite 缓存数据库路径
            default_ttl: 默认缓存过期时间（秒），默认 7 天
        """
        self.db_path = db_path
        self.default_ttl = default_ttl
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                query_text TEXT NOT NULL,
                results_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                UNIQUE(tool_name, query_hash)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup
            ON search_cache(tool_name, query_hash)
        """)
        conn.commit()
        conn.close()

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, tool_name: str, query: str) -> Optional[List[Dict]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_hash = self._hash_query(query)
        now = datetime.now().isoformat()

        cursor.execute(
            """
            SELECT results_json FROM search_cache
            WHERE tool_name = ? AND query_hash = ? AND expires_at > ?
            """,
            (tool_name, query_hash, now),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            logger.debug(f"Cache hit: {tool_name}/{query[:30]}...")
            return json.loads(row[0])
        return None

    def set(self, tool_name: str, query: str, results: List[Dict], ttl: Optional[int] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_hash = self._hash_query(query)
        now = datetime.now()
        expires = now + timedelta(seconds=ttl or self.default_ttl)

        cursor.execute(
            """
            INSERT OR REPLACE INTO search_cache
            (tool_name, query_hash, query_text, results_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                query_hash,
                query[:500],
                json.dumps(results, ensure_ascii=False),
                now.isoformat(),
                expires.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Cache set: {tool_name}/{query[:30]}...")

    def clear_expired(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("DELETE FROM search_cache WHERE expires_at <= ?", (now,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted > 0:
            logger.info(f"Cleared {deleted} expired cache entries")


class MarketRegimeTool:
    """
    市场环境检测工具

    通过 AKShare 获取市场指数数据，计算当前市场状态：
    - 大盘/小盘偏好（沪深300 vs 中证1000）
    - 价值/成长偏好
    - 波动率水平
    """

    TOOL_NAME = "market_regime"

    def __init__(self, cache: Optional[ExpertToolCache] = None):
        self.cache = cache

    def search(self, query: str = "current") -> List[Dict]:
        if self.cache:
            cached = self.cache.get(self.TOOL_NAME, query)
            if cached is not None:
                return cached

        results = self._fetch_market_data()

        if self.cache and results:
            self.cache.set(self.TOOL_NAME, query, results, ttl=86400)

        return results

    def _fetch_market_data(self) -> List[Dict]:
        try:
            import akshare as ak

            results = []

            # 沪深300 vs 中证1000 近60日涨跌幅比较
            try:
                hs300 = ak.stock_zh_index_daily(symbol="sh000300")
                zz1000 = ak.stock_zh_index_daily(symbol="sh000852")

                if len(hs300) >= 60 and len(zz1000) >= 60:
                    hs300_ret = (hs300["close"].iloc[-1] / hs300["close"].iloc[-60] - 1) * 100
                    zz1000_ret = (zz1000["close"].iloc[-1] / zz1000["close"].iloc[-60] - 1) * 100

                    if hs300_ret > zz1000_ret + 3:
                        style = "大盘占优"
                    elif zz1000_ret > hs300_ret + 3:
                        style = "小盘占优"
                    else:
                        style = "均衡"

                    results.append({
                        "type": "market_style",
                        "title": f"市值风格: {style}",
                        "content": f"近60日沪深300涨幅{hs300_ret:.1f}%，中证1000涨幅{zz1000_ret:.1f}%",
                        "data": {"hs300_60d": round(hs300_ret, 2), "zz1000_60d": round(zz1000_ret, 2)},
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch index data: {e}")

            # 波动率水平
            try:
                hs300 = ak.stock_zh_index_daily(symbol="sh000300")
                if len(hs300) >= 20:
                    import numpy as np
                    returns = hs300["close"].pct_change().dropna().tail(20)
                    vol = float(np.std(returns) * (252 ** 0.5) * 100)

                    if vol > 25:
                        vol_level = "高波动"
                    elif vol > 15:
                        vol_level = "中等波动"
                    else:
                        vol_level = "低波动"

                    results.append({
                        "type": "volatility",
                        "title": f"波动率水平: {vol_level}",
                        "content": f"沪深300近20日年化波动率{vol:.1f}%",
                        "data": {"annualized_vol": round(vol, 2)},
                    })
            except Exception as e:
                logger.warning(f"Failed to compute volatility: {e}")

            if not results:
                results.append({
                    "type": "market_style",
                    "title": "市场数据获取失败",
                    "content": "AKShare 数据源暂时不可用",
                    "data": {},
                })

            return results

        except ImportError:
            logger.warning("akshare not installed, market regime detection unavailable")
            return [{
                "type": "error",
                "title": "依赖缺失",
                "content": "需要安装 akshare: pip install akshare",
                "data": {},
            }]


class AcademicPaperTool:
    """
    学术论文检索工具

    通过 Semantic Scholar API 和 arXiv 检索量化因子相关论文。
    """

    TOOL_NAME = "academic_paper"

    def __init__(self, cache: Optional[ExpertToolCache] = None, max_results: int = 5):
        self.cache = cache
        self.max_results = max_results

    def search(self, query: str) -> List[Dict]:
        if self.cache:
            cached = self.cache.get(self.TOOL_NAME, query)
            if cached is not None:
                return cached

        results = self._search_semantic_scholar(query)

        if not results:
            results = self._search_arxiv(query)

        if self.cache and results:
            self.cache.set(self.TOOL_NAME, query, results)

        return results

    def _search_semantic_scholar(self, query: str) -> List[Dict]:
        try:
            import requests

            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": self.max_results,
                "fields": "title,abstract,year,citationCount,url",
            }

            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Semantic Scholar API returned {resp.status_code}")
                return []

            data = resp.json()
            results = []
            for paper in data.get("data", []):
                results.append({
                    "type": "paper",
                    "title": paper.get("title", ""),
                    "content": (paper.get("abstract") or "")[:500],
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "url": paper.get("url", ""),
                    "source": "semantic_scholar",
                })

            return results

        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
            return []

    def _search_arxiv(self, query: str) -> List[Dict]:
        try:
            import requests
            import xml.etree.ElementTree as ET

            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "relevance",
            }

            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                link = entry.find("atom:id", ns)

                results.append({
                    "type": "paper",
                    "title": title.text.strip() if title is not None else "",
                    "content": (summary.text.strip() if summary is not None else "")[:500],
                    "year": published.text[:4] if published is not None else "",
                    "url": link.text if link is not None else "",
                    "source": "arxiv",
                })

            return results

        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return []


class WebSearchTool:
    """
    通用网页搜索工具

    支持百度搜索。用于检索量化因子策略文章、研报摘要等。
    """

    TOOL_NAME = "web_search"

    def __init__(self, cache: Optional[ExpertToolCache] = None, max_results: int = 3):
        self.cache = cache
        self.max_results = max_results

    def search(self, query: str) -> List[Dict]:
        if self.cache:
            cached = self.cache.get(self.TOOL_NAME, query)
            if cached is not None:
                return cached

        results = self._search_baidu(query)

        if self.cache and results:
            self.cache.set(self.TOOL_NAME, query, results)

        return results

    def _search_baidu(self, query: str) -> List[Dict]:
        try:
            import requests
            from urllib.parse import quote

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            url = f"https://www.baidu.com/s?wd={quote(query)}&rn={self.max_results}"
            resp = requests.get(url, headers=headers, timeout=15)

            if resp.status_code != 200:
                return []

            # 简单解析搜索结果
            from html.parser import HTMLParser

            results = []
            content = resp.text

            # 提取搜索结果标题和摘要（简化版解析）
            import re

            # 匹配百度搜索结果的标题
            title_pattern = re.compile(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', re.DOTALL)
            titles = title_pattern.findall(content)

            for i, title in enumerate(titles[:self.max_results]):
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                if clean_title:
                    results.append({
                        "type": "web",
                        "title": clean_title,
                        "content": "",
                        "url": "",
                        "source": "baidu",
                    })

            return results

        except Exception as e:
            logger.warning(f"Baidu search failed: {e}")
            return []


class TushareFactorTool:
    """
    Tushare 因子数据探索工具

    查询 Tushare Pro 可用的因子/数据接口，帮助发现新的数据源。
    """

    TOOL_NAME = "tushare_factor"

    # Tushare Pro 量化相关接口列表（静态知识）
    AVAILABLE_FACTORS = [
        {"api": "daily_basic", "fields": "turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv", "desc": "每日基本面指标"},
        {"api": "adj_factor", "fields": "adj_factor", "desc": "复权因子"},
        {"api": "fina_indicator", "fields": "roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets,op_income_growth,net_profit_growth", "desc": "财务指标"},
        {"api": "income", "fields": "revenue,operate_profit,total_profit,n_income", "desc": "利润表"},
        {"api": "balancesheet", "fields": "total_assets,total_liab,total_hldr_eqy_exc_min_int", "desc": "资产负债表"},
        {"api": "cashflow", "fields": "n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,free_cashflow", "desc": "现金流量表"},
        {"api": "stk_factor", "fields": "macd_dif,macd_dea,macd,kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,boll_upper,boll_mid,boll_lower,cci", "desc": "技术因子"},
        {"api": "cyq_perf", "fields": "his_low,his_high,cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,weight_avg,winner_rate", "desc": "筹码分布"},
        {"api": "margin_detail", "fields": "rzye,rzmre,rzche,rqye,rqmcl,rqchl", "desc": "融资融券"},
        {"api": "moneyflow", "fields": "buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,buy_lg_vol,sell_lg_vol,buy_elg_vol,sell_elg_vol,net_mf_vol", "desc": "资金流向"},
    ]

    def __init__(self, cache: Optional[ExpertToolCache] = None):
        self.cache = cache

    def search(self, query: str) -> List[Dict]:
        if self.cache:
            cached = self.cache.get(self.TOOL_NAME, query)
            if cached is not None:
                return cached

        results = self._search_factors(query)

        if self.cache and results:
            self.cache.set(self.TOOL_NAME, query, results)

        return results

    def _search_factors(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []

        for factor_info in self.AVAILABLE_FACTORS:
            fields_lower = factor_info["fields"].lower()
            desc_lower = factor_info["desc"].lower()

            relevance = 0
            # Check if query words appear in fields/desc, or vice versa
            for word in query_lower.split():
                if word in fields_lower or word in desc_lower:
                    relevance += 1
            # Also check if fields/desc keywords appear in query (handles Chinese)
            for keyword in desc_lower.split("、") + desc_lower.split():
                if len(keyword) >= 2 and keyword in query_lower:
                    relevance += 1
            for field in factor_info["fields"].split(","):
                if field.lower() in query_lower:
                    relevance += 1

            if relevance > 0 or query_lower in ("all", "全部", "列表"):
                results.append({
                    "type": "tushare_api",
                    "title": f"{factor_info['api']} - {factor_info['desc']}",
                    "content": f"可用字段: {factor_info['fields']}",
                    "api_name": factor_info["api"],
                    "fields": factor_info["fields"].split(","),
                    "relevance": relevance,
                    "source": "tushare",
                })

        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:10]
