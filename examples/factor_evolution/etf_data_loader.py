"""
A股ETF数据加载器 - 基于 Tushare

支持：
- 全市场ETF列表获取与分类
- ETF日线行情 (fund_daily)
- ETF份额变动 (fund_share) — 资金流入/流出代理信号
- 复权因子 (fund_adj)
- 基金净值 (fund_nav) — 折溢价计算
- 北向资金 (moneyflow_hsgt)
- 指数日线 (index_daily)
- 本地 parquet 缓存
"""

import os
import time
import logging
import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import tushare as ts
except ImportError:
    raise ImportError("请安装 tushare: pip install tushare")

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# ETF 分类关键词映射
ETF_CATEGORIES = {
    "宽基": {
        "keywords": ["沪深300", "中证500", "中证1000", "上证50", "科创50",
                     "创业板", "中证A500", "上证180", "深证100", "MSCI"],
        "exclude": ["增强", "备兑"],
    },
    "行业": {
        "keywords": ["银行", "证券", "保险", "金融", "医药", "医疗", "生物",
                     "半导体", "芯片", "新能源", "光伏", "电力", "煤炭",
                     "有色", "钢铁", "化工", "军工", "国防", "消费",
                     "食品", "白酒", "家电", "汽车", "地产", "房地产",
                     "建材", "农业", "传媒", "通信", "计算机", "电子",
                     "机械", "交通", "基建", "环保", "旅游"],
        "exclude": [],
    },
    "红利": {
        "keywords": ["红利", "高股息", "股息", "dividend"],
        "exclude": [],
    },
    "主题": {
        "keywords": ["AI", "人工智能", "机器人", "碳中和", "一带一路",
                     "信创", "数字经济", "元宇宙", "5G", "物联网",
                     "云计算", "大数据", "区块链", "网络安全", "新基建",
                     "养老", "ESG", "港股通"],
        "exclude": [],
    },
}


class ETFDataLoader:
    """A股ETF数据加载器，带本地parquet缓存"""

    def __init__(self, token: str, cache_dir: str = CACHE_DIR):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _api_call(self, func, sleep_time=0.2, **kwargs):
        """带限频的API调用"""
        try:
            result = func(**kwargs)
            time.sleep(sleep_time)
            return result
        except Exception as e:
            logger.warning(f"API调用失败: {e}, 参数: {kwargs}")
            time.sleep(1)
            return None

    # ── ETF 列表 ──────────────────────────────────────────────

    def get_etf_list(self, status: str = "L") -> pd.DataFrame:
        """
        获取全部ETF基金列表

        Args:
            status: 'L'=上市中, 'D'=已退市, 'P'=暂停上市

        Returns:
            DataFrame: ts_code, name, fund_type, management, list_date, benchmark, ...
        """
        cache_file = os.path.join(self.cache_dir, f"etf_list_{status}.parquet")
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        logger.info("获取ETF基金列表...")
        df = self._api_call(self.pro.fund_basic, market="E", status=status)
        if df is None or df.empty:
            logger.error("获取ETF列表失败")
            return pd.DataFrame()

        df.to_parquet(cache_file, index=False)
        logger.info(f"获取到 {len(df)} 只ETF")
        return df

    def get_tradeable_etfs(self, min_list_days: int = 252) -> pd.DataFrame:
        """
        获取可交易ETF列表（过滤条件：上市中 + 上市满1年）

        Args:
            min_list_days: 最少上市天数

        Returns:
            筛选后的ETF列表
        """
        df = self.get_etf_list(status="L")
        if df.empty:
            return df

        df["list_date"] = pd.to_datetime(df["list_date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=min_list_days)
        df = df[df["list_date"] <= cutoff].copy()

        logger.info(f"上市满{min_list_days}天的ETF: {len(df)} 只")
        return df

    def classify_etfs(self, etf_list: pd.DataFrame) -> Dict[str, List[str]]:
        """
        按类别对ETF进行分类

        Returns:
            {类别名: [ts_code列表]}
        """
        result = {cat: [] for cat in ETF_CATEGORIES}
        result["其他"] = []
        classified = set()

        for _, row in etf_list.iterrows():
            code = row["ts_code"]
            name = row.get("name", "")
            matched = False

            for cat, config in ETF_CATEGORIES.items():
                if any(kw in name for kw in config.get("exclude", [])):
                    continue
                if any(kw.lower() in name.lower() for kw in config["keywords"]):
                    result[cat].append(code)
                    classified.add(code)
                    matched = True
                    break

            if not matched:
                result["其他"].append(code)

        for cat, codes in result.items():
            if codes:
                logger.info(f"  {cat}: {len(codes)} 只")

        return result

    # ── ETF 行情数据 ──────────────────────────────────────────

    def _download_fund_daily(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载ETF日线行情"""
        df = self._api_call(self.pro.fund_daily, ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def _download_fund_share(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载ETF份额数据"""
        df = self._api_call(self.pro.fund_share, ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def _download_fund_adj(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载ETF复权因子"""
        df = self._api_call(self.pro.fund_adj, ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df

    def _download_fund_nav(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载基金净值"""
        df = self._api_call(
            self.pro.fund_nav,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,ann_date,nav_date,unit_nav,accum_nav",
        )
        return df

    def load_etf_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        include_share: bool = True,
        include_nav: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        加载单只ETF全部数据（带缓存）

        Returns:
            DataFrame with index=trade_date (datetime), sorted ascending
            列包含: open, high, low, close, vol, amount, pct_chg, fd_share, ...
        """
        cache_key = hashlib.md5(f"etf_{ts_code}_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"etf_{ts_code}_{cache_key}.parquet")

        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # 下载日线行情
        daily = self._download_fund_daily(ts_code, start_date, end_date)
        if daily is None or daily.empty:
            return None

        df = daily.copy()

        # 下载份额数据
        if include_share:
            share = self._download_fund_share(ts_code, start_date, end_date)
            if share is not None and not share.empty:
                share = share.rename(columns={"trade_date": "trade_date"})
                df = df.merge(share[["ts_code", "trade_date", "fd_share"]],
                              on=["ts_code", "trade_date"], how="left")

        # 下载复权因子
        adj = self._download_fund_adj(ts_code, start_date, end_date)
        if adj is not None and not adj.empty:
            df = df.merge(adj[["ts_code", "trade_date", "adj_factor"]],
                          on=["ts_code", "trade_date"], how="left")

        # 下载净值（可选）
        if include_nav:
            nav = self._download_fund_nav(ts_code, start_date, end_date)
            if nav is not None and not nav.empty:
                nav = nav.rename(columns={"nav_date": "trade_date"})
                nav = nav[["ts_code", "trade_date", "unit_nav", "accum_nav"]].drop_duplicates(
                    subset=["ts_code", "trade_date"], keep="first"
                )
                df = df.merge(nav, on=["ts_code", "trade_date"], how="left")

        # 格式化
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df.set_index("trade_date")

        # 后复权价格
        if "adj_factor" in df.columns:
            df["adj_factor"] = df["adj_factor"].ffill()
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[f"{col}_adj"] = df[col] * df["adj_factor"]

        # 计算份额变动
        if "fd_share" in df.columns:
            df["fd_share"] = df["fd_share"].ffill()
            df["share_chg"] = df["fd_share"].diff()
            df["share_chg_pct"] = df["share_chg"] / df["fd_share"].shift(1)

        # 计算折溢价（如果有净值）
        if "unit_nav" in df.columns and "close" in df.columns:
            df["unit_nav"] = df["unit_nav"].ffill()
            df["nav_premium"] = df["close"] / df["unit_nav"] - 1

        # 添加衍生指标
        df = self._add_derived_features(df)

        # 清洗
        df = df[df["close"] > 0].copy()

        # 缓存
        df.to_parquet(cache_file)
        logger.info(f"已缓存 {ts_code} ETF数据 ({len(df)} 行)")

        return df

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生特征列"""
        close = df["close"]
        vol = df.get("vol")
        high = df.get("high")
        low = df.get("low")

        # 收益率
        for n in [1, 5, 10, 20, 60]:
            df[f"returns_{n}d"] = close.pct_change(n)

        # 波动率
        df["volatility_20d"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)
        df["volatility_60d"] = df["returns_1d"].rolling(60).std() * np.sqrt(252)

        # 动量
        df["momentum_5d"] = close / close.shift(5) - 1
        df["momentum_20d"] = close / close.shift(20) - 1
        df["momentum_60d"] = close / close.shift(60) - 1

        # 均线
        for w in [5, 10, 20, 60]:
            df[f"ma{w}"] = close.rolling(w).mean()
            df[f"ma{w}_bias"] = close / df[f"ma{w}"] - 1

        # 量能
        if vol is not None:
            df["vol_ma5"] = vol.rolling(5).mean()
            df["vol_ma20"] = vol.rolling(20).mean()
            df["vol_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)

        # 振幅
        if high is not None and low is not None:
            pre_close = close.shift(1)
            df["amplitude"] = (high - low) / pre_close.replace(0, np.nan)
            df["amplitude_ma5"] = df["amplitude"].rolling(5).mean()

        return df

    def load_etf_pool_data(
        self,
        etf_codes: List[str],
        start_date: str,
        end_date: str,
        include_share: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载ETF数据

        Returns:
            {ts_code: DataFrame} 字典
        """
        result = {}
        total = len(etf_codes)
        failed = []

        for i, code in enumerate(etf_codes):
            if show_progress and (i + 1) % 20 == 0:
                logger.info(f"ETF加载进度: {i + 1}/{total}")
                print(f"  ETF加载进度: {i + 1}/{total} ...")

            df = self.load_etf_data(code, start_date, end_date, include_share=include_share)
            if df is not None and len(df) > 60:
                result[code] = df
            else:
                failed.append(code)

        logger.info(f"成功加载 {len(result)}/{total} 只ETF数据")
        if failed:
            logger.info(f"加载失败: {len(failed)} 只")
        return result

    # ── 北向资金 ──────────────────────────────────────────────

    def load_northbound_flow(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        下载北向资金数据（沪股通+深股通）

        Returns:
            DataFrame with index=trade_date
            列: north_money(北向净买入,亿), south_money(南向净买入,亿)
        """
        cache_key = hashlib.md5(f"northbound_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"northbound_{cache_key}.parquet")

        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        logger.info("下载北向资金数据...")
        df = self._api_call(
            self.pro.moneyflow_hsgt,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            return None

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").set_index("trade_date")

        # 北向 = 沪股通 + 深股通
        if "north_money" not in df.columns:
            hgt = df.get("hgt", 0)  # 沪股通
            sgt = df.get("sgt", 0)  # 深股通
            df["north_money"] = hgt + sgt

        if "south_money" not in df.columns:
            ggt_ss = df.get("ggt_ss", 0)  # 港股通(沪)
            ggt_sz = df.get("ggt_sz", 0)  # 港股通(深)
            df["south_money"] = ggt_ss + ggt_sz

        df.to_parquet(cache_file)
        logger.info(f"北向资金数据: {len(df)} 天")
        return df

    # ── 指数数据 ──────────────────────────────────────────────

    def load_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        下载指数日线数据

        常用指数:
            000300.SH: 沪深300
            000905.SH: 中证500
            000852.SH: 中证1000
            399006.SZ: 创业板指
            000688.SH: 科创50

        Returns:
            DataFrame with index=trade_date
        """
        cache_key = hashlib.md5(f"index_{index_code}_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"index_{index_code}_{cache_key}.parquet")

        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        df = self._api_call(
            self.pro.index_daily,
            ts_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            return None

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").set_index("trade_date")

        df.to_parquet(cache_file)
        logger.info(f"指数 {index_code} 数据: {len(df)} 天")
        return df

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def build_panel(etf_data: Dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
        """构建截面面板 (日期 x ETF)"""
        series_dict = {}
        for code, df in etf_data.items():
            if column in df.columns:
                series_dict[code] = df[column]
        if not series_dict:
            return pd.DataFrame()
        return pd.DataFrame(series_dict).sort_index()

    @staticmethod
    def filter_by_liquidity(
        etf_data: Dict[str, pd.DataFrame],
        min_avg_amount: float = 10000,
        min_days: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """按流动性筛选ETF（min_avg_amount单位：千元，即1000万=10000千元）"""
        filtered = {}
        for code, df in etf_data.items():
            if len(df) < min_days:
                continue
            avg_amount = df["amount"].mean() if "amount" in df.columns else 0
            if avg_amount >= min_avg_amount:
                filtered[code] = df
        logger.info(f"流动性筛选: {len(etf_data)} → {len(filtered)} 只ETF (日均成交额>{min_avg_amount/1000:.0f}万)")
        return filtered
