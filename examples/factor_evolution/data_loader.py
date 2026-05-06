"""
A股多股票数据加载器 - 基于 Tushare

支持：
- 沪深300股票池获取
- 日线行情 + 基本面数据批量下载
- 技术指标计算
- 本地 parquet 缓存（避免重复下载）
- 截面数据构建
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

# 默认缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


class TushareDataLoader:
    """Tushare 数据加载器，带本地缓存"""

    def __init__(self, token: str, cache_dir: str = CACHE_DIR):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ── 股票池 ──────────────────────────────────────────────

    def get_hs300_stocks(self, date: str = "") -> List[str]:
        """
        获取沪深300成分股列表

        Args:
            date: 日期 YYYYMMDD，空则取最新

        Returns:
            股票代码列表，如 ['000001.SZ', '000002.SZ', ...]
        """
        cache_file = os.path.join(self.cache_dir, f"hs300_stocks_{date or 'latest'}.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            return df["con_code"].tolist()

        try:
            df = self.pro.index_weight(
                index_code="399300.SZ",
                start_date=date or None,
                end_date=date or None,
            )
            if df is None or df.empty:
                # 备选：直接获取成分
                df = self.pro.index_weight(index_code="399300.SZ")
            time.sleep(0.3)

            if df is not None and not df.empty:
                # 取最新一期
                latest_date = df["trade_date"].max()
                df = df[df["trade_date"] == latest_date]
                df.to_parquet(cache_file, index=False)
                codes = df["con_code"].tolist()
                logger.info(f"获取沪深300成分股 {len(codes)} 只 (日期: {latest_date})")
                return codes
        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")

        return []

    def get_stock_pool(self, pool_type: str = "hs300", date: str = "") -> List[str]:
        """获取股票池"""
        if pool_type == "hs300":
            return self.get_hs300_stocks(date)
        raise ValueError(f"不支持的股票池类型: {pool_type}")

    # ── 单股票数据下载 ────────────────────────────────────────

    def _download_daily(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载日线行情"""
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            time.sleep(0.15)  # 限频
            return df
        except Exception as e:
            logger.warning(f"下载 {ts_code} 日线失败: {e}")
            return None

    def _download_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载日线基本面指标"""
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,turnover_rate,turnover_rate_f,"
                       "volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
                       "dv_ratio,dv_ttm,total_share,float_share,"
                       "total_mv,circ_mv",
            )
            time.sleep(0.15)
            return df
        except Exception as e:
            logger.warning(f"下载 {ts_code} 基本面失败: {e}")
            return None

    def _download_adj_factor(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """下载复权因子"""
        try:
            df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            time.sleep(0.15)
            return df
        except Exception as e:
            logger.warning(f"下载 {ts_code} 复权因子失败: {e}")
            return None

    # ── 数据加工 ──────────────────────────────────────────────

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """为单只股票数据添加技术指标"""
        d = df.copy()

        close = d["close"]
        high = d["high"]
        low = d["low"]
        volume = d["vol"]

        # 移动平均
        for w in [5, 10, 20, 60]:
            d[f"ma{w}"] = close.rolling(w).mean()

        # EMA
        d["ema12"] = close.ewm(span=12).mean()
        d["ema26"] = close.ewm(span=26).mean()

        # MACD
        d["macd"] = d["ema12"] - d["ema26"]
        d["macd_signal"] = d["macd"].ewm(span=9).mean()
        d["macd_histogram"] = d["macd"] - d["macd_signal"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi"] = 100 - (100 / (1 + rs))

        # KDJ
        lowest = low.rolling(9).min()
        highest = high.rolling(9).max()
        rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
        d["kdj_k"] = rsv.ewm(alpha=1 / 3).mean()
        d["kdj_d"] = d["kdj_k"].ewm(alpha=1 / 3).mean()
        d["kdj_j"] = 3 * d["kdj_k"] - 2 * d["kdj_d"]

        # 布林带
        d["bb_middle"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        d["bb_upper"] = d["bb_middle"] + 2 * bb_std
        d["bb_lower"] = d["bb_middle"] - 2 * bb_std

        # MA 斜率
        d["ma5_slope"] = d["ma5"].diff(5) / 5
        d["ma20_slope"] = d["ma20"].diff(20) / 20

        # 额外常用因子原料
        d["returns_1d"] = close.pct_change(1)
        d["returns_5d"] = close.pct_change(5)
        d["returns_20d"] = close.pct_change(20)
        d["vol_ma5"] = volume.rolling(5).mean()
        d["vol_ma20"] = volume.rolling(20).mean()
        d["vol_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
        d["amplitude"] = (high - low) / close.shift(1).replace(0, np.nan)
        d["upper_shadow"] = (high - np.maximum(close, d["open"])) / close.replace(0, np.nan)
        d["lower_shadow"] = (np.minimum(close, d["open"]) - low) / close.replace(0, np.nan)

        return d

    # ── 核心加载接口 ──────────────────────────────────────────

    def load_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        add_indicators: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        加载单只股票全部数据（带缓存）

        Returns:
            DataFrame with index=trade_date (datetime), sorted ascending
        """
        cache_key = hashlib.md5(f"{ts_code}_{start_date}_{end_date}".encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"stock_{ts_code}_{cache_key}.parquet")

        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if add_indicators and "ma5" not in df.columns:
                df = self.add_technical_indicators(df)
            return df

        # 下载日线行情
        daily = self._download_daily(ts_code, start_date, end_date)
        if daily is None or daily.empty:
            return None

        # 下载基本面
        basic = self._download_daily_basic(ts_code, start_date, end_date)

        # 下载复权因子
        adj = self._download_adj_factor(ts_code, start_date, end_date)

        # 合并
        df = daily.copy()
        if basic is not None and not basic.empty:
            df = df.merge(basic, on=["ts_code", "trade_date"], how="left", suffixes=("", "_basic"))
        if adj is not None and not adj.empty:
            df = df.merge(adj, on=["ts_code", "trade_date"], how="left")

        # 格式化
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df.set_index("trade_date")

        # 后复权价格
        if "adj_factor" in df.columns:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[f"{col}_adj"] = df[col] * df["adj_factor"]

        # 清洗
        df = df[df["close"] > 0].copy()

        # 添加技术指标
        if add_indicators:
            df = self.add_technical_indicators(df)

        # 缓存
        df.to_parquet(cache_file)
        logger.info(f"已缓存 {ts_code} 数据 ({len(df)} 行)")

        return df

    def load_stock_pool_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        add_indicators: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载股票池数据

        Returns:
            {ts_code: DataFrame} 字典
        """
        result = {}
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"加载进度: {i + 1}/{total}")
                print(f"  加载进度: {i + 1}/{total} ...")

            df = self.load_stock_data(code, start_date, end_date, add_indicators)
            if df is not None and len(df) > 60:  # 至少60个交易日
                result[code] = df

        logger.info(f"成功加载 {len(result)}/{total} 只股票数据")
        print(f"成功加载 {len(result)}/{total} 只股票数据")
        return result

    # ── 截面数据构建 ──────────────────────────────────────────

    @staticmethod
    def build_cross_section(
        stock_data: Dict[str, pd.DataFrame],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        构建截面矩阵 (日期 x 股票)

        Args:
            stock_data: {ts_code: DataFrame} 字典
            column: 要提取的列名

        Returns:
            DataFrame: index=trade_date, columns=ts_code
        """
        series_dict = {}
        for code, df in stock_data.items():
            if column in df.columns:
                series_dict[code] = df[column]

        if not series_dict:
            return pd.DataFrame()

        panel = pd.DataFrame(series_dict)
        panel = panel.sort_index()
        return panel

    @staticmethod
    def build_forward_returns(
        stock_data: Dict[str, pd.DataFrame],
        periods: int = 5,
    ) -> pd.DataFrame:
        """
        构建未来N日收益率截面矩阵

        Args:
            stock_data: {ts_code: DataFrame} 字典
            periods: 未来收益计算天数

        Returns:
            DataFrame: index=trade_date, columns=ts_code
        """
        series_dict = {}
        for code, df in stock_data.items():
            if "close" in df.columns:
                fwd_ret = df["close"].pct_change(periods).shift(-periods)
                series_dict[code] = fwd_ret

        if not series_dict:
            return pd.DataFrame()

        panel = pd.DataFrame(series_dict)
        panel = panel.sort_index()
        return panel


# ── 便捷函数 ────────────────────────────────────────────────

def create_loader(token: str, cache_dir: str = CACHE_DIR) -> TushareDataLoader:
    """创建数据加载器"""
    return TushareDataLoader(token=token, cache_dir=cache_dir)


def download_all_data(
    token: str,
    start_date: str = "20210101",
    end_date: str = "20250101",
    max_stocks: int = 100,
    cache_dir: str = CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    一键下载全部数据到本地缓存

    Args:
        token: Tushare token
        start_date: 起始日期
        end_date: 截止日期
        max_stocks: 最多下载股票数量
        cache_dir: 缓存目录

    Returns:
        {ts_code: DataFrame} 字典
    """
    loader = TushareDataLoader(token=token, cache_dir=cache_dir)

    print("=" * 50)
    print("A股数据下载 (Tushare)")
    print("=" * 50)

    # 获取股票池
    print("正在获取沪深300成分股...")
    codes = loader.get_stock_pool("hs300")
    if not codes:
        print("获取股票池失败，使用默认股票列表")
        codes = [
            "600519.SH", "000858.SZ", "601318.SH", "600036.SH",
            "000333.SZ", "600276.SH", "601166.SH", "000651.SZ",
            "600030.SH", "601398.SH", "600900.SH", "000568.SZ",
            "002714.SZ", "601888.SH", "600809.SH", "000001.SZ",
            "002594.SZ", "300750.SZ", "601012.SH", "600585.SH",
        ]

    # 限制数量
    codes = codes[:max_stocks]
    print(f"将下载 {len(codes)} 只股票的数据 ({start_date} ~ {end_date})")

    # 批量下载
    print("开始下载数据（首次下载较慢，后续从缓存读取）...")
    data = loader.load_stock_pool_data(
        codes, start_date, end_date,
        add_indicators=True,
        show_progress=True,
    )

    print(f"\n下载完成！有效股票: {len(data)}")
    print(f"缓存目录: {cache_dir}")
    print("=" * 50)

    return data


if __name__ == "__main__":
    # 测试数据加载
    TOKEN = os.environ.get("TUSHARE_TOKEN", "")
    if not TOKEN:
        print("请设置环境变量 TUSHARE_TOKEN")
        sys.exit(1)
    data = download_all_data(token=TOKEN, max_stocks=5, start_date="20220101", end_date="20240601")

    if data:
        code = list(data.keys())[0]
        df = data[code]
        print(f"\n示例数据 ({code}):")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {df.columns.tolist()[:20]}...")
        print(f"  日期范围: {df.index.min()} ~ {df.index.max()}")
