"""
向量化截面因子回测工具

从 backtest_factor.py 中提取的通用回测框架。
适用于多股票截面因子轮动策略的快速回测。

核心思想：
- 截面因子回测天然适合矩阵运算（date × stock）
- 避免事件驱动回测框架（backtrader等）的复杂性
- 每月按因子排名换仓，等权配置前N%股票
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """
    向量化截面因子轮动回测器

    使用方式：
        backtester = VectorizedBacktester(
            price_matrix=price_df,
            factor_matrix=factor_df,
            top_fraction=0.2,
            commission=0.00085,
        )
        results = backtester.run()

    或使用便捷函数：
        results = vectorized_backtest(price_df, factor_df)
    """

    def __init__(
        self,
        price_matrix: pd.DataFrame,
        factor_matrix: pd.DataFrame,
        top_fraction: float = 0.2,
        commission: float = 0.00085,
        min_stocks: int = 10,
        rebalance_freq: str = "monthly",
        lot_size: int = 100,
        initial_capital: float = 1_000_000.0,
    ):
        """
        Args:
            price_matrix: 价格截面矩阵 (date × stock)，index为日期，columns为股票代码
            factor_matrix: 因子截面矩阵 (date × stock)，与price_matrix对齐
            top_fraction: 选股比例（如0.2=选因子排名前20%的股票）
            commission: 单边手续费率（默认0.085%）
            min_stocks: 最少持仓股票数
            rebalance_freq: 换仓频率，"monthly" 或 "weekly"
            lot_size: 交易单位（A股=100股）
            initial_capital: 初始资金
        """
        self.price = price_matrix
        self.factor = factor_matrix
        self.top_fraction = top_fraction
        self.commission = commission
        self.min_stocks = min_stocks
        self.rebalance_freq = rebalance_freq
        self.lot_size = lot_size
        self.initial_capital = initial_capital

    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        执行回测

        Args:
            start_date: 回测开始日期（可选，默认使用全部数据）
            end_date: 回测结束日期（可选）

        Returns:
            Dict with:
                - total_return_pct: 总收益率(%)
                - final_value: 期末资金
                - total_fee: 总手续费
                - max_drawdown_pct: 最大回撤(%)
                - daily_win_rate: 日胜率
                - avg_daily_return_pct: 日均收益率(%)
                - daily_vol_pct: 日波动率(%)
                - annual_sharpe: 年化夏普率
                - monthly_returns: {月份: 收益率}
                - daily_nav: 每日净值Series
                - trades: 换仓记录列表
        """
        # 过滤日期范围
        price = self.price.copy()
        factor = self.factor.copy()

        if start_date:
            price = price[price.index >= start_date]
            factor = factor[factor.index >= start_date]
        if end_date:
            price = price[price.index <= end_date]
            factor = factor[factor.index <= end_date]

        all_dates = sorted(price.index)
        if len(all_dates) < 2:
            logger.error("回测数据不足（至少需要2个交易日）")
            return self._empty_result()

        # 找换仓日期
        rebalance_dates = self._find_rebalance_dates(all_dates)

        logger.info(
            f"回测: {all_dates[0].date()} ~ {all_dates[-1].date()}, "
            f"{len(all_dates)}天, {len(rebalance_dates)}次换仓, "
            f"{len(price.columns)}只股票"
        )

        # 每月选股
        monthly_picks = self._select_stocks(rebalance_dates, factor, price)

        if not monthly_picks:
            logger.error("无有效选股")
            return self._empty_result()

        # 回测主循环
        cash = self.initial_capital
        holdings: Dict[str, int] = {}
        daily_records = []
        trades = []

        for i, date in enumerate(all_dates):
            px = price.loc[date]

            # 换仓
            if date in monthly_picks:
                trade = self._rebalance(date, px, monthly_picks[date], cash, holdings)
                cash = trade["cash"]
                holdings = trade["holdings"]
                trades.append(trade["record"])

            # 计算当日净值
            holdings_value = 0.0
            for code, shares in holdings.items():
                if code in px.index and not np.isnan(px[code]) and px[code] > 0:
                    holdings_value += shares * px[code]

            total_value = cash + holdings_value
            daily_records.append({
                "date": date,
                "cash": cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
            })

        # 构建结果DataFrame
        df = pd.DataFrame(daily_records).set_index("date")
        df["daily_return"] = df["total_value"].pct_change()
        df = df.iloc[1:]  # 去掉第一天

        # 基准: 全市场等权
        ret_matrix = price.pct_change().dropna()
        bench_daily = ret_matrix[ret_matrix.index >= all_dates[0]].mean(axis=1).dropna()

        # 计算指标
        return self._compute_metrics(df, bench_daily, trades, all_dates)

    def _find_rebalance_dates(self, all_dates: List) -> List:
        """找到换仓日期"""
        rebalance_dates = []
        for d in all_dates:
            if not rebalance_dates:
                rebalance_dates.append(d)
            elif self.rebalance_freq == "monthly":
                if d.month != rebalance_dates[-1].month:
                    rebalance_dates.append(d)
            elif self.rebalance_freq == "weekly":
                if d.isocalendar()[1] != rebalance_dates[-1].isocalendar()[1]:
                    rebalance_dates.append(d)
        return rebalance_dates

    def _select_stocks(
        self, rebalance_dates: List, factor: pd.DataFrame, price: pd.DataFrame
    ) -> Dict:
        """每月根据因子排名选股"""
        monthly_picks = {}
        for rb_date in rebalance_dates:
            factor_dates = factor.index[factor.index <= rb_date]
            if len(factor_dates) == 0:
                continue

            fv = factor.loc[factor_dates[-1]].dropna()
            valid = [c for c in fv.index if c in price.columns]
            fv = fv[valid]

            if len(fv) < self.min_stocks:
                continue

            sorted_fv = fv.sort_values(ascending=False)
            top_n = max(int(len(sorted_fv) * self.top_fraction), self.min_stocks)
            monthly_picks[rb_date] = list(sorted_fv.index[:top_n])

        return monthly_picks

    def _rebalance(
        self, date, px: pd.Series, picks: List[str],
        cash: float, holdings: Dict[str, int]
    ) -> Dict:
        """执行换仓操作"""
        # 卖出全部持仓
        sell_proceeds = 0.0
        for code, shares in holdings.items():
            if code in px.index and not np.isnan(px[code]) and px[code] > 0:
                sell_proceeds += shares * px[code]
        sell_fee = sell_proceeds * self.commission
        cash += sell_proceeds - sell_fee
        holdings = {}

        # 买入新股
        picks = [c for c in picks if c in px.index and not np.isnan(px[c]) and px[c] > 0]
        total_buy = 0.0

        if picks:
            total_equity = cash
            weight = 1.0 / len(picks)

            for code in picks:
                target_value = total_equity * weight
                price_t = px[code]
                shares = int(target_value / price_t / self.lot_size) * self.lot_size
                if shares >= self.lot_size:
                    cost = shares * price_t
                    if cost <= cash:
                        cash -= cost
                        total_buy += cost
                        holdings[code] = shares

        buy_fee = total_buy * self.commission
        cash -= buy_fee

        equity = cash + sum(
            h * px[c] for c, h in holdings.items()
            if c in px.index and not np.isnan(px[c])
        )

        return {
            "cash": cash,
            "holdings": holdings,
            "record": {
                "date": str(date)[:10],
                "n_stocks": len(holdings),
                "sell_fee": round(sell_fee, 2),
                "buy_fee": round(buy_fee, 2),
                "total_fee": round(sell_fee + buy_fee, 2),
                "equity": round(equity, 2),
            },
        }

    def _compute_metrics(
        self, df: pd.DataFrame, bench_daily: pd.Series,
        trades: List[Dict], all_dates: List
    ) -> Dict[str, Any]:
        """计算回测绩效指标"""
        final_value = df["total_value"].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        total_fee = sum(t["total_fee"] for t in trades)

        daily_rets = df["daily_return"]
        avg_daily = daily_rets.mean() * 100
        daily_vol = daily_rets.std() * 100

        # 月度
        monthly_ret = daily_rets.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
        bench_monthly = bench_daily.resample("ME").apply(lambda x: np.prod(1 + x) - 1)

        # 净值
        nav = df["total_value"] / self.initial_capital
        cum_max = nav.expanding().max()
        drawdown = (nav - cum_max) / cum_max
        max_dd = drawdown.min() * 100
        win_rate = (daily_rets > 0).sum() / len(daily_rets)

        # 年化指标
        annual_sharpe = (avg_daily / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0

        factor_cum = float((1 + monthly_ret).prod() - 1)
        bench_cum = float((1 + bench_monthly).prod() - 1)

        return {
            "total_return_pct": round(total_return, 2),
            "final_value": round(float(final_value), 2),
            "total_fee": round(float(total_fee), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "daily_win_rate": round(float(win_rate), 4),
            "avg_daily_return_pct": round(float(avg_daily), 4),
            "daily_vol_pct": round(float(daily_vol), 4),
            "annual_sharpe": round(annual_sharpe, 2),
            "monthly_returns": {
                str(dt)[:7]: round(float(r) * 100, 2)
                for dt, r in monthly_ret.items()
            },
            "bench_monthly_returns": {
                str(dt)[:7]: round(float(r) * 100, 2)
                for dt, r in bench_monthly.items()
            },
            "factor_cum_return_pct": round(float(factor_cum) * 100, 2),
            "bench_cum_return_pct": round(float(bench_cum) * 100, 2),
            "daily_nav": nav,
            "trades": trades,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0,
            "final_value": 0.0,
            "total_fee": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_win_rate": 0.0,
            "error": "回测数据不足",
        }

    def print_report(self, results: Dict[str, Any]):
        """打印回测报告"""
        if "error" in results:
            print(f"回测失败: {results['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  向量化因子轮动回测报告")
        print(f"{'='*60}")
        print(f"  初始资金:       ¥{self.initial_capital:,.0f}")
        print(f"  期末资金:       ¥{results['final_value']:,.0f}")
        print(f"  总收益率:       {results['total_return_pct']:+.2f}%")
        print(f"  总手续费:       ¥{results['total_fee']:,.0f}")
        print(f"  日均收益率:     {results['avg_daily_return_pct']:+.3f}%")
        print(f"  日波动率:       {results['daily_vol_pct']:.3f}%")
        print(f"  年化Sharpe:     {results['annual_sharpe']:.2f}")
        print(f"  最大回撤:       {results['max_drawdown_pct']:.2f}%")
        print(f"  日胜率:         {results['daily_win_rate']:.1%}")

        if "monthly_returns" in results:
            print(f"\n  ═══ 月度收益 ═══")
            print(f"  {'月份':<10} {'因子策略':>10} {'等权基准':>10} {'超额收益':>10}")
            print(f"  {'-'*44}")
            for month in results["monthly_returns"]:
                fm = results["monthly_returns"][month]
                bm = results["bench_monthly_returns"].get(month, 0)
                print(f"  {month:<10} {fm:>+9.2f}% {bm:>+9.2f}% {(fm-bm):>+9.2f}%")
            print(f"  {'-'*44}")
            print(f"  {'累计':<10} {results['factor_cum_return_pct']:>+9.2f}% "
                  f"{results['bench_cum_return_pct']:>+9.2f}% "
                  f"{(results['factor_cum_return_pct']-results['bench_cum_return_pct']):>+9.2f}%")


# ── 便捷函数 ──

def build_price_matrix(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从 {code: DataFrame} 构建价格截面矩阵 (date × stock)"""
    price_dict = {}
    for code, df in stock_data.items():
        if "close" in df.columns and len(df) >= 20:
            price_dict[code] = df["close"]
    return pd.DataFrame(price_dict).sort_index()


def build_factor_matrix(
    stock_data: Dict[str, pd.DataFrame], factor_func: Callable
) -> pd.DataFrame:
    """从 {code: DataFrame} 和因子函数构建因子截面矩阵 (date × stock)"""
    factor_dict = {}
    for code, df in stock_data.items():
        try:
            fv = factor_func(df)
            if fv is not None and isinstance(fv, pd.Series):
                factor_dict[code] = fv
        except Exception:
            continue
    return pd.DataFrame(factor_dict).sort_index()


def vectorized_backtest(
    price: pd.DataFrame,
    factor: pd.DataFrame,
    top_fraction: float = 0.2,
    commission: float = 0.00085,
    initial_capital: float = 1_000_000.0,
) -> Optional[Dict[str, Any]]:
    """
    一站式向量化因子轮动回测

    Args:
        price: 价格截面矩阵 (date × stock)
        factor: 因子截面矩阵 (date × stock)
        top_fraction: 选股比例
        commission: 单边手续费率
        initial_capital: 初始资金

    Returns:
        回测结果字典，或 None（数据不足时）
    """
    backtester = VectorizedBacktester(
        price_matrix=price,
        factor_matrix=factor,
        top_fraction=top_fraction,
        commission=commission,
        initial_capital=initial_capital,
    )
    results = backtester.run()
    if "error" in results:
        return None
    return results
