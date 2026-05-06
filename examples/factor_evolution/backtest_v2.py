#!/usr/bin/env python3
"""
v2 最优因子 backtrader 回测
- 股票池: 沪深300
- 回测期: 2026-01-01 ~ 2026-04-28
- 手续费: 0.0085% (双边)
- 策略: 月度调仓, 等权持有因子排名前20%的股票
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import backtrader as bt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import TushareDataLoader
from best_program import compute_factor

logging.basicConfig(level=logging.WARNING)

# ── 配置 ──────────────────────────────────────────────
TOKEN = os.environ.get("TUSHARE_TOKEN", "")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
BACKTEST_START = "20260101"
BACKTEST_END = "20260428"
LOOKBACK_DAYS = 150
TOP_PCT = 0.20
COMMISSION = 0.000085  # 0.0085%
INITIAL_CASH = 10_000_000

os.makedirs(CACHE_DIR, exist_ok=True)


def load_stock_data_from_cache():
    """从本地缓存加载股票数据，合并多个缓存文件"""
    loader = TushareDataLoader(token=TOKEN, cache_dir=CACHE_DIR)
    stock_codes = loader.get_stock_pool("hs300")
    print(f"沪深300成分股: {len(stock_codes)} 只")

    stock_data = {}
    for i, code in enumerate(stock_codes):
        dfs = []
        for fname in sorted(os.listdir(CACHE_DIR)):
            if not (fname.startswith(f"stock_{code}_") and fname.endswith(".parquet")):
                continue
            try:
                df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            continue

        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=LOOKBACK_DAYS)
        combined = combined[combined.index >= cutoff]

        if len(combined) < 80:
            continue

        if "ma5" not in combined.columns:
            combined = loader.add_technical_indicators(combined)

        stock_data[code] = combined

        if (i + 1) % 50 == 0:
            print(f"  加载进度: {i+1}/{len(stock_codes)}")

    print(f"  有效股票: {len(stock_data)}")
    return stock_data


def get_trading_dates(stock_data):
    """获取回测期的交易日历"""
    all_dates = set()
    for df in stock_data.values():
        dates = df.index[
            (df.index >= pd.Timestamp(BACKTEST_START))
            & (df.index <= pd.Timestamp(BACKTEST_END))
        ]
        all_dates.update(dates)
    return sorted(all_dates)


def compute_monthly_portfolio(stock_data, trading_dates):
    """
    月度因子选股 + 组合收益计算
    返回: (组合日收益率Series, 月度调仓详情DataFrame)
    """
    start_date = pd.Timestamp(BACKTEST_START)
    end_date = pd.Timestamp(BACKTEST_END)

    portfolio_returns = []
    rebalance_details = []

    # 找到每月第一个交易日
    month_starts = {}
    for dt in trading_dates:
        month_key = (dt.year, dt.month)
        if month_key not in month_starts:
            month_starts[month_key] = dt

    # 按时间顺序处理每个交易日
    current_stocks = []
    for dt in trading_dates:
        month_key = (dt.year, dt.month)

        # 在每月第一个交易日调仓
        if dt == month_starts.get(month_key):
            # 计算所有股票的因子值（只用到 dt 之前的数据）
            scores = {}
            for code, df in stock_data.items():
                hist = df[df.index <= dt]
                if len(hist) < 80:
                    continue
                try:
                    factor = compute_factor(hist)
                    val = factor.iloc[-1]
                    if not np.isnan(val):
                        scores[code] = val
                except Exception:
                    continue

            if len(scores) >= 10:
                n_select = max(5, int(len(scores) * TOP_PCT))
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                current_stocks = [code for code, _ in sorted_scores[:n_select]]

                rebalance_details.append({
                    'date': dt,
                    'n_stocks': len(scores),
                    'n_select': n_select,
                    'top5': current_stocks[:5],
                    'top_scores': [f"{s:.4f}" for _, s in sorted_scores[:3]],
                })

        # 计算当天组合收益率（等权）
        day_ret = 0
        valid_count = 0
        for code in current_stocks:
            df = stock_data.get(code)
            if df is None or dt not in df.index:
                continue
            idx = df.index.get_loc(dt)
            if idx > 0:
                r = df['close'].iloc[idx] / df['close'].iloc[idx - 1] - 1
                day_ret += r
                valid_count += 1

        if valid_count > 0:
            daily_r = day_ret / valid_count
            # 扣除调仓日的手续费
            if dt in month_starts.values() and len(current_stocks) > 0:
                # 卖出旧持仓 + 买入新持仓 = 双边手续费
                # 假设全部换手，手续费 = 2 * COMMISSION
                daily_r -= 2 * COMMISSION / len(current_stocks)
            portfolio_returns.append({'date': dt, 'return': daily_r})
        else:
            portfolio_returns.append({'date': dt, 'return': 0.0})

    return pd.DataFrame(portfolio_returns).set_index('date'), pd.DataFrame(rebalance_details)


def run_backtest():
    print("=" * 60)
    print("v2 最优因子 backtrader 回测")
    print(f"回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"手续费: {COMMISSION*100:.4f}% (双边)")
    print(f"选股: 因子排名前 {int(TOP_PCT*100)}%, 等权配置")
    print("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────
    print("\n[1/4] 加载数据...")
    stock_data = load_stock_data_from_cache()

    # ── 2. 计算因子和组合 ─────────────────────────────────
    print("\n[2/4] 计算因子截面与组合收益...")
    trading_dates = get_trading_dates(stock_data)
    print(f"  回测期交易日: {len(trading_dates)} 天")

    port_df, rebal_df = compute_monthly_portfolio(stock_data, trading_dates)

    print("\n  月度调仓明细:")
    for _, row in rebal_df.iterrows():
        print(f"  {row['date'].date()} | 候选{row['n_stocks']}只 → 选{row['n_select']}只"
              f" | Top: {row['top5'][:3]}")

    # ── 3. backtrader 分析 ───────────────────────────────
    print("\n[3/4] backtrader 指标计算...")
    port_df['nav'] = (1 + port_df['return']).cumprod()
    port_df['value'] = port_df['nav'] * INITIAL_CASH

    # 创建 backtrader 的 portfolio 数据源
    bt_df = pd.DataFrame({
        'open': port_df['value'],
        'high': port_df['value'],
        'low': port_df['value'],
        'close': port_df['value'],
        'volume': 1.0,
    }, index=pd.to_datetime(port_df.index))

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    data_feed = bt.feeds.PandasData(
        dataname=bt_df,
        datetime=None,
        open='open', high='high', low='low', close='close', volume='volume',
    )
    cerebro.adddata(data_feed)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        riskfreerate=0.02, timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn',
                        timeframe=bt.TimeFrame.Months)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annreturn')

    # 简单的 buy-and-hold 策略（直接持有组合）
    class HoldStrategy(bt.Strategy):
        def __init__(self):
            pass
        def next(self):
            if not self.position:
                self.buy(size=1)

    cerebro.addstrategy(HoldStrategy)
    results = cerebro.run()

    # ── 4. 计算基准(沪深300等权) ──────────────────────────
    print("[4/4] 计算基准与汇总指标...")
    bench_returns = []
    for dt in trading_dates:
        day_ret = 0
        count = 0
        for code, df in stock_data.items():
            if dt in df.index:
                idx = df.index.get_loc(dt)
                if idx > 0:
                    day_ret += df['close'].iloc[idx] / df['close'].iloc[idx - 1] - 1
                    count += 1
        bench_returns.append(day_ret / count if count > 0 else 0)

    benchmark_nav = (1 + pd.Series(bench_returns, index=trading_dates)).cumprod()

    # ── 分析器结果 ───────────────────────────────────────
    sharpe = results[0].analyzers.sharpe.get_analysis()
    dd = results[0].analyzers.drawdown.get_analysis()
    returns_analysis = results[0].analyzers.returns.get_analysis()
    timereturn = results[0].analyzers.timereturn.get_analysis()

    # 月度收益
    monthly_ret = {}
    for dt, ret in timereturn.items():
        month_key = f"{dt.year}-{dt.month:02d}"
        monthly_ret[month_key] = monthly_ret.get(month_key, 0) + ret

    final_nav = port_df['nav'].iloc[-1]
    total_return = (final_nav - 1) * 100
    n_trading_days = len(trading_dates)
    annual_return = ((final_nav) ** (252 / n_trading_days) - 1) * 100

    # 手动计算日收益率统计
    daily_ret = port_df['return'].dropna()
    ann_vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe_ratio = (annual_return - 2.0) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    nav_series = port_df['nav']
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = drawdown.min() * 100
    max_dd_date = drawdown.idxmin()

    # 基准指标
    bench_final = benchmark_nav.iloc[-1]
    bench_total = (bench_final - 1) * 100
    bench_annual = ((bench_final) ** (252 / n_trading_days) - 1) * 100
    bench_dd = (benchmark_nav - benchmark_nav.cummax()) / benchmark_nav.cummax()

    # ── 月度基准收益 ────────────────────────────────────
    bench_monthly = {}
    bench_ret_series = pd.Series(bench_returns, index=trading_dates)
    for month_key in sorted(monthly_ret.keys()):
        y, m = month_key.split('-')
        m_start = pd.Timestamp(f"{y}-{m}-01")
        m_end = m_start + pd.offsets.MonthEnd(1)
        m_data = bench_ret_series[(bench_ret_series.index >= m_start) & (bench_ret_series.index <= m_end)]
        bench_monthly[month_key] = (1 + m_data).prod() - 1 if len(m_data) > 0 else 0

    # ── 输出 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 回测结果汇总")
    print("=" * 60)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │ {'指标':<16} {'策略':>10} {'基准(等权)':>12} │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │ {'总收益率':<16} {total_return:>+9.2f}% {bench_total:>+11.2f}% │")
    print(f"  │ {'年化收益率':<16} {annual_return:>+9.2f}% {bench_annual:>+11.2f}% │")
    print(f"  │ {'年化波动率':<16} {ann_vol:>9.2f}% {'':>12} │")
    print(f"  │ {'夏普比率':<16} {sharpe_ratio:>9.3f} {'':>12} │")
    print(f"  │ {'最大回撤':<16} {max_dd:>9.2f}% {bench_dd.min()*100:>+11.2f}% │")
    print(f"  │ {'最终净值':<16} {final_nav:>9.4f} {bench_final:>11.4f} │")
    print(f"  └─────────────────────────────────────────┘")

    # 月度收益对比
    print(f"\n  📅 月度收益率对比")
    print(f"  {'月份':<10} {'策略':>10} {'基准':>10} {'超额':>10}")
    print(f"  {'-'*42}")
    total_excess = 0
    for month in sorted(monthly_ret.keys()):
        s_r = monthly_ret[month] * 100
        b_r = bench_monthly.get(month, 0) * 100
        excess = s_r - b_r
        total_excess += excess
        print(f"  {month:<10} {s_r:>+9.2f}% {b_r:>+9.2f}% {excess:>+9.2f}%")
    print(f"  {'-'*42}")
    print(f"  {'累计超额':<10} {'':>10} {'':>10} {total_excess:>+9.2f}%")

    # 回撤详情
    print(f"\n  📉 最大回撤详情")
    print(f"  最大回撤: {max_dd:.2f}% (发生于 {max_dd_date.date()})")

    # 胜率
    win_days = (daily_ret > 0).sum()
    total_days = len(daily_ret)
    win_rate = win_days / total_days * 100 if total_days > 0 else 0
    print(f"\n  📈 交易统计")
    print(f"  交易日数: {total_days}")
    print(f"  日胜率:   {win_rate:.1f}% ({win_days}/{total_days})")
    print(f"  日均收益: {daily_ret.mean()*100:+.4f}%")
    print(f"  调仓次数: {len(rebal_df)}")

    # 资金曲线对比数据
    print(f"\n  📊 净值对比 (策略 vs 基准)")
    for dt in [pd.Timestamp('2026-01-28'), pd.Timestamp('2026-02-28'),
               pd.Timestamp('2026-03-31'), pd.Timestamp('2026-04-28')]:
        closest = min(port_df.index, key=lambda x: abs(x - dt))
        s_nav = port_df.loc[closest, 'nav']
        b_idx = min(range(len(trading_dates)), key=lambda i: abs(trading_dates[i] - dt))
        b_nav = benchmark_nav.iloc[b_idx]
        print(f"  {closest.date()} | 策略: {s_nav:.4f} | 基准: {b_nav:.4f}")

    print("\n" + "=" * 60)
    print("  回测完成")
    print("=" * 60)

    return port_df, benchmark_nav, rebal_df


if __name__ == "__main__":
    ret = run_backtest()
