"""
2026年1-4月因子向量化回测
每月按因子排名换仓，等权配置HS300前20%股票
手续费: 0.085% 单边（买卖各收）
交易单位: 100股整数倍（A股整手规则）
"""

import os
import sys
import pickle
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMMISSION_RATE = 0.00085  # 0.085% per side


def build_price_matrix(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """构建价格截面矩阵 (date × stock)"""
    price_dict = {}
    for code, df in stock_data.items():
        if "close" in df.columns and len(df) >= 20:
            price_dict[code] = df["close"]
    return pd.DataFrame(price_dict).sort_index()


def compute_factor_matrix(stock_data: Dict[str, pd.DataFrame], factor_func) -> pd.DataFrame:
    """计算因子截面矩阵 (date × stock)"""
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
    commission: float = COMMISSION_RATE,
):
    """
    向量化因子轮动回测
    """
    price = price[price.index >= "2026-01-01"].copy()
    factor = factor[factor.index >= "2026-01-01"].copy()

    all_dates = sorted(price.index)
    if not all_dates:
        return None

    # 找每月首个交易日
    rebalance_dates = []
    for d in all_dates:
        if not rebalance_dates or d.month != rebalance_dates[-1].month:
            rebalance_dates.append(d)

    print(f"\n回测日期: {all_dates[0].date()} ~ {all_dates[-1].date()}, {len(all_dates)}个交易日")
    print(f"换仓日期: {[str(d)[:10] for d in rebalance_dates]}")
    print(f"股票数量: {len(price.columns)}, 手续费: {commission*100:.3f}%单边")

    # 每月选股 (使用换仓日前一日的因子值)
    monthly_picks = {}
    for rb_date in rebalance_dates:
        factor_dates = factor.index[factor.index <= rb_date]
        if len(factor_dates) == 0:
            continue
        fv = factor.loc[factor_dates[-1]].dropna()
        valid = [c for c in fv.index if c in price.columns]
        fv = fv[valid]
        if len(fv) < 30:
            continue

        sorted_fv = fv.sort_values(ascending=False)
        top_n = max(int(len(sorted_fv) * top_fraction), 10)
        monthly_picks[rb_date] = list(sorted_fv.index[:top_n])

        top5 = ", ".join([f"{c}({v:.3f})" for c, v in sorted_fv.head(5).items()])
        print(f"  {str(rb_date)[:10]}: Top{top_n} | {top5} | "
              f"因子[{sorted_fv.iloc[0]:.4f}, {sorted_fv.iloc[-1]:.4f}]")

    if not monthly_picks:
        logger.error("无有效选股")
        return None

    # ─── 回测主循环 ───
    cash = 1_000_000.0
    holdings = {}       # {code: shares}
    daily_records = []  # [{date, cash, holdings_value, total_value}]
    trades = []

    for i, date in enumerate(all_dates):
        px = price.loc[date]  # 当日收盘价

        # ── 月初换仓 ──
        if date in monthly_picks:
            # 1. 卖出全部持仓（按当日收盘价）
            sell_proceeds = 0.0
            for code, shares in holdings.items():
                if code in px.index and not np.isnan(px[code]) and px[code] > 0:
                    sell_proceeds += shares * px[code]
            sell_fee = sell_proceeds * commission
            cash += sell_proceeds - sell_fee
            holdings = {}

            # 2. 买入新股
            picks = [c for c in monthly_picks[date]
                     if c in px.index and not np.isnan(px[c]) and px[c] > 0]
            if picks:
                total_equity = cash  # 已清仓, cash = 全部净值
                weight = 1.0 / len(picks)
                total_buy = 0.0

                for code in picks:
                    target_value = total_equity * weight
                    price_t = px[code]
                    shares = int(target_value / price_t / 100) * 100
                    if shares >= 100:
                        cost = shares * price_t
                        if cost <= cash:
                            cash -= cost
                            total_buy += cost
                            holdings[code] = shares

                buy_fee = total_buy * commission
                cash -= buy_fee

                trades.append({
                    "date": str(date)[:10], "n_stocks": len(holdings),
                    "sell_fee": round(sell_fee, 2), "buy_fee": round(buy_fee, 2),
                    "total_fee": round(sell_fee + buy_fee, 2),
                    "equity": round(cash + sum(
                        h * px[c] for c, h in holdings.items()
                        if c in px.index and not np.isnan(px[c])
                    ), 2),
                })

        # ── 计算当日总净值 ──
        holdings_value = 0.0
        for code, shares in holdings.items():
            if code in px.index and not np.isnan(px[code]) and px[code] > 0:
                holdings_value += shares * px[code]

        total_value = cash + holdings_value
        daily_records.append({
            "date": date, "cash": cash, "holdings_value": holdings_value,
            "total_value": total_value,
        })

    # ─── 构建结果DataFrame ───
    df = pd.DataFrame(daily_records).set_index("date")
    df["daily_return"] = df["total_value"].pct_change()
    df = df.iloc[1:]  # 去掉第一天 (无前一日收益)

    # ─── 基准: 全市场等权 ───
    ret_matrix = price.pct_change().dropna()
    bench_daily = ret_matrix[ret_matrix.index >= all_dates[0]].mean(axis=1).dropna()

    # ─── 结果计算 ───
    final_value = df["total_value"].iloc[-1]
    total_return = (final_value / 1_000_000 - 1) * 100
    total_fee = sum(t["total_fee"] for t in trades)

    daily_rets = df["daily_return"]
    avg_daily = daily_rets.mean() * 100
    daily_vol = daily_rets.std() * 100

    # 月度
    monthly_ret = daily_rets.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
    bench_monthly = bench_daily.resample("ME").apply(lambda x: np.prod(1 + x) - 1)

    # 净值
    nav = df["total_value"] / 1_000_000
    cum_max = nav.expanding().max()
    drawdown = (nav - cum_max) / cum_max
    max_dd = drawdown.min() * 100
    win_rate = (daily_rets > 0).sum() / len(daily_rets)

    # ─── 打印 ───
    print(f"\n{'='*60}")
    print(f"  回测结果 (2026年1月-4月, 沪深300因子轮动)")
    print(f"{'='*60}")
    print(f"  初始资金:       ¥{1_000_000:,.0f}")
    print(f"  期末资金:       ¥{final_value:,.0f}")
    print(f"  总收益率:       {total_return:+.2f}%")
    print(f"  总手续费:       ¥{total_fee:,.0f} ({total_fee/1_000_000*100:.3f}%)")
    print(f"  日均收益率:     {avg_daily:+.3f}%")
    print(f"  日波动率:       {daily_vol:.3f}%")
    print(f"  年化Sharpe:     {(avg_daily/daily_vol)*np.sqrt(252):.2f}" if daily_vol > 0 else "  年化Sharpe:     N/A")
    print(f"  最大回撤:       {max_dd:.2f}%")
    print(f"  日胜率:         {(daily_rets>0).sum()}/{len(daily_rets)} = {win_rate:.1%}")

    print(f"\n  ═══ 月度收益 ═══")
    print(f"  {'月份':<10} {'因子策略':>10} {'等权基准':>10} {'超额收益':>10}")
    print(f"  {'-'*44}")
    for dt in monthly_ret.index:
        fm = monthly_ret.get(dt, 0)
        bm = bench_monthly.get(dt, 0)
        print(f"  {dt.strftime('%Y-%m'):<10} {fm*100:>+9.2f}% {bm*100:>+9.2f}% {(fm-bm)*100:>+9.2f}%")

    factor_cum = float((1 + monthly_ret).prod() - 1)
    bench_cum = float((1 + bench_monthly).prod() - 1)
    print(f"  {'-'*44}")
    print(f"  {'累计':<10} {factor_cum*100:>+9.2f}% {bench_cum*100:>+9.2f}% {(factor_cum-bench_cum)*100:>+9.2f}%")

    print(f"\n  ═══ 换仓记录 ═══")
    for t in trades:
        print(f"  {t['date']}: {t['n_stocks']}只, 手续费¥{t['total_fee']:,.0f}, 净值¥{t['equity']:,.0f}")

    # 股票持仓分析
    print(f"\n  ═══ 逐日净值(min/mean/max) ═══")
    print(f"  净值: min={nav.min():.4f}, mean={nav.mean():.4f}, max={nav.max():.4f}")

    return {
        "total_return_pct": round(total_return, 2),
        "final_value": round(float(final_value), 2),
        "total_fee": round(float(total_fee), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "daily_win_rate": round(float(win_rate), 4),
        "avg_daily_return_pct": round(float(avg_daily), 4),
        "daily_vol_pct": round(float(daily_vol), 4),
        "monthly_returns": {str(dt)[:7]: round(float(r)*100, 2)
                            for dt, r in monthly_ret.items()},
        "bench_monthly_returns": {str(dt)[:7]: round(float(r)*100, 2)
                                  for dt, r in bench_monthly.items()},
        "factor_cum_return_pct": round(float(factor_cum)*100, 2),
        "bench_cum_return_pct": round(float(bench_cum)*100, 2),
        "trades": trades,
    }


def main():
    import yaml
    import importlib.util

    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # 加载因子
    best_factor = os.path.join(base_dir, "output", "best", "best_program.py")
    spec = importlib.util.spec_from_file_location("best_factor", best_factor)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = mod.get_factor_info()
    print(f"因子: {info['name']} - {info['description']}")

    # 加载2026数据
    cache_2026 = os.path.join(base_dir, "data_cache", "stock_data_2026.pkl")
    with open(cache_2026, "rb") as f:
        stock_data = pickle.load(f)
    print(f"加载 {len(stock_data)} 只股票2026数据")

    # 截面矩阵
    price_matrix = build_price_matrix(stock_data)
    factor_matrix = compute_factor_matrix(stock_data, mod.compute_factor)
    print(f"价格: {price_matrix.shape}, 因子: {factor_matrix.shape}")

    # 回测
    results = vectorized_backtest(price_matrix, factor_matrix)

    # 保存
    if results:
        output_dir = os.path.join(base_dir, "output_test")
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "factor": info,
            "period": "2026-01 ~ 2026-04",
            "commission": "0.085% per side",
            "initial_capital": 1_000_000,
            "strategy": "monthly rebalance, top 20% equal-weight, 100-share lots",
            "results": results,
        }
        path = os.path.join(output_dir, "backtest_2026_results.json")
        with open(path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果保存: {path}")


if __name__ == "__main__":
    main()
