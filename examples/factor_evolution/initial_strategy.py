"""
ETF轮动策略 — 初始种子（LLM演化起点）

策略框架：
- Regime判断：用波动率和均线判断市场状态
- 红利底仓 + 动量轮动卫星仓
- 滑动止损 + 波动率自适应仓位
- 熊市减仓保护
"""

import pandas as pd
import numpy as np
from typing import Dict


# EVOLVE-BLOCK-START
def compute_strategy(
    etf_data: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    current_holdings: Dict[str, float],
    entry_prices: Dict[str, float],
) -> Dict[str, float]:
    """
    ETF轮动策略

    Args:
        etf_data: {ETF代码: DataFrame} 截止current_date的历史数据
        current_date: 当前决策日期
        current_holdings: 当前持仓 {代码: 权重}
        entry_prices: 各持仓入场价格 {代码: 价格}

    Returns:
        目标权重向量 {代码: 权重}，和≤1.0，最多10个持仓
    """
    # ── 参数 ──
    STOP_LOSS_PCT = 0.08
    TRAILING_STOP_PCT = 0.10
    REBALANCE_DAYS = 5
    VOL_HIGH_THRESHOLD = 0.25
    VOL_LOW_THRESHOLD = 0.15
    MAX_TOTAL_WEIGHT = 0.85

    # ── 1. 止损检查 ──
    surviving = {}
    for code, weight in current_holdings.items():
        if code not in etf_data or weight <= 0:
            continue
        df = etf_data[code]
        if len(df) == 0:
            continue
        current_price = df["close"].iloc[-1]
        entry = entry_prices.get(code, current_price)

        if entry > 0:
            pnl = current_price / entry - 1
            if pnl < -STOP_LOSS_PCT:
                continue
            # 盈利超过5%后启动滑动止损
            if pnl > 0.05:
                peak = df["close"].iloc[-20:].max() if len(df) >= 20 else current_price
                if peak > 0 and (current_price / peak - 1) < -TRAILING_STOP_PCT:
                    continue

        surviving[code] = weight

    # ── 2. 市场regime判断 ──
    # 用沪深300或最大宽基ETF的波动率判断
    benchmark_codes = ["510300.SH", "510500.SH", "510050.SH"]
    market_vol = 0.18
    market_trend = 0.0

    for bc in benchmark_codes:
        if bc in etf_data and len(etf_data[bc]) >= 60:
            close = etf_data[bc]["close"]
            returns = close.pct_change().dropna()
            if len(returns) >= 20:
                market_vol = returns.iloc[-20:].std() * np.sqrt(252)
                ma20 = close.iloc[-20:].mean()
                ma60 = close.iloc[-60:].mean() if len(close) >= 60 else ma20
                market_trend = (ma20 / ma60 - 1) if ma60 > 0 else 0
            break

    # regime分类
    if market_vol > VOL_HIGH_THRESHOLD and market_trend < -0.02:
        regime = "bear"
        max_weight = 0.30
    elif market_vol > VOL_HIGH_THRESHOLD:
        regime = "volatile"
        max_weight = 0.50
    elif market_trend > 0.03:
        regime = "bull"
        max_weight = MAX_TOTAL_WEIGHT
    else:
        regime = "neutral"
        max_weight = 0.65

    # ── 3. 是否需要调仓 ──
    day_of_month = current_date.day
    should_rebalance = (day_of_month % REBALANCE_DAYS <= 1) or len(surviving) == 0

    # 如果regime突变为bear，强制减仓
    total_current = sum(surviving.values())
    if total_current > max_weight:
        scale = max_weight / total_current
        surviving = {k: v * scale for k, v in surviving.items()}
        should_rebalance = True

    if not should_rebalance and len(surviving) > 0:
        return surviving

    # ── 4. 选标的 ──
    scores = {}
    dividend_etfs = ["510880.SH", "515080.SH", "512890.SH"]

    for code, df in etf_data.items():
        if len(df) < 60:
            continue
        close = df["close"]
        returns = close.pct_change().dropna()

        # 20日动量
        mom_20 = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0
        # 60日动量
        mom_60 = close.iloc[-1] / close.iloc[-60] - 1 if len(close) >= 60 else 0
        # 波动率
        vol = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.5
        # 成交量变化（放量信号）
        if "vol" in df.columns and len(df) >= 20:
            vol_ratio = df["vol"].iloc[-5:].mean() / df["vol"].iloc[-20:].mean()
        else:
            vol_ratio = 1.0

        # 综合评分（动量/波动率 + 放量加分）
        if vol > 0.01:
            base_score = (mom_20 * 0.5 + mom_60 * 0.3) / vol
            volume_bonus = max(0, (vol_ratio - 1.0) * 0.2) if mom_20 > 0 else 0
            scores[code] = base_score + volume_bonus
        else:
            scores[code] = 0

    # ── 5. 构建目标组合 ──
    target_weights = {}

    # 红利底仓（regime=bear时增加，bull时减少）
    if regime == "bear":
        div_each = 0.10
    elif regime == "bull":
        div_each = 0.05
    else:
        div_each = 0.08

    dividend_total = 0.0
    for code in dividend_etfs:
        if code in etf_data and len(etf_data[code]) > 60:
            target_weights[code] = div_each
            dividend_total += div_each

    # 动量轮动仓位
    remaining = max_weight - dividend_total
    if remaining > 0 and regime != "bear":
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_select = 5 if regime == "bull" else 3
        selected = []
        for code, score in sorted_scores:
            if code in target_weights:
                continue
            if score <= 0:
                break
            selected.append(code)
            if len(selected) >= n_select:
                break

        if selected:
            per_weight = remaining / len(selected)
            for code in selected:
                target_weights[code] = per_weight

    # 确保总权重不超限
    total = sum(target_weights.values())
    if total > max_weight:
        scale = max_weight / total
        target_weights = {k: v * scale for k, v in target_weights.items()}

    return target_weights
# EVOLVE-BLOCK-END
