"""
初始因子 - 六维度多因子模型（正IC方向）

覆盖六大类因子维度，作为进化的高质量起点：
- 价值(BP): 1/PB，A股最稳健价值因子，训练IC=+0.044
- 盈利(EP): 1/PE_ttm，盈利收益率，质量+价值双维度
- 低波动: -60日收益率标准差，低风险溢价
- 反转: 波动率调整5日反转，行为金融
- 小市值: -log总市值，A股小盘效应
- 低换手: -换手率，低流动性溢价

每个子因子: 缩尾(1%/99%) → EWM平滑 → 截面排名
最终组合: 等权 → 截面排名
"""

import pandas as pd
import numpy as np
from typing import Dict


# EVOLVE-BLOCK-START
def compute_factor(data: pd.DataFrame) -> pd.Series:
    """
    计算量化因子

    Args:
        data: 单只股票的日线数据 DataFrame，包含以下列：
              价格: close, open, high, low, pre_close
              量价: vol, amount, turnover_rate, turnover_rate_f
              基本面: pe, pe_ttm, pb, ps, ps_ttm, total_mv, circ_mv
              技术指标: ma5, ma10, ma20, ma60, ema12, ema26
                       macd, macd_signal, macd_histogram
                       rsi, kdj_k, kdj_d, kdj_j
                       bb_upper, bb_middle, bb_lower
              衍生: returns_1d, returns_5d, returns_20d
                    vol_ma5, vol_ma20, vol_ratio
                    amplitude, upper_shadow, lower_shadow
                    ma5_slope, ma20_slope

    Returns:
        pd.Series: 因子值序列（与 data 同 index）
    """
    # 日收益率
    ret = data['close'].pct_change()

    # --- 子因子1: BP (账面市值比, 1/PB) ---
    pb = data['pb'].replace(0, np.nan)
    bp = 1.0 / pb
    bp_lo, bp_hi = bp.quantile(0.01), bp.quantile(0.99)
    bp = bp.clip(lower=bp_lo, upper=bp_hi)
    bp = bp.ewm(span=30, min_periods=15, adjust=False).mean()
    rank_bp = bp.rank(pct=True)

    # --- 子因子2: EP (盈利收益率, 1/PE_ttm) ---
    pe = data['pe_ttm'].replace([0, np.inf, -np.inf], np.nan)
    ep = 1.0 / pe
    ep_lo, ep_hi = ep.quantile(0.01), ep.quantile(0.99)
    ep = ep.clip(lower=ep_lo, upper=ep_hi)
    ep = ep.ewm(span=30, min_periods=15, adjust=False).mean()
    rank_ep = ep.rank(pct=True)

    # --- 子因子3: 低波动 (-60日收益标准差) ---
    vol60 = ret.rolling(60, min_periods=20).std()
    low_vol = -vol60
    low_vol = low_vol.ewm(span=30, min_periods=15, adjust=False).mean()
    rank_lowvol = low_vol.rank(pct=True)

    # --- 子因子4: 波动率调整的反转 ---
    ret5 = data['close'].pct_change(5)
    vol20 = ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    reversal = -ret5 / (vol20 + 1e-8)
    rev_lo, rev_hi = reversal.quantile(0.01), reversal.quantile(0.99)
    reversal = reversal.clip(lower=rev_lo, upper=rev_hi)
    reversal = reversal.ewm(span=15, min_periods=8, adjust=False).mean()
    rank_rev = reversal.rank(pct=True)

    # --- 子因子5: 小市值 (-log总市值) ---
    total_mv = data['total_mv'].replace(0, np.nan).ffill()
    log_mv = np.log(total_mv)
    size = -log_mv
    sz_lo, sz_hi = size.quantile(0.01), size.quantile(0.99)
    size = size.clip(lower=sz_lo, upper=sz_hi)
    size = size.ewm(span=60, min_periods=20, adjust=False).mean()
    rank_size = size.rank(pct=True)

    # --- 子因子6: 低换手率 (-换手率, 低流动性溢价) ---
    turnover = data['turnover_rate_f'].replace(0, np.nan)
    illiq = -turnover
    il_lo, il_hi = illiq.quantile(0.01), illiq.quantile(0.99)
    illiq = illiq.clip(lower=il_lo, upper=il_hi)
    illiq = illiq.ewm(span=20, min_periods=10, adjust=False).mean()
    rank_illiq = illiq.rank(pct=True)

    # --- 等权组合 ---
    factor_raw = (rank_bp + rank_ep + rank_lowvol + rank_rev + rank_size + rank_illiq) / 6.0
    factor = factor_raw.rank(pct=True)
    return factor
# EVOLVE-BLOCK-END


def get_factor_info() -> Dict[str, str]:
    """返回因子元信息"""
    return {
        "name": "six_factor_multi_dim",
        "description": "六维度多因子：价值(BP)+盈利(EP)+低波动+反转+小市值+低换手，截面排名等权组合",
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import TushareDataLoader

    TOKEN = os.environ.get("TUSHARE_TOKEN", "")
    if not TOKEN:
        print("请设置环境变量 TUSHARE_TOKEN")
        sys.exit(1)
    loader = TushareDataLoader(token=TOKEN)

    data = loader.load_stock_data("600519.SH", "20220101", "20240101")
    if data is not None:
        factor = compute_factor(data)
        valid = factor.dropna()
        print("=== 初始因子测试 ===")
        print(f"因子长度: {len(valid)}")
        print(f"均值: {valid.mean():.4f}")
        print(f"标准差: {valid.std():.4f}")
        print(f"最小/最大: {valid.min():.4f} / {valid.max():.4f}")
