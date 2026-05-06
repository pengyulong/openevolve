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
import pandas as pd
import numpy as np

def compute_factor(data: pd.DataFrame) -> pd.Series:
    """
    四因子稳健组合（去噪优化版）：价值(BP) + 波动率调整反转 + 低波动 + 小市值
    
    针对验证集IR偏低的问题进行优化：
    - 降低反转因子权重（从0.30降至0.20），减少噪声
    - 延长各子因子EWM平滑窗口，提升稳定性
    - 去掉成交量确认（vol_ratio），简化反转公式
    - 低波动窗口从40日延长至50日
    - 小市值窗口从40日延长至60日
    - BP权重提升至0.40，强化价值驱动
    
    每个子因子：缩尾(1%/99%) → EWM平滑 → 截面排名
    组合权重：BP 0.40，反转 0.20，低波动 0.25，小市值 0.15
    最终：加权组合 → 截面排名
    """
    def winsorize(series, low=0.01, high=0.99):
        lo = series.quantile(low)
        hi = series.quantile(high)
        return series.clip(lower=lo, upper=hi)

    ret = data['close'].pct_change()

    # --- 子因子1: BP (账面市值比) ---
    pb = data['pb'].replace(0, np.nan)
    bp = 1.0 / pb
    bp = winsorize(bp, 0.01, 0.99)
    bp_smooth = bp.ewm(span=80, min_periods=30, adjust=False).mean()
    rank_bp = bp_smooth.rank(pct=True)

    # --- 子因子2: 波动率调整反转 (去掉成交量，更稳健) ---
    ret5 = data['close'].pct_change(5)
    vol20 = ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    rev_raw = -ret5 / (vol20 + 1e-8)
    rev = winsorize(rev_raw, 0.01, 0.99)
    rev_smooth = rev.ewm(span=30, min_periods=12, adjust=False).mean()
    rank_rev = rev_smooth.rank(pct=True)

    # --- 子因子3: 低波动率 (50日窗口，提升稳定性) ---
    vol_raw = -ret.rolling(50, min_periods=25).std()
    vol = winsorize(vol_raw, 0.01, 0.99)
    vol_smooth = vol.ewm(span=50, min_periods=25, adjust=False).mean()
    rank_vol = vol_smooth.rank(pct=True)

    # --- 子因子4: 小市值 (延长平滑窗口) ---
    mv = data['total_mv'] if 'total_mv' in data.columns and data['total_mv'].notna().any() else data['circ_mv']
    mv = mv.replace(0, np.nan)
    log_mv = np.log(mv)
    size = -log_mv  # 小市值 → 大因子值
    size = winsorize(size, 0.01, 0.99)
    size_smooth = size.ewm(span=60, min_periods=30, adjust=False).mean()
    rank_size = size_smooth.rank(pct=True)

    # --- 加权组合 (BP:0.40, 反转:0.20, 低波动:0.25, 小市值:0.15) ---
    factor_raw = 0.40 * rank_bp + 0.20 * rank_rev + 0.25 * rank_vol + 0.15 * rank_size
    factor = factor_raw.rank(pct=True).astype(float)
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
