"""
评估最佳因子在2015-2025年沪深300上的月度IC表现

用法：
    python evaluate_best_factor.py
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_extended_data(
    token: str,
    start_date: str = "20150101",
    end_date: str = "20251231",
    max_stocks: int = 300,
) -> Dict[str, pd.DataFrame]:
    """下载2015-2025年沪深300数据"""
    from data_loader import TushareDataLoader

    loader = TushareDataLoader(token=token)

    print("获取HS300成分股...")
    codes = loader.get_stock_pool("hs300")[:max_stocks]
    print(f"将下载 {len(codes)} 只股票 ({start_date} ~ {end_date})")

    print("下载数据中（首次较慢，后续读取缓存）...")
    data = loader.load_stock_pool_data(
        codes, start_date, end_date,
        add_indicators=True,
        show_progress=True,
    )
    print(f"成功加载 {len(data)} 只股票")
    return data


def evaluate_factor_monthly(
    stock_data: Dict[str, pd.DataFrame],
    compute_func,
    forward_period: int = 5,
    min_stocks_per_day: int = 30,
) -> Dict:
    """
    完整评估因子，返回月度IC分解

    Returns:
        Dict with:
        - monthly_ic: Series (year-month -> mean IC)
        - overall_metrics: dict with IC mean, IR, win_rate, etc.
        - ic_series: full daily IC series
        - yearly_summary: DataFrame with yearly IC stats
    """
    logger.info("构建未来收益矩阵...")
    # Build forward returns
    series_dict = {}
    for code, df in stock_data.items():
        if "close" in df.columns:
            fwd = df["close"].pct_change(forward_period).shift(-forward_period)
            series_dict[code] = fwd
    returns_panel = pd.DataFrame(series_dict).sort_index()
    logger.info(f"收益矩阵: {returns_panel.shape}")

    logger.info("计算因子面板...")
    # Compute factor panel
    factor_dict = {}
    for code, df in stock_data.items():
        try:
            fv = compute_func(df)
            if fv is not None and isinstance(fv, pd.Series):
                factor_dict[code] = fv
        except Exception:
            continue
    factor_panel = pd.DataFrame(factor_dict).sort_index()
    logger.info(f"因子面板: {factor_panel.shape}")

    # Align dates and stocks
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_stocks = factor_panel.columns.intersection(returns_panel.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        logger.error("No common dates/stocks between factor and returns")
        return {}

    factor_aligned = factor_panel.loc[common_dates, common_stocks]
    returns_aligned = returns_panel.loc[common_dates, common_stocks]

    logger.info(f"对齐后: {len(common_dates)} 个交易日, {len(common_stocks)} 只股票")

    # Calculate daily cross-sectional Spearman IC
    logger.info("计算逐日截面Rank IC...")
    ic_records = []
    for date in common_dates:
        f_row = factor_aligned.loc[date].dropna()
        r_row = returns_aligned.loc[date].dropna()
        valid = f_row.index.intersection(r_row.index)
        if len(valid) < min_stocks_per_day:
            continue
        f_vals = f_row[valid].values
        r_vals = r_row[valid].values
        # Winsorize 1%-99%
        f_vals = np.clip(f_vals, np.percentile(f_vals, 1), np.percentile(f_vals, 99))
        try:
            corr, _ = stats.spearmanr(f_vals, r_vals)
            if not np.isnan(corr):
                ic_records.append({"date": date, "ic": corr})
        except Exception:
            continue

    ic_df = pd.DataFrame(ic_records).set_index("date")
    ic_series = ic_df["ic"].sort_index()

    logger.info(f"有效IC日: {len(ic_series)}")

    if len(ic_series) < 60:
        logger.warning("IC序列太短，可能数据不足")
        return {}

    # Auto-flip: if overall IC is negative, flip
    auto_flipped = False
    if ic_series.mean() < 0:
        ic_series = -ic_series
        auto_flipped = True
        logger.info("IC方向已自动翻转")

    # Monthly aggregation
    monthly_ic = ic_series.groupby(pd.Grouper(freq="ME")).mean()
    # Format as year-month
    monthly_ic.index = monthly_ic.index.strftime("%Y-%m")

    # Yearly summary
    ic_series_yearly = ic_series.copy()
    yearly_groups = ic_series_yearly.groupby(ic_series_yearly.index.year)

    yearly_summary = []
    for year, group in yearly_groups:
        if len(group) < 10:
            continue
        yearly_summary.append({
            "year": year,
            "mean_ic": group.mean(),
            "std_ic": group.std(),
            "ir": group.mean() / group.std() if group.std() > 0 else 0,
            "win_rate": (group > 0).sum() / len(group),
            "n_days": len(group),
            "n_months": group.groupby(group.index.month).ngroups,
        })
    yearly_df = pd.DataFrame(yearly_summary).set_index("year")

    # Overall metrics
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_win_rate = (ic_series > 0).sum() / len(ic_series)

    # Monthly stats
    monthly_mean = monthly_ic.mean()
    monthly_std = monthly_ic.std()
    monthly_positive_ratio = (monthly_ic > 0).sum() / len(monthly_ic)

    overall = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_win_rate": ic_win_rate,
        "ic_count": len(ic_series),
        "monthly_ic_mean": monthly_mean,
        "monthly_ic_std": monthly_std,
        "monthly_positive_ratio": monthly_positive_ratio,
        "monthly_count": len(monthly_ic),
        "auto_flipped": auto_flipped,
        "date_range_start": str(ic_series.index.min().date()),
        "date_range_end": str(ic_series.index.max().date()),
    }

    return {
        "monthly_ic": monthly_ic,
        "overall_metrics": overall,
        "ic_series": ic_series,
        "yearly_summary": yearly_df,
    }


def print_results(results: Dict):
    """打印月度IC评估结果"""
    overall = results["overall_metrics"]
    monthly_ic = results["monthly_ic"]
    yearly_df = results["yearly_summary"]

    print("\n" + "=" * 70)
    print("  因子长期稳定性评估 (2015-2025)")
    print("=" * 70)
    print(f"  日期范围: {overall['date_range_start']} ~ {overall['date_range_end']}")
    print(f"  有效IC日: {overall['ic_count']} 天")
    print(f"  有效月数: {overall['monthly_count']} 个月")
    if overall['auto_flipped']:
        print(f"  IC方向已自动翻转")

    print("\n  ═══ 整体指标 ═══")
    print(f"  日均IC (Rank):  {overall['ic_mean']:.4f}")
    print(f"  IC标准差:       {overall['ic_std']:.4f}")
    print(f"  IC信息比 (IR):  {overall['ic_ir']:.3f}")
    print(f"  IC胜率:         {overall['ic_win_rate']:.1%}")
    print(f"  月度IC均值:     {overall['monthly_ic_mean']:.4f}")
    print(f"  月度IC标准差:   {overall['monthly_ic_std']:.4f}")
    print(f"  月度IC正率:     {overall['monthly_positive_ratio']:.1%}")

    print("\n  ═══ 年度IC汇总 ═══")
    print(f"  {'年份':<6} {'均值IC':>8} {'标准差':>8} {'IR':>8} {'胜率':>8} {'天数':>6}")
    print("  " + "-" * 50)
    for year, row in yearly_df.iterrows():
        print(f"  {year:<6} {row['mean_ic']:>8.4f} {row['std_ic']:>8.4f} {row['ir']:>8.3f} {row['win_rate']:>8.1%} {int(row['n_days']):>6}")

    print("\n  ═══ 月度IC明细 ═══")
    # Print in a matrix format (year × month)
    monthly_df = monthly_ic.reset_index()
    monthly_df.columns = ["month", "ic"]
    monthly_df["year"] = monthly_df["month"].str[:4]
    monthly_df["month_num"] = monthly_df["month"].str[5:7]

    # Pivot table
    pivot = monthly_df.pivot_table(
        values="ic", index="year", columns="month_num", aggfunc="first"
    )
    pivot = pivot[
        [f"{m:02d}" for m in range(1, 13)]  # Ensure all months
    ]

    # Header
    print("  " + " " * 5 + "".join(f"{'M'+m:>7}" for m in ["01","02","03","04","05","06","07","08","09","10","11","12"]))
    print("  " + " " * 5 + "-" * 84)

    for year in pivot.index:
        row_str = f"  {year} "
        for m in [f"{m:02d}" for m in range(1, 13)]:
            val = pivot.loc[year, m]
            if pd.isna(val):
                row_str += f"   {'·':>5}"
            else:
                if val >= 0.08:
                    marker = f" {val:+.3f}★"
                elif val >= 0.05:
                    marker = f" {val:+.3f}▲"
                elif val >= 0.02:
                    marker = f"  {val:+.3f}"
                elif val >= 0:
                    marker = f"  {val:+.3f} "
                elif val >= -0.02:
                    marker = f" {val:+.3f}"
                else:
                    marker = f" {val:+.3f}✗"
                row_str += marker
        # Year total
        year_total = pivot.loc[year].mean()
        row_str += f"  │{year_total:+.3f}"
        print(row_str)

    print("  " + " " * 5 + "-" * 84)
    # Last row: month averages
    avg_row = "  Avg  "
    for m in [f"{m:02d}" for m in range(1, 13)]:
        vals = pivot[m].dropna()
        if len(vals) > 0:
            avg_row += f" {vals.mean():+.3f}"
        else:
            avg_row += f"   ·   "
    print(avg_row)

    print(f"\n  ★ IC>=0.08  ▲ IC>=0.05  · 无数据  ✗ IC<-0.02")

    # Distribution analysis
    print("\n  ═══ 月度IC分布分析 ═══")
    ic_vals = monthly_ic.dropna()
    print(f"  月度IC最大值:  {ic_vals.max():.4f}  ({ic_vals.idxmax()})")
    print(f"  月度IC最小值:  {ic_vals.min():.4f}  ({ic_vals.idxmin()})")
    print(f"  月度IC中位数:  {ic_vals.median():.4f}")

    # Count months by IC range
    ranges = [
        ("IC >= 0.10", ic_vals >= 0.10),
        ("0.08 <= IC < 0.10", (ic_vals >= 0.08) & (ic_vals < 0.10)),
        ("0.05 <= IC < 0.08", (ic_vals >= 0.05) & (ic_vals < 0.08)),
        ("0.02 <= IC < 0.05", (ic_vals >= 0.02) & (ic_vals < 0.05)),
        ("0.00 <= IC < 0.02", (ic_vals >= 0.00) & (ic_vals < 0.02)),
        ("-0.02 <= IC < 0.00", (ic_vals >= -0.02) & (ic_vals < 0.00)),
        ("IC < -0.02", ic_vals < -0.02),
    ]
    for label, mask in ranges:
        count = mask.sum()
        pct = count / len(ic_vals) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<25} {count:>3} 个月 ({pct:5.1f}%) {bar}")

    # Market regime analysis
    print("\n  ═══ 市场环境适应性 ═══")
    # 2015: bull-bear, 2016: recovery, 2017: blue-chip bull
    # 2018: bear, 2019-2020: recovery/tech, 2021: structural
    # 2022: bear, 2023: recovery, 2024: volatile
    regimes = {
        "2015大起大落": "2015",
        "2016-2017蓝筹慢牛": "2016-2017",
        "2018熊市": "2018",
        "2019-2020结构牛": "2019-2020",
        "2021-2023震荡市": "2021-2023",
        "2024": "2024",
    }
    for regime, year_filter in regimes.items():
        subset = monthly_ic[monthly_ic.index.str[:4].isin(year_filter.split("-"))]
        if len(subset) > 0:
            print(f"  {regime:<20} 月均值IC={subset.mean():.4f}  "
                  f"正比={ (subset>0).sum()}/{len(subset)} = {(subset>0).sum()/len(subset):.0%}")


def main():
    import yaml

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    token = config.get("tushare_token", "")
    if not token:
        logger.error("No tushare token configured")
        sys.exit(1)

    # ─── Step 1: Download extended data (2015-2025) ───
    extended_cache = os.path.join(base_dir, "data_cache", "stock_data_2015_2025.pkl")

    if os.path.exists(extended_cache):
        print(f"从缓存加载扩展数据: {extended_cache}")
        with open(extended_cache, "rb") as f:
            stock_data = pickle.load(f)
        print(f"加载完成: {len(stock_data)} 只股票")
    else:
        print("下载2015-2025年数据...")
        stock_data = download_extended_data(
            token=token,
            start_date="20150101",
            end_date="20251231",
            max_stocks=300,
        )
        if stock_data:
            print(f"保存缓存到 {extended_cache}")
            with open(extended_cache, "wb") as f:
                pickle.dump(stock_data, f)

    if not stock_data:
        logger.error("没有加载到任何股票数据")
        sys.exit(1)

    # ─── Step 2: Load best factor ───
    best_factor_path = os.path.join(base_dir, "output", "best", "best_program.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("best_factor", best_factor_path)
    best_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(best_module)

    if hasattr(best_module, "get_factor_info"):
        print(f"\n因子信息: {best_module.get_factor_info()}")

    # ─── Step 3: Evaluate with monthly IC ───
    results = evaluate_factor_monthly(
        stock_data=stock_data,
        compute_func=best_module.compute_factor,
        forward_period=config.get("forward_period", 5),
        min_stocks_per_day=30,
    )

    if not results:
        logger.error("评估失败")
        sys.exit(1)

    # ─── Step 4: Print results ───
    print_results(results)

    # ─── Step 5: Save detailed CSV ───
    output_dir = os.path.join(base_dir, "output_test")
    os.makedirs(output_dir, exist_ok=True)

    monthly_ic = results["monthly_ic"]
    monthly_ic.to_csv(os.path.join(output_dir, "monthly_ic_2015_2025.csv"), header=["mean_ic"])
    results["yearly_summary"].to_csv(os.path.join(output_dir, "yearly_ic_summary.csv"))

    print(f"\n详细数据已保存至 {output_dir}/")


if __name__ == "__main__":
    main()
