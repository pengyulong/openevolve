#!/usr/bin/env python3
"""
ETF数据一键下载脚本

下载A股全市场可交易ETF的行情+份额数据，存储到本地缓存。
覆盖 2018-01-01 ~ 2026-05-22，包含完整牛熊周期。
"""

import os
import sys
import time
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from etf_data_loader import ETFDataLoader, ETF_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──
START_DATE = "20150101"
END_DATE = "20260101"
MIN_LIST_DAYS = 252       # 最少上市1年
MIN_AVG_AMOUNT = 10000    # 日均成交额 > 1000万（单位千元）

# 重点关注的核心ETF（确保一定下载）
CORE_ETFS = [
    # 宽基
    "510300.SH",  # 华泰柏瑞沪深300ETF
    "510500.SH",  # 南方中证500ETF
    "510050.SH",  # 华夏上证50ETF
    "159919.SZ",  # 嘉实沪深300ETF
    "588000.SH",  # 华夏科创50ETF
    "159915.SZ",  # 易方达创业板ETF
    "512100.SH",  # 南方中证1000ETF
    "560610.SH",  # 中证A500ETF
    # 红利
    "515080.SH",  # 中证红利ETF
    "510880.SH",  # 上证红利ETF
    "512890.SH",  # 红利低波ETF
    "159691.SZ",  # 港股红利ETF
    # 行业 - 金融
    "512800.SH",  # 银行ETF
    "512880.SH",  # 证券ETF
    # 行业 - 消费
    "159928.SZ",  # 消费ETF
    "512690.SH",  # 白酒ETF
    "159996.SZ",  # 家电ETF
    # 行业 - 医药
    "512010.SH",  # 医药ETF
    "159992.SZ",  # 创新药ETF
    # 行业 - 科技
    "512480.SH",  # 半导体ETF
    "515030.SH",  # 新能源车ETF
    "516160.SH",  # 新能源ETF
    "515790.SH",  # 光伏ETF
    "512660.SH",  # 军工ETF
    "159770.SZ",  # 芯片ETF
    # 行业 - 周期
    "515220.SH",  # 煤炭ETF
    "512400.SH",  # 有色ETF
    "512050.SH",  # 钢铁ETF
    # 行业 - 地产基建
    "512200.SH",  # 地产ETF
    "516950.SH",  # 基建ETF
    # 主题
    "515070.SH",  # 人工智能ETF
    "562500.SH",  # 机器人ETF
    "159786.SZ",  # 云计算ETF
]

# 对标指数
BENCHMARK_INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "000015.SH": "红利指数",
}


def main():
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        print("ERROR: 请设置 TUSHARE_TOKEN 环境变量")
        sys.exit(1)

    loader = ETFDataLoader(token=token)

    print("=" * 60)
    print("A股ETF数据下载")
    print("=" * 60)
    print(f"  数据范围: {START_DATE} ~ {END_DATE}")
    print(f"  最少上市: {MIN_LIST_DAYS} 天")
    print(f"  最低流动性: 日均成交额 > {MIN_AVG_AMOUNT/10:.0f}万元")
    print("=" * 60)

    # Step 1: 获取ETF列表
    print("\n[1/5] 获取ETF列表...")
    etf_list = loader.get_tradeable_etfs(min_list_days=MIN_LIST_DAYS)
    print(f"  上市中ETF: {len(etf_list)} 只")

    # Step 2: 分类
    print("\n[2/5] ETF分类...")
    categories = loader.classify_etfs(etf_list)
    for cat, codes in categories.items():
        print(f"  {cat}: {len(codes)} 只")

    # Step 3: 确定下载列表（核心ETF + 分类筛选）
    all_codes = set(CORE_ETFS)
    for cat, codes in categories.items():
        if cat != "其他":
            all_codes.update(codes)

    # 确保所有核心ETF都在列表中
    download_codes = sorted(all_codes)
    print(f"\n  将下载 {len(download_codes)} 只ETF")

    # Step 4: 批量下载ETF数据
    print(f"\n[3/5] 下载ETF行情+份额数据...")
    print(f"  （首次下载较慢，约需 {len(download_codes) * 0.8 / 60:.0f} 分钟）")

    etf_data = loader.load_etf_pool_data(
        download_codes,
        START_DATE,
        END_DATE,
        include_share=True,
        show_progress=True,
    )
    print(f"  成功下载: {len(etf_data)} 只ETF")

    # 流动性筛选
    etf_data_filtered = loader.filter_by_liquidity(etf_data, min_avg_amount=MIN_AVG_AMOUNT)
    print(f"  流动性筛选后: {len(etf_data_filtered)} 只ETF")

    # Step 5: 下载指数数据
    print(f"\n[4/5] 下载对标指数数据...")
    for idx_code, idx_name in BENCHMARK_INDICES.items():
        idx_df = loader.load_index_data(idx_code, START_DATE, END_DATE)
        if idx_df is not None:
            print(f"  {idx_name} ({idx_code}): {len(idx_df)} 天")
        else:
            print(f"  {idx_name} ({idx_code}): 下载失败")

    # Step 6: 下载北向资金
    print(f"\n[5/5] 下载北向资金数据...")
    north_df = loader.load_northbound_flow(START_DATE, END_DATE)
    if north_df is not None:
        print(f"  北向资金数据: {len(north_df)} 天")
    else:
        print("  北向资金数据下载失败")

    # ── 输出摘要 ──
    print("\n" + "=" * 60)
    print("下载完成！数据摘要")
    print("=" * 60)

    # 统计
    total_rows = sum(len(df) for df in etf_data_filtered.values())
    sample_code = list(etf_data_filtered.keys())[0] if etf_data_filtered else None

    print(f"\n  ETF总数: {len(etf_data_filtered)} 只")
    print(f"  总数据行数: {total_rows:,}")
    print(f"  数据日期: {START_DATE} ~ {END_DATE}")
    print(f"  缓存目录: {loader.cache_dir}")

    if sample_code:
        sample_df = etf_data_filtered[sample_code]
        print(f"\n  示例ETF ({sample_code}):")
        print(f"    行数: {len(sample_df)}")
        print(f"    日期: {sample_df.index.min().strftime('%Y-%m-%d')} ~ {sample_df.index.max().strftime('%Y-%m-%d')}")
        print(f"    列数: {len(sample_df.columns)}")
        print(f"    列名: {sample_df.columns.tolist()}")

    # 分类统计
    print(f"\n  分类统计:")
    cat_stats = {cat: 0 for cat in ETF_CATEGORIES}
    cat_stats["其他"] = 0
    etf_names = etf_list.set_index("ts_code")["name"].to_dict() if not etf_list.empty else {}

    for code in etf_data_filtered.keys():
        matched = False
        name = etf_names.get(code, "")
        for cat, config in ETF_CATEGORIES.items():
            if any(kw.lower() in name.lower() for kw in config["keywords"]):
                cat_stats[cat] += 1
                matched = True
                break
        if not matched:
            cat_stats["其他"] += 1

    for cat, count in cat_stats.items():
        if count > 0:
            print(f"    {cat}: {count} 只")

    # 列出核心ETF状态
    print(f"\n  核心ETF下载状态:")
    for code in CORE_ETFS[:15]:
        name = etf_names.get(code, "未知")
        status = "OK" if code in etf_data_filtered else ("低流动性" if code in etf_data else "失败")
        if code in etf_data_filtered:
            days = len(etf_data_filtered[code])
            print(f"    {code} {name:<12} {status} ({days}天)")
        else:
            print(f"    {code} {name:<12} {status}")

    print("\n" + "=" * 60)
    print("数据就绪，可用于 ETF 策略演化。")
    print("=" * 60)


if __name__ == "__main__":
    main()
