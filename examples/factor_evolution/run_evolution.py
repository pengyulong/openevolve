#!/usr/bin/env python3
"""
A股量化因子自动演化系统

使用 OpenEvolve 框架，通过 LLM 驱动的进化算法自动发现高 IC 因子。

用法：
    python run_evolution.py                    # 默认50次迭代
    python run_evolution.py --iterations 200   # 200次迭代
    python run_evolution.py --test             # 仅测试初始因子评估
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from openevolve import run_evolution


def test_initial_factor():
    """测试初始因子评估"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    from evaluator import evaluate

    initial_path = os.path.join(base_dir, 'initial_factor.py')

    print("=" * 60)
    print("测试初始因子评估")
    print("=" * 60)

    result = evaluate(initial_path)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    key_metrics = [
        ("训练集 Rank IC", "train_rank_ic_mean"),
        ("训练集 IC_IR", "train_rank_ic_ir"),
        ("训练集 IC 胜率", "train_ic_win_rate"),
        ("验证集 Rank IC", "val_rank_ic_mean"),
        ("验证集 IC_IR", "val_rank_ic_ir"),
        ("综合评分", "combined_score"),
        ("因子覆盖率", "factor_coverage"),
        ("因子换手率", "factor_turnover"),
    ]

    for label, key in key_metrics:
        val = result.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {label}: {val:.4f}")
        else:
            print(f"  {label}: {val}")

    if result.get("_error"):
        print(f"\n  错误: {result['_error']}")

    print("=" * 60)
    return result


def run_factor_evolution(iterations: int = 50):
    """运行因子进化"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    initial_path = os.path.join(base_dir, 'initial_factor.py')
    evaluator_path = os.path.join(base_dir, 'evaluator.py')
    config_path = os.path.join(base_dir, 'config.yaml')
    output_path = os.path.join(base_dir, 'output')

    print("=" * 60)
    print("A股量化因子自动演化系统")
    print("=" * 60)
    print(f"  初始因子: {initial_path}")
    print(f"  评估器:   {evaluator_path}")
    print(f"  配置文件: {config_path}")
    print(f"  输出目录: {output_path}")
    print(f"  迭代次数: {iterations}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    print("\n开始因子演化...")
    result = run_evolution(
        initial_program=initial_path,
        evaluator=evaluator_path,
        config=config_path,
        iterations=iterations,
        output_dir=output_path,
        cleanup=False,
    )

    print("\n" + "=" * 60)
    print("演化完成!")
    print("=" * 60)
    print(f"  最佳评分: {result.best_score:.4f}")

    if result.metrics:
        ic = result.metrics.get("train_rank_ic_mean", "N/A")
        ir = result.metrics.get("train_rank_ic_ir", "N/A")
        val_ic = result.metrics.get("val_rank_ic_mean", "N/A")
        print(f"  训练集 IC: {ic}")
        print(f"  训练集 IR: {ir}")
        print(f"  验证集 IC: {val_ic}")

    print(f"\n  最佳因子代码:")
    print("-" * 40)
    # 只打印 EVOLVE-BLOCK 部分
    code = result.best_code
    if "EVOLVE-BLOCK-START" in code:
        start = code.index("EVOLVE-BLOCK-START")
        end = code.index("EVOLVE-BLOCK-END") + len("EVOLVE-BLOCK-END")
        print(code[start:end])
    else:
        print(code[:500])

    print("-" * 40)
    print(f"  结果保存在: {output_path}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="A股量化因子自动演化系统")
    parser.add_argument("--iterations", type=int, default=50, help="进化迭代次数")
    parser.add_argument("--test", action="store_true", help="仅测试初始因子评估")
    args = parser.parse_args()

    if args.test:
        test_initial_factor()
    else:
        run_factor_evolution(args.iterations)


if __name__ == "__main__":
    main()
