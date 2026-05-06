#!/usr/bin/env python3
"""
框架验证：使用 framework_evaluator.py 运行小规模因子演化实验

用法：
    python test_framework_run.py              # 默认5次迭代
    python test_framework_run.py --iterations 10  # 10次迭代
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from openevolve import run_evolution


def main():
    parser = argparse.ArgumentParser(description="框架验证实验")
    parser.add_argument("--iterations", type=int, default=5,
                        help="进化迭代次数（默认5）")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    initial_path = os.path.join(base_dir, "initial_factor.py")
    evaluator_path = os.path.join(base_dir, "framework_evaluator.py")
    config_path = os.path.join(base_dir, "config.yaml")
    output_path = os.path.join(base_dir, "output_framework_test")

    print("=" * 60)
    print("  框架验证实验 - A股量化因子演化")
    print("=" * 60)
    print(f"  评估器:   framework_evaluator.py (BaseEvaluator)")
    print(f"  诊断引擎: DiagnosticsEngine (P0/P1/P2)")
    print(f"  评分配置: ScoringConfig")
    print(f"  初始因子: initial_factor.py (BP单因子)")
    print(f"  迭代次数: {args.iterations}")
    print(f"  输出目录: {output_path}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    print("\n开始演化...\n")
    result = run_evolution(
        initial_program=initial_path,
        evaluator=evaluator_path,
        config=config_path,
        iterations=args.iterations,
        output_dir=output_path,
        cleanup=False,
    )

    print("\n" + "=" * 60)
    print("  演化完成!")
    print("=" * 60)
    print(f"  最佳评分: {result.best_score:.4f}")

    if result.metrics:
        metrics = result.metrics
        ic = metrics.get("train_rank_ic_mean", "N/A")
        ir = metrics.get("train_rank_ic_ir", "N/A")
        val_ic = metrics.get("val_rank_ic_mean", "N/A")
        print(f"  训练集 IC: {ic}")
        print(f"  训练集 IR: {ir}")
        print(f"  验证集 IC: {val_ic}")
        print(f"  Framework版本: {metrics.get('_framework_version', 'N/A')}")

    print(f"\n  最佳因子代码:")
    print("-" * 40)
    code = result.best_code
    if "EVOLVE-BLOCK-START" in code:
        start = code.index("EVOLVE-BLOCK-START")
        end = code.index("EVOLVE-BLOCK-END") + len("EVOLVE-BLOCK-END")
        print(code[start:end])
    else:
        print(code[:500])
    print("-" * 40)
    print(f"\n  结果保存在: {output_path}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
