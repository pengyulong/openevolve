#!/usr/bin/env python3
"""检查演化进度并输出报告"""
import os
import json
import sys
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
checkpoints_dir = os.path.join(output_dir, "checkpoints")

TARGET_CHECKPOINTS = [50, 100, 150, 200]

# 记录已报告过的 checkpoint
reported_file = os.path.join(output_dir, ".reported_checkpoints")
reported = set()
if os.path.exists(reported_file):
    with open(reported_file) as f:
        reported = set(line.strip() for line in f if line.strip())

new_reports = []

for cp in TARGET_CHECKPOINTS:
    cp_dir = os.path.join(checkpoints_dir, f"checkpoint_{cp}")
    info_file = os.path.join(cp_dir, "best_program_info.json")

    if not os.path.exists(info_file):
        continue

    if str(cp) in reported:
        continue

    with open(info_file) as f:
        info = json.load(f)

    m = info.get("metrics", {})
    score = m.get("combined_score", 0)
    train_ic = m.get("train_rank_ic_mean", 0)
    train_ir = m.get("train_rank_ic_ir", 0)
    val_ic = m.get("val_rank_ic_mean", 0)
    val_ir = m.get("val_rank_ic_ir", 0)
    gen = info.get("generation", "?")
    iteration = info.get("iteration", "?")
    diag = m.get("_diagnostics_prio", "")

    report = f"""
{'='*60}
  演化进度报告 — Checkpoint {cp}
  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
  当前代数:   Gen {gen} (Iteration {iteration})
  综合评分:   {score:.4f}
  训练集 IC:  {train_ic:+.4f}
  训练集 IR:  {train_ir:.3f}
  验证集 IC:  {val_ic:+.4f}
  验证集 IR:  {val_ir:.3f}
  诊断概要:   {diag[:200]}
{'='*60}
"""
    new_reports.append(report)
    reported.add(str(cp))

if new_reports:
    for r in new_reports:
        print(r)

    with open(reported_file, "w") as f:
        for r in sorted(reported):
            f.write(r + "\n")
    sys.exit(0)
else:
    # 静默：还没有新的 checkpoint 达到目标
    # 打印当前最新进度
    if os.path.exists(checkpoints_dir):
        existing = sorted([
            int(d.replace("checkpoint_", ""))
            for d in os.listdir(checkpoints_dir)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoints_dir, d))
        ])
        if existing:
            # 读取最新 checkpoint
            latest = existing[-1]
            info_file = os.path.join(checkpoints_dir, f"checkpoint_{latest}", "best_program_info.json")
            if os.path.exists(info_file):
                with open(info_file) as f:
                    info = json.load(f)
                m = info.get("metrics", {})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 当前最新: checkpoint_{latest} | "
                      f"Score={m.get('combined_score', 0):.4f} | "
                      f"TrainIC={m.get('train_rank_ic_mean', 0):+.4f} | "
                      f"ValIC={m.get('val_rank_ic_mean', 0):+.4f}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 最新 checkpoint_{latest}，但无 best_program_info.json")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 暂无 checkpoint 生成...")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 暂无 checkpoint 生成...")
