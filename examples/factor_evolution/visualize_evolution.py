#!/usr/bin/env python3
"""
因子演化轨迹可视化仪表盘 (Streamlit)

用法:
    streamlit run visualize_evolution.py

输入: 选择一个演化输出目录 (包含 checkpoints/ 子目录)
输出: 交互式演化过程可视化
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_checkpoint_data(output_dir: str) -> Dict:
    """加载所有 checkpoint 数据"""
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return {}

    checkpoints = {}
    for cp_dir in sorted(checkpoints_dir.iterdir(), key=lambda x: int(x.name.split("_")[1])):
        if not cp_dir.is_dir():
            continue
        iteration = int(cp_dir.name.split("_")[1])

        # 加载 metadata
        meta_path = cp_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        # 加载最佳程序信息
        best_path = cp_dir / "best_program_info.json"
        best_info = {}
        if best_path.exists():
            with open(best_path) as f:
                best_info = json.load(f)

        # 加载所有程序
        programs = {}
        progs_dir = cp_dir / "programs"
        if progs_dir.exists():
            for prog_file in progs_dir.glob("*.json"):
                with open(prog_file) as f:
                    prog = json.load(f)
                    programs[prog["id"]] = prog

        checkpoints[iteration] = {
            "metadata": metadata,
            "best_info": best_info,
            "programs": programs,
        }

    return checkpoints


@st.cache_data
def load_knowledge_base_data(output_dir: str) -> Dict:
    """加载知识库数据"""
    kb_dir = Path(output_dir).parent / "knowledge_base"
    kb_db = kb_dir / "kb_store.db"

    result = {"entries": [], "writebacks": []}

    if not kb_db.exists():
        return result

    try:
        import sqlite3
        conn = sqlite3.connect(str(kb_db))
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT * FROM knowledge_entries WHERE status='active' ORDER BY usage_count DESC"
        ).fetchall()
        result["entries"] = [dict(r) for r in rows]

        try:
            wb_rows = conn.execute(
                "SELECT * FROM writeback_queue ORDER BY created_at"
            ).fetchall()
            result["writebacks"] = [dict(r) for r in wb_rows]
        except Exception:
            pass

        conn.close()
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════
# 数据处理
# ═══════════════════════════════════════════════════════════

def build_metrics_timeline(checkpoints: Dict) -> pd.DataFrame:
    """从检查点构建指标时间线"""
    rows = []
    for iteration in sorted(checkpoints.keys()):
        cp = checkpoints[iteration]
        best = cp["best_info"]
        metrics = best.get("metrics", {})

        # 计算该检查点的种群统计
        programs = cp["programs"]
        all_scores = [
            p["metrics"].get("combined_score", 0)
            for p in programs.values()
            if p.get("metrics", {}).get("combined_score", 0) > 0
        ]

        rows.append({
            "iteration": iteration,
            "best_score": metrics.get("combined_score", 0),
            "best_train_ic": metrics.get("train_rank_ic_mean", 0),
            "best_val_ic": metrics.get("val_rank_ic_mean", 0),
            "best_ir": metrics.get("train_rank_ic_ir", 0),
            "best_win_rate": metrics.get("train_ic_win_rate", 0),
            "best_turnover": metrics.get("factor_turnover", 0),
            "best_coverage": metrics.get("factor_coverage", 0),
            "auto_flipped": metrics.get("auto_flipped", 0),
            "pop_size": len(programs),
            "pop_avg_score": np.mean(all_scores) if all_scores else 0,
            "pop_max_score": np.max(all_scores) if all_scores else 0,
            "pop_std_score": np.std(all_scores) if all_scores else 0,
        })

    return pd.DataFrame(rows)


def build_evolution_tree(checkpoints: Dict) -> Dict:
    """构建演化树 - 追踪最优因子的祖先链"""
    if not checkpoints:
        return {}

    last_iter = max(checkpoints.keys())
    cp = checkpoints[last_iter]
    best_id = cp["metadata"].get("best_program_id", "")
    programs = cp["programs"]

    # 从最终最优开始，回溯祖先链
    lineage = []
    current_id = best_id

    while current_id and current_id in programs:
        prog = programs[current_id]
        lineage.append({
            "id": current_id[:8],
            "generation": prog.get("generation", 0),
            "iteration": prog.get("iteration_found", 0),
            "score": prog.get("metrics", {}).get("combined_score", 0),
            "train_ic": prog.get("metrics", {}).get("train_rank_ic_mean", 0),
            "val_ic": prog.get("metrics", {}).get("val_rank_ic_mean", 0),
            "ir": prog.get("metrics", {}).get("train_rank_ic_ir", 0),
            "island": prog.get("metadata", {}).get("island", 0),
            "changes": prog.get("metadata", {}).get("changes", ""),
            "code": prog.get("code", ""),
        })
        current_id = prog.get("parent_id", "")

    lineage.reverse()
    return lineage


def build_island_metrics(checkpoints: Dict) -> pd.DataFrame:
    """构建各岛屿指标对比"""
    rows = []
    for iteration in sorted(checkpoints.keys()):
        cp = checkpoints[iteration]
        metadata = cp.get("metadata", {})
        island_bests = metadata.get("island_best_programs", [])
        programs = cp["programs"]

        for i, best_id in enumerate(island_bests):
            if best_id and best_id in programs:
                prog = programs[best_id]
                rows.append({
                    "iteration": iteration,
                    "island": f"岛 {i}",
                    "score": prog.get("metrics", {}).get("combined_score", 0),
                    "train_ic": prog.get("metrics", {}).get("train_rank_ic_mean", 0),
                    "ir": prog.get("metrics", {}).get("train_rank_ic_ir", 0),
                })

    return pd.DataFrame(rows)


def build_map_elites_grid(checkpoints: Dict) -> Dict:
    """构建 MAP-Elites 网格数据"""
    result = {}
    for iteration in sorted(checkpoints.keys()):
        cp = checkpoints[iteration]
        metadata = cp.get("metadata", {})
        feature_maps = metadata.get("island_feature_maps", [])
        programs = cp["programs"]

        grid_data = []
        for island_idx, feature_map in enumerate(feature_maps):
            for coord_str, prog_id in feature_map.items():
                if prog_id and prog_id in programs:
                    prog = programs[prog_id]
                    coords = [int(c) for c in coord_str.split("-")]
                    grid_data.append({
                        "island": island_idx,
                        "coord": coord_str,
                        "bin_ic_mean": coords[0] if len(coords) > 0 else 0,
                        "bin_ir": coords[1] if len(coords) > 1 else 0,
                        "bin_stability": coords[2] if len(coords) > 2 else 0,
                        "bin_turnover": coords[3] if len(coords) > 3 else 0,
                        "score": prog.get("metrics", {}).get("combined_score", 0),
                        "train_ic": prog.get("metrics", {}).get("train_rank_ic_mean", 0),
                        "prog_id": prog_id[:8],
                    })

        result[iteration] = grid_data

    return result


def build_diversity_timeline(checkpoints: Dict) -> pd.DataFrame:
    """构建种群多样性时间线"""
    rows = []
    for iteration in sorted(checkpoints.keys()):
        cp = checkpoints[iteration]
        metadata = cp.get("metadata", {})
        islands = metadata.get("islands", [])

        for i, island_progs in enumerate(islands):
            prog_count = len(island_progs) if island_progs else 0
            rows.append({
                "iteration": iteration,
                "island": f"岛 {i}",
                "programs": prog_count,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 可视化图表
# ═══════════════════════════════════════════════════════════

def plot_metrics_overview(df: pd.DataFrame):
    """演化指标概览"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("综合评分", "训练集IC vs 验证集IC", "IC_IR (信息比率)", "种群统计"),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # 综合评分
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["best_score"], mode="lines+markers",
                   name="最优评分", line=dict(color="#636efa", width=3),
                   marker=dict(size=10)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["pop_avg_score"], mode="lines",
                   name="种群均值", line=dict(color="#636efa", width=1, dash="dash"),
                   fill="tonexty", fillcolor="rgba(99,110,250,0.1)"),
        row=1, col=1,
    )

    # IC 对比
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["best_train_ic"], mode="lines+markers",
                   name="训练集IC", line=dict(color="#00cc96", width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["best_val_ic"], mode="lines+markers",
                   name="验证集IC", line=dict(color="#ff7f0e", width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=0.08, line_dash="dash", line_color="gray",
                  annotation_text="IC目标 0.08", row=1, col=2)

    # IC_IR
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["best_ir"], mode="lines+markers",
                   name="IC_IR", line=dict(color="#ab63fa", width=3),
                   marker=dict(size=10)),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="优秀阈值 0.5", row=2, col=1)

    # 种群统计
    fig.add_trace(
        go.Bar(x=df["iteration"], y=df["pop_size"], name="种群数量",
               marker=dict(color="#19d3f3")),
        row=2, col=2,
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode="x unified",
        title=dict(text="演化指标概览", x=0.5, font=dict(size=20)),
    )
    fig.update_xaxes(title_text="迭代轮次", row=2, col=1)
    fig.update_xaxes(title_text="迭代轮次", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=2)
    fig.update_yaxes(title_text="IR", row=2, col=1)
    fig.update_yaxes(title_text="程序数", row=2, col=2)

    return fig


def plot_evolution_lineage(lineage: List[Dict]):
    """演化谱系桑基图/瀑布图"""
    if not lineage:
        return go.Figure()

    gens = [f"Gen {l['generation']}" for l in lineage]
    scores = [l["score"] for l in lineage]
    train_ics = [l["train_ic"] for l in lineage]
    val_ics = [l["val_ic"] for l in lineage]
    iterations = [l["iteration"] for l in lineage]
    changes = [l.get("changes", "")[:60] for l in lineage]
    ids = [l["id"] for l in lineage]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("最优因子演化路径 (Score)", "IC 变化趋势"),
        vertical_spacing=0.15,
        row_heights=[0.55, 0.45],
    )

    # Score 瀑布图
    increasing_color = "#00cc96"
    decreasing_color = "#ef553b"

    fig.add_trace(
        go.Waterfall(
            x=gens,
            y=[scores[0]] + [scores[i] - scores[i-1] for i in range(1, len(scores))],
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
            increasing=dict(marker=dict(color=increasing_color)),
            decreasing=dict(marker=dict(color=decreasing_color)),
            connector=dict(line=dict(color="gray", dash="dot")),
            name="Score变化",
            hovertemplate="%{x}<br>Score: %{text}<br>Iter: %{customdata}<br>%{hovertext}<extra></extra>",
            customdata=iterations,
            hovertext=changes,
        ),
        row=1, col=1,
    )

    # IC 趋势
    fig.add_trace(
        go.Scatter(x=gens, y=train_ics, mode="lines+markers",
                   name="训练集IC", line=dict(color="#00cc96", width=2),
                   marker=dict(size=8), customdata=ids,
                   hovertemplate="%{x}<br>TrainIC: %{y:.4f}<br>ID: %{customdata}"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=gens, y=val_ics, mode="lines+markers",
                   name="验证集IC", line=dict(color="#ff7f0e", width=2),
                   marker=dict(size=8)),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        height=650,
        showlegend=True,
        title=dict(text="最优因子演化谱系", x=0.5, font=dict(size=20)),
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="IC", row=2, col=1)

    return fig


def plot_island_comparison(df: pd.DataFrame):
    """岛屿对比图"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("各岛屿最优Score", "各岛屿训练IC", "各岛屿IC_IR"),
    )

    island_colors = px.colors.qualitative.Set2[:8]

    for i, island in enumerate(sorted(df["island"].unique())):
        island_df = df[df["island"] == island]
        color = island_colors[i % len(island_colors)]

        fig.add_trace(
            go.Scatter(x=island_df["iteration"], y=island_df["score"],
                       mode="lines+markers", name=island,
                       line=dict(color=color, width=2), marker=dict(size=6),
                       legendgroup=island),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=island_df["iteration"], y=island_df["train_ic"],
                       mode="lines+markers", name=island,
                       line=dict(color=color, width=2), marker=dict(size=6),
                       legendgroup=island, showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=island_df["iteration"], y=island_df["ir"],
                       mode="lines+markers", name=island,
                       line=dict(color=color, width=2), marker=dict(size=6),
                       legendgroup=island, showlegend=False),
            row=1, col=3,
        )

    fig.update_layout(
        height=450,
        title=dict(text="岛屿演化对比 (8岛并行)", x=0.5, font=dict(size=20)),
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="TrainIC", row=1, col=2)
    fig.update_yaxes(title_text="IC_IR", row=1, col=3)
    fig.update_xaxes(title_text="迭代轮次", row=1, col=1)
    fig.update_xaxes(title_text="迭代轮次", row=1, col=2)
    fig.update_xaxes(title_text="迭代轮次", row=1, col=3)

    return fig


def plot_map_elites_2d(grid_data: Dict, selected_iteration: int):
    """MAP-Elites 网格可视化 (2D投影)"""
    if selected_iteration not in grid_data:
        return go.Figure()

    data = grid_data[selected_iteration]
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    fig = go.Figure()

    island_colors = px.colors.qualitative.Set2[:8]

    for island in sorted(df["island"].unique()):
        island_df = df[df["island"] == island]
        color = island_colors[island % len(island_colors)]

        fig.add_trace(go.Scatter(
            x=island_df["bin_ic_mean"],
            y=island_df["bin_ir"],
            mode="markers+text",
            name=f"岛 {island}",
            text=island_df["prog_id"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=island_df["score"] * 50 + 8,
                color=color,
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            customdata=island_df[["score", "train_ic", "prog_id"]].values,
            hovertemplate=(
                "岛 %{name}<br>"
                "Score: %{customdata[0]:.4f}<br>"
                "TrainIC: %{customdata[1]:+.4f}<br>"
                "ID: %{customdata[2]}"
            ),
        ))

    fig.update_layout(
        height=500,
        title=dict(
            text=f"MAP-Elites 特征空间分布 (Checkpoint {selected_iteration})",
            x=0.5, font=dict(size=18),
        ),
        xaxis=dict(title="IC均值分箱 (abs_ic_mean bin)", dtick=1),
        yaxis=dict(title="IC_IR分箱 (ic_ir bin)", dtick=1),
        showlegend=True,
    )

    return fig


def plot_diversity_heatmap(df: pd.DataFrame):
    """岛屿种群分布 - 堆叠柱状图"""

    island_colors = px.colors.qualitative.Set2[:8]
    iterations = sorted(df["iteration"].unique())
    islands = sorted(df["island"].unique())

    fig = go.Figure()
    for i, island in enumerate(islands):
        island_df = df[df["island"] == island].set_index("iteration")
        vals = [int(island_df.loc[it, "programs"]) if it in island_df.index else 0 for it in iterations]
        fig.add_trace(go.Bar(
            x=iterations,
            y=vals,
            name=str(island),
            marker=dict(color=island_colors[i % len(island_colors)]),
            hovertemplate="%{x}: %{y}个程序<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        barmode="stack",
        height=380,
        title=dict(text="各岛屿种群数量分布", x=0.5, font=dict(size=18)),
        xaxis=dict(title="迭代轮次", dtick=25),
        yaxis=dict(title="程序总数"),
        legend=dict(title="岛屿"),
        hovermode="x unified",
    )

    return fig


def plot_kb_writeback_timeline(writebacks: List[Dict]):
    """知识库写回时间线"""
    if not writebacks:
        return go.Figure()

    wb_data = []
    for wb in writebacks:
        try:
            child_m = json.loads(wb["child_metrics"]) if isinstance(wb["child_metrics"], str) else wb["child_metrics"]
            parent_m = json.loads(wb["parent_metrics"]) if isinstance(wb["parent_metrics"], str) else wb["parent_metrics"]
        except Exception:
            continue

        wb_data.append({
            "id": str(wb.get("id", "")),
            "improvement": wb.get("improvement_ratio", 0),
            "parent_score": parent_m.get("combined_score", 0),
            "child_score": child_m.get("combined_score", 0),
            "parent_ic": parent_m.get("train_rank_ic_mean", 0),
            "child_ic": child_m.get("train_rank_ic_mean", 0),
            "status": wb.get("status", "unknown"),
            "created": wb.get("created_at", ""),
        })

    if not wb_data:
        return go.Figure()

    df = pd.DataFrame(wb_data)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("写回 Score 提升", "写回 IC 提升"),
    )

    # Score 对比
    for i, row in df.iterrows():
        color = "#00cc96" if row["status"] == "stored" else "#ffa15a"
        fig.add_trace(
            go.Scatter(
                x=[f"#{i+1}"], y=[row["child_score"] - row["parent_score"]],
                mode="markers",
                marker=dict(size=max(row["improvement"] * 100, 10), color=color),
                name=f"写回 {row['id']}",
                text=f"Score: {row['parent_score']:.3f}→{row['child_score']:.3f}",
                hovertemplate="%{text}<br>提升: %{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # IC 对比
    for i, row in df.iterrows():
        color = "#00cc96" if row["status"] == "stored" else "#ffa15a"
        fig.add_trace(
            go.Scatter(
                x=[f"#{i+1}"], y=[row["child_ic"] - row["parent_ic"]],
                mode="markers",
                marker=dict(size=max(abs(row["child_ic"] - row["parent_ic"]) * 200, 8), color=color),
                name=f"写回 {row['id']}",
                text=f"IC: {row['parent_ic']:+.4f}→{row['child_ic']:+.4f}",
                hovertemplate="%{text}<br>提升: %{y:+.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        height=350,
        title=dict(text="知识库写回事件", x=0.5, font=dict(size=18)),
    )
    fig.update_yaxes(title_text="Score Δ", row=1, col=1)
    fig.update_yaxes(title_text="IC Δ", row=1, col=2)

    return fig


def show_code_diff_viewer(lineage: List[Dict]):
    """代码差异对比器"""
    if len(lineage) < 2:
        st.info("需要至少2代才能对比代码")
        return

    st.subheader("因子代码演化对比")

    # 选择要对比的代数
    gen_labels = [f"Gen {l['generation']} (Iter {l['iteration']}, Score={l['score']:.4f})" for l in lineage]

    col1, col2 = st.columns(2)
    with col1:
        left_idx = st.selectbox("父代", range(len(lineage) - 1), format_func=lambda i: gen_labels[i])
    with col2:
        right_idx = st.selectbox("子代", range(left_idx + 1, len(lineage)),
                                  format_func=lambda i: gen_labels[i], index=0)

    left_code = lineage[left_idx].get("code", "")
    right_code = lineage[right_idx].get("code", "")

    # 提取 EVOLVE-BLOCK 内的代码
    def extract_block(code):
        if "EVOLVE-BLOCK-START" in code and "EVOLVE-BLOCK-END" in code:
            start = code.find("EVOLVE-BLOCK-START")
            end = code.find("EVOLVE-BLOCK-END")
            return code[start:end + len("EVOLVE-BLOCK-END")]
        return code

    left_block = extract_block(left_code)
    right_block = extract_block(right_code)

    left_lines = left_block.split("\n")
    right_lines = right_block.split("\n")

    # 简单 diff
    st.markdown(f"**Gen {lineage[left_idx]['generation']} → Gen {lineage[right_idx]['generation']}**")
    st.markdown(f"Score: {lineage[left_idx]['score']:.4f} → {lineage[right_idx]['score']:.4f} "
                f"(Δ {lineage[right_idx]['score'] - lineage[left_idx]['score']:+.4f})")

    # 显示变更说明
    changes = lineage[right_idx].get("changes", "")
    if changes and changes != "Full rewrite":
        st.info(f"变更摘要: {changes}")

    with st.expander("查看代码差异", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.code(left_block, language="python", line_numbers=True)
        with col_b:
            st.code(right_block, language="python", line_numbers=True)


# ═══════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="因子演化轨迹",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 因子演化轨迹可视化")
st.markdown("OpenEvolve 演化过程分析仪表盘")

# ── 侧边栏: 选择数据源 ──
with st.sidebar:
    st.header("📂 数据源")

    # 自动发现可用的输出目录
    base_dir = Path(__file__).parent
    output_dirs = sorted(
        [str(d) for d in base_dir.glob("output*") if d.is_dir() and (d / "checkpoints").exists()],
        reverse=True,
    )

    if not output_dirs:
        st.error("未找到包含 checkpoints/ 的输出目录")
        st.stop()

    selected_dir = st.selectbox(
        "选择演化输出目录",
        output_dirs,
        format_func=lambda x: f"{Path(x).name} ({len(list((Path(x)/'checkpoints').iterdir()))} checkpoints)",
    )

    st.divider()
    st.header("⚙️ 显示选项")
    show_lineage = st.checkbox("演化谱系", value=True)
    show_islands = st.checkbox("岛屿对比", value=True)
    show_map_elites = st.checkbox("MAP-Elites 分布", value=True)
    show_diversity = st.checkbox("岛屿多样性", value=True)
    show_kb = st.checkbox("知识库轨迹", value=True)
    show_code_diff = st.checkbox("代码对比", value=True)

# ── 加载数据 ──
with st.spinner("加载演化数据..."):
    checkpoints = load_checkpoint_data(selected_dir)
    kb_data = load_knowledge_base_data(selected_dir)

if not checkpoints:
    st.error(f"目录 {selected_dir} 中没有有效的 checkpoint 数据")
    st.stop()

st.sidebar.success(f"已加载 {len(checkpoints)} 个 checkpoints")

# ── 关键指标卡片 ──
df_metrics = build_metrics_timeline(checkpoints)
if not df_metrics.empty:
    last = df_metrics.iloc[-1]
    first = df_metrics.iloc[0]

    cols = st.columns(6)
    cols[0].metric("最终 Score", f"{last['best_score']:.4f}", f"{last['best_score'] - first['best_score']:+.4f}")
    cols[1].metric("训练集 IC", f"{last['best_train_ic']:+.4f}", f"{last['best_train_ic'] - first['best_train_ic']:+.4f}")
    cols[2].metric("验证集 IC", f"{last['best_val_ic']:+.4f}", f"{last['best_val_ic'] - first['best_val_ic']:+.4f}")
    cols[3].metric("IC_IR", f"{last['best_ir']:.3f}", f"{last['best_ir'] - first['best_ir']:+.3f}")
    cols[4].metric("种群规模", int(last['pop_size']))
    cols[5].metric("Checkpoints", len(checkpoints))

# ── Tab 1: 演化概览 ──
st.header("📈 演化指标概览")
st.plotly_chart(plot_metrics_overview(df_metrics), use_container_width=True)

# ── Tab 2: 演化谱系 ──
if show_lineage:
    st.header("🌳 最优因子演化谱系")
    lineage = build_evolution_tree(checkpoints)

    if lineage:
        col_lineage, col_info = st.columns([3, 1])
        with col_lineage:
            st.plotly_chart(plot_evolution_lineage(lineage), use_container_width=True)
        with col_info:
            st.subheader("演化路径")
            for i, node in enumerate(lineage):
                arrow = "└─" if i == len(lineage) - 1 else "├─"
                st.markdown(
                    f"`{arrow}` **Gen {node['generation']}** "
                    f"(Iter {node['iteration']})<br>"
                    f"&nbsp;&nbsp;&nbsp;Score: {node['score']:.4f} | "
                    f"IC: {node['train_ic']:+.4f} | "
                    f"IR: {node['ir']:.2f}"
                )

        if show_code_diff:
            show_code_diff_viewer(lineage)
    else:
        st.info("无法构建演化谱系（缺少 parent_id 数据）")

# ── Tab 3: 岛屿对比 ──
if show_islands:
    st.header("🏝️ 岛屿并行演化")
    df_islands = build_island_metrics(checkpoints)
    if not df_islands.empty:
        st.plotly_chart(plot_island_comparison(df_islands), use_container_width=True)
    else:
        st.info("无岛屿数据")

# ── Tab 4: MAP-Elites ──
if show_map_elites:
    st.header("🗺️ MAP-Elites 特征空间")

    grid_data = build_map_elites_grid(checkpoints)
    if grid_data:
        available_iters = sorted(grid_data.keys())
        selected_iter = st.select_slider(
            "选择 Checkpoint", options=available_iters,
            value=available_iters[-1],
        )
        st.plotly_chart(plot_map_elites_2d(grid_data, selected_iter), use_container_width=True)
    else:
        st.info("无 MAP-Elites 数据")

# ── Tab 5: 岛屿多样性 ──
if show_diversity:
    st.header("🔬 岛屿种群分布")
    df_diversity = build_diversity_timeline(checkpoints)
    if not df_diversity.empty:
        st.plotly_chart(plot_diversity_heatmap(df_diversity), use_container_width=True)

        # 汇总统计
        last_iter = df_diversity["iteration"].max()
        last_div = df_diversity[df_diversity["iteration"] == last_iter]
        total_progs = int(last_div["programs"].sum())
        avg_progs = int(last_div["programs"].mean())
        max_progs = int(last_div["programs"].max())
        min_progs = int(last_div["programs"].min())

        cols = st.columns(4)
        cols[0].metric("总程序数", total_progs)
        cols[1].metric("岛均程序数", avg_progs)
        cols[2].metric("最多岛屿", f"{max_progs} 程序")
        cols[3].metric("最少岛屿", f"{min_progs} 程序")
    else:
        st.info("无岛屿多样性数据")

# ── Tab 6: 知识库 ──
if show_kb:
    st.header("📚 知识库轨迹")

    if kb_data["writebacks"]:
        st.plotly_chart(
            plot_kb_writeback_timeline(kb_data["writebacks"]), use_container_width=True
        )

        st.subheader("写回记录")
        for wb in kb_data["writebacks"][:10]:
            try:
                child_m = json.loads(wb["child_metrics"]) if isinstance(wb["child_metrics"], str) else wb["child_metrics"]
                parent_m = json.loads(wb["parent_metrics"]) if isinstance(wb["parent_metrics"], str) else wb["parent_metrics"]
                problem_codes = json.loads(wb["problem_codes"]) if isinstance(wb["problem_codes"], str) else wb.get("problem_codes", [])
            except Exception:
                problem_codes = []
                child_m = {}
                parent_m = {}

            improvement = wb.get("improvement_ratio", 0)
            status_icon = "✅" if wb.get("status") == "stored" else "⏳"
            knowledge = wb.get("summarized_knowledge", "")
            created_at = wb.get("created_at", "")

            with st.expander(
                f"{status_icon} #{str(wb.get('id', ''))} | "
                f"Score: {parent_m.get('combined_score', 0):.3f} → "
                f"{child_m.get('combined_score', 0):.3f} (+{improvement:.0%})"
            ):
                if problem_codes:
                    st.markdown(f"**问题码:** {' '.join('`'+c+'`' for c in problem_codes)}")
                st.markdown(f"**创建时间:** {created_at}")

                ic_delta = child_m.get("train_rank_ic_mean", 0) - parent_m.get("train_rank_ic_mean", 0)
                ir_delta = child_m.get("train_rank_ic_ir", 0) - parent_m.get("train_rank_ic_ir", 0)
                st.markdown(f"**IC 变化:** {parent_m.get('train_rank_ic_mean', 0):+.4f} → "
                           f"{child_m.get('train_rank_ic_mean', 0):+.4f} (Δ {ic_delta:+.4f})")
                st.markdown(f"**IR 变化:** {parent_m.get('train_rank_ic_ir', 0):.3f} → "
                           f"{child_m.get('train_rank_ic_ir', 0):.3f} (Δ {ir_delta:+.3f})")

                if knowledge:
                    st.markdown("**知识总结:**")
                    st.markdown(knowledge[:500] + ("..." if len(knowledge) > 500 else ""))
    elif kb_data["entries"]:
        st.info(f"知识库有 {len(kb_data['entries'])} 条知识，但无写回记录（KB 可能在演化时未启用）")

        st.subheader("种子知识概览")
        for entry in kb_data["entries"][:5]:
            with st.expander(
                f"📖 {entry.get('factor_category', 'unknown')}: "
                f"{entry.get('context_before', '')[:80]}..."
            ):
                st.markdown(f"**改进方法:** {entry.get('improvement_action', '')[:200]}")
                st.markdown(f"**改进效果:** {entry.get('improvement_result', '')[:200]}")
                st.caption(f"使用次数: {entry.get('usage_count', 0)}")
    else:
        st.info("无知识库数据")

# ── 页脚 ──
st.divider()
st.caption(f"数据源: {selected_dir} | 共 {len(checkpoints)} 个 checkpoints")

if not df_metrics.empty:
    st.caption(
        f"起始迭代: {int(df_metrics['iteration'].min())} → "
        f"最终迭代: {int(df_metrics['iteration'].max())} | "
        f"最优 Score: {df_metrics['best_score'].max():.4f} "
        f"(Iter {int(df_metrics.loc[df_metrics['best_score'].idxmax(), 'iteration'])})"
    )
