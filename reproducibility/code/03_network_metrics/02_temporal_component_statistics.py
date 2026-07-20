# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_cumulative_stats_advanced_v2.py

==============================================================================
【程序功能：累积网络演化深度统计 (支持双对数分布与LCC分析)】
==============================================================================
修改说明：
本次修改针对“组件大小分布”的统计逻辑进行了调整。
- 原有逻辑：仅统计 S1 网络的所有组件。
- 现有逻辑：统计 S1, S5-S1, S5-S3-S1 三个网络，且**剔除最大连通组件 (LCC)** 后，
           统计剩余所有组件的大小及对应数量。

输出文件：
1. `cumulative_evolution_stats_v2.csv`:
   包含逐年、分网络的总节点数、LCC大小、组件总数等时间序列数据。

2. `component_size_distribution_no_lcc_2025.csv`:
   包含2025年三个网络(Overall, Active, Backbone)在**去除LCC后**的详细组件大小分布。
   列名：[pipeline, component_size, frequency]

==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

# 尝试导入 SciPy 以加速矩阵运算
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# 1. 全局配置区 (Configuration)
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "temporal_component_statistics")

AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

METHOD = "newman"
S3_ALPHA = 0.1
MIN_PAPER_COUNT = 2
TIME_START = 2006
TIME_END = 2025

PIPELINES = {
    "S1": ["S1"],  # Overall
    "S5-S1": ["S5", "S1"],  # Active
    "S5-S3-S1": ["S5", "S3", "S1"]  # Backbone
}


# =========================
# 2. 工具函数 (Utils)
# =========================

def ensure_outdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                return df_col
    return ""


def load_data(csv_path: str):
    print(f"Reading raw data from: {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)
    year_col = detect_column(df, YEAR_COL_CANDIDATES)
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)
    return df, year_col


def build_index_maps(df_slice):
    author_ids = df_slice[AUTHOR_COL].unique()
    paper_ids = df_slice[PAPER_COL].unique()
    aid2idx = {a: i for i, a in enumerate(author_ids)}
    pid2idx = {p: i for i, p in enumerate(paper_ids)}
    df_idx = df_slice.copy()
    df_idx["ai"] = df_idx[AUTHOR_COL].map(aid2idx).astype(int)
    df_idx["pk"] = df_idx[PAPER_COL].map(pid2idx).astype(int)
    return author_ids, paper_ids, df_idx


# =========================
# 3. 核心构图与过滤算法 (Algorithms)
# =========================

def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()
    return U


def edges_from_sparse(S, author_ids):
    S = S.tocoo()
    mask = S.row < S.col
    rows, cols, data = S.row[mask], S.col[mask], S.data[mask]
    return pd.DataFrame({
        "src_author_id": [author_ids[i] for i in rows],
        "dst_author_id": [author_ids[j] for j in cols],
        "weight": data.astype(float)
    })


def build_network_matrix(method, author_ids, paper_ids, df_idx):
    if not HAVE_SCIPY:
        raise RuntimeError("Need SciPy library.")
    A = sp.csr_matrix(
        (np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
        shape=(len(author_ids), len(paper_ids))
    )
    if method == "newman":
        n = np.asarray(A.sum(axis=0)).ravel()
        w = np.zeros_like(n, dtype=float)
        mask = n > 1
        w[mask] = 1.0 / (n[mask] - 1.0)
        U = A @ sp.diags(w) @ A.T
    else:
        U = A @ A.T
    U = zero_diag_csr_inplace(U.tocsr())
    return edges_from_sparse(U, author_ids)


# --- Filters ---

def filter_isolates_s1(G):
    if G.number_of_nodes() == 0: return G
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    return G


def filter_disparity_s3(G, alpha_threshold=0.1):
    if G.number_of_edges() == 0: return G
    strength = {n: 0.0 for n in G.nodes()}
    degree = {n: 0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        strength[u] += w;
        strength[v] += w;
        degree[u] += 1;
        degree[v] += 1
    rem_edges = []
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        k_u = degree[u];
        a_u = (max(0.0, 1.0 - w / strength[u]) ** (k_u - 1)) if k_u > 1 else 0.0
        k_v = degree[v];
        a_v = (max(0.0, 1.0 - w / strength[v]) ** (k_v - 1)) if k_v > 1 else 0.0
        if not (a_u < alpha_threshold or a_v < alpha_threshold):
            rem_edges.append((u, v))
    G.remove_edges_from(rem_edges)
    return filter_isolates_s1(G)


def filter_min_papers_s5(G, paper_counts, min_count):
    nodes_to_remove = [n for n in G.nodes() if paper_counts.get(n, 0) < min_count]
    if nodes_to_remove: G.remove_nodes_from(nodes_to_remove)
    return G


def run_filter_pipeline(G, pipeline_list, alpha_val, paper_counts, min_paper_count):
    G_curr = G.copy()
    for step in pipeline_list:
        if step == "S1":
            G_curr = filter_isolates_s1(G_curr)
        elif step == "S3":
            G_curr = filter_disparity_s3(G_curr, alpha_val)
        elif step == "S5":
            G_curr = filter_min_papers_s5(G_curr, paper_counts, min_paper_count)
    return G_curr


# =========================
# 4. 核心统计逻辑 (Statistics)
# =========================

def analyze_components_detailed(G):
    """
    详细分析连通组件分布
    返回: lcc_size, num_components, size_counter
    """
    if G.number_of_nodes() == 0:
        return 0, 0, Counter()

    # 获取所有连通组件
    components = list(nx.connected_components(G))
    # 计算每个组件的大小
    sizes = [len(c) for c in components]

    if not sizes:
        return 0, 0, Counter()

    lcc_size = max(sizes)
    num_components = len(sizes)
    size_counter = Counter(sizes)

    return lcc_size, num_components, size_counter


def main():
    print(">>> 启动累积网络深度统计 (Analysis for Plots)...")
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载全量数据
    df_all, year_col = load_data(PATH_INPUT)
    if not year_col:
        print("Error: No year column found.")
        return

    # 用于存储主图的时间序列数据
    time_series_records = []
    # 用于存储2025年详细分布数据 (No LCC)
    dist_2025_records = []

    years = range(TIME_START, TIME_END + 1)

    for year in years:
        print(f"\nProcessing Cumulative Year: {year} ...")

        df_cum = df_all[df_all[year_col] <= year]
        if df_cum.empty: continue

        curr_paper_counts = df_cum[AUTHOR_COL].value_counts().to_dict()

        # 构建基础网络
        auth_ids, paper_ids, df_idx = build_index_maps(df_cum)
        edges = build_network_matrix(METHOD, auth_ids, paper_ids, df_idx)
        G_cum_base = nx.Graph()
        for r in edges.itertuples(index=False):
            G_cum_base.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

        # 遍历三个网络管道
        for pipe_name, pipeline in PIPELINES.items():

            # 执行过滤
            G_curr = run_filter_pipeline(G_cum_base, pipeline, S3_ALPHA, curr_paper_counts, MIN_PAPER_COUNT)

            # 统计组件信息
            lcc_size, num_comps, size_counter = analyze_components_detailed(G_curr)
            total_nodes = G_curr.number_of_nodes()

            # 记录时间序列数据
            rec = {
                "year": year,
                "pipeline": pipe_name,
                "total_nodes": total_nodes,
                "total_edges": G_curr.number_of_edges(),
                "lcc_size": lcc_size,
                "num_components": num_comps
            }
            time_series_records.append(rec)

            # [修改部分] 针对 2025 年，统计所有三个网络去除 LCC 后的组件分布
            if year == 2025:
                print(f"  --> Processing component distribution (No LCC) for 2025 [{pipe_name}]...")

                # 创建副本以免修改原始统计
                dist_no_lcc = size_counter.copy()

                # 核心逻辑：从计数器中移除 1 个 LCC 实例
                # 只有当 LCC 存在时才移除 (防止空图报错)
                if lcc_size > 0 and dist_no_lcc[lcc_size] > 0:
                    dist_no_lcc[lcc_size] -= 1
                    # 如果减去后数量为0，删除该 Key
                    if dist_no_lcc[lcc_size] == 0:
                        del dist_no_lcc[lcc_size]

                # 将剩余的组件分布加入列表
                # 格式: pipeline, size, count
                for s, count in dist_no_lcc.items():
                    dist_2025_records.append({
                        "pipeline": pipe_name,
                        "component_size": s,
                        "frequency": count
                    })

    # --- 3. 保存文件 ---

    # 3.1 保存时间序列汇总数据
    df_ts = pd.DataFrame(time_series_records)
    out_ts_path = os.path.join(PATH_OUTDIR, "cumulative_evolution_stats_v2.csv")
    df_ts.to_csv(out_ts_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> 统计完成 1/2。")
    print(f"    时间序列数据已保存至: {out_ts_path}")

    # 3.2 保存 2025 年无 LCC 的组件分布数据
    if dist_2025_records:
        df_dist = pd.DataFrame(dist_2025_records)
        # 按 pipeline 和 component_size 排序，方便查看
        df_dist.sort_values(by=["pipeline", "component_size"], inplace=True)

        dist_out_path = os.path.join(PATH_OUTDIR, "component_size_distribution_no_lcc_2025.csv")
        df_dist.to_csv(dist_out_path, index=False, encoding="utf-8-sig")
        print(f"    2025年组件分布(No LCC)已保存至: {dist_out_path}")
    else:
        print("    Warning: No distribution data collected for 2025.")


if __name__ == "__main__":
    main()
