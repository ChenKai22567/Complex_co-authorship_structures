# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_node_stats_and_avg_degree.py

==============================================================================
【程序功能：网络节点度、出现年份及所属网络统计】
==============================================================================
此脚本基于之前确认的逻辑 (Global-First 策略)，统计下述信息：
1. 节点统计表 (Node Stats):
   - 节点ID (Author ID)
   - 最早出现年份 (First Year)
   - 在三个网络 (Overall, Active, Backbone) 中的度 (Degree)
   - 所属网络标记 (Membership): 标记节点属于哪个层级的网络

2. 网络年平均度 (Annual Average Degree):
   - 统计每个网络 (S2, S5-S2, S5-S3-S2) 在每一年 (2006-2025) 的平均度。

==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

# 尝试导入 SciPy 加速矩阵运算
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# 1. 全局配置区
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "temporal_degree_statistics")

# 列名配置
AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

# 核心参数
METHOD = "newman"
PIPELINES = {
    "Overall": ["S2"],
    "Active": ["S5", "S2"],
    "Backbone": ["S5", "S3", "S2"]
}
# 注意：通常 Backbone ⊆ Active ⊆ Overall，但我们独立计算各自的度。

S3_ALPHA = 0.1
MIN_PAPER_COUNT = 2
TIME_START = 2006
TIME_END = 2025


# =========================
# 工具函数 (复用)
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
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")
    print(f"Reading: {csv_path} ...")
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
# 构图与过滤 (复用 S3 逻辑)
# =========================

def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False);
    U.setdiag(0);
    U = U.tocsr(copy=False);
    U.eliminate_zeros();
    return U


def edges_from_sparse(S, author_ids):
    S = S.tocoo();
    mask = S.row < S.col;
    rows, cols, data = S.row[mask], S.col[mask], S.data[mask]
    return pd.DataFrame({"src_author_id": [author_ids[i] for i in rows], "dst_author_id": [author_ids[j] for j in cols],
                         "weight": data.astype(float)})


def build_network_matrix(method, author_ids, paper_ids, df_idx):
    if not HAVE_SCIPY: raise RuntimeError("Need SciPy.")
    A = sp.csr_matrix((np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
                      shape=(len(author_ids), len(paper_ids)))
    if method == "full":
        U = A @ A.T
    elif method == "newman":
        n = np.asarray(A.sum(axis=0)).ravel();
        w = np.zeros_like(n, dtype=float);
        mask = n > 1
        w[mask] = 1.0 / (n[mask] - 1.0)
        U = A @ sp.diags(w) @ A.T
    else:
        raise ValueError(method)
    U = zero_diag_csr_inplace(U.tocsr())
    return edges_from_sparse(U, author_ids)


def filter_isolates_s1(G):
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates);
    return G


def filter_lcc_s2(G):
    if G.number_of_nodes() == 0: return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def filter_disparity_s3(G, alpha_threshold=0.1):
    if G.number_of_edges() == 0: return G
    strength = {n: 0.0 for n in G.nodes()};
    degree = {n: 0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0);
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
        if not (a_u < alpha_threshold or a_v < alpha_threshold): rem_edges.append((u, v))
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
        elif step == "S2":
            G_curr = filter_lcc_s2(G_curr)
        elif step == "S3":
            G_curr = filter_disparity_s3(G_curr, alpha_val)
        elif step == "S5":
            G_curr = filter_min_papers_s5(G_curr, paper_counts, min_paper_count)
    return G_curr


# =========================
# 统计逻辑
# =========================

def get_node_first_year(df, year_col):
    """统计每个节点最早出现的年份"""
    print("    Calculating first year for all nodes...")
    # 按作者分组找最小年份
    first_years = df.groupby(AUTHOR_COL)[year_col].min().to_dict()
    return first_years


def main():
    print(">>> 启动节点统计与年平均度分析...")
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载数据
    df_all, year_col = load_data(PATH_INPUT)
    if not year_col: return

    # 2. 计算所有节点的 First Year
    node_first_years = get_node_first_year(df_all, year_col)

    # 3. 构建三个全局网络并统计度
    # 字典用于存储每个节点在不同网络中的度
    # 结构: { node_id: {'Overall_deg': 0, 'Active_deg': 0, 'Backbone_deg': 0} }
    node_stats = {}

    # 字典用于存储网络对象，以便后续计算年平均度
    valid_nodes_sets = {}  # { 'Overall': set(...), ... }

    print("\n--- 构建全局网络 (Global-First) ---")

    # 全局基础构图 (只构建一次 Graph 对象)
    print("    Building Base Global Weighted Network...")
    auth_ids_g, paper_ids_g, df_idx_g = build_index_maps(df_all)
    edges_g = build_network_matrix(METHOD, auth_ids_g, paper_ids_g, df_idx_g)

    G_base = nx.Graph()
    for r in edges_g.itertuples(index=False):
        G_base.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

    global_paper_counts = df_all[AUTHOR_COL].value_counts().to_dict()

    # 对三个网络分别过滤并记录信息
    for name, pipeline in PIPELINES.items():
        print(f"    Processing [{name}] pipeline: {pipeline}...")
        G_filtered = run_filter_pipeline(G_base, pipeline, S3_ALPHA, global_paper_counts, MIN_PAPER_COUNT)

        # 保存有效节点集合，用于后续时间切片过滤
        valid_nodes_sets[name] = set(G_filtered.nodes())

        # 记录每个节点的度
        deg_dict = dict(G_filtered.degree())
        for node in G_filtered.nodes():
            if node not in node_stats:
                node_stats[node] = {'Overall_deg': 0, 'Active_deg': 0, 'Backbone_deg': 0}

            # 记录度数
            key_deg = f"{name}_deg"
            node_stats[node][key_deg] = deg_dict[node]

    # 4. 生成节点统计表 (Node Stats CSV)
    print("\n--- 生成节点统计表 ---")
    rows = []
    for node, stats in node_stats.items():
        # 确定 Membership (所属网络标记)
        # 逻辑: Backbone 是 Active 的子集，Active 是 Overall 的子集
        # 如果 Backbone_deg > 0 -> 属于 Backbone
        # else if Active_deg > 0 -> 属于 Active
        # else -> 属于 Overall
        if stats['Backbone_deg'] > 0:
            membership = "Backbone"
        elif stats['Active_deg'] > 0:
            membership = "Active"
        else:
            membership = "Overall"

        row = {
            "author_id": node,
            "first_year": node_first_years.get(node, 0),
            "membership": membership,
            "degree_overall": stats['Overall_deg'],
            "degree_active": stats['Active_deg'],
            "degree_backbone": stats['Backbone_deg']
        }
        rows.append(row)

    df_node_stats = pd.DataFrame(rows)
    # 按 membership 和 degree 排序方便查看
    df_node_stats.sort_values(by=["degree_backbone", "degree_active", "degree_overall"], ascending=False, inplace=True)

    out_node_path = os.path.join(PATH_OUTDIR, "node_degree_and_year_stats.csv")
    df_node_stats.to_csv(out_node_path, index=False, encoding="utf-8-sig")
    print(f"    [Saved] Node Stats: {out_node_path} (Rows: {len(df_node_stats)})")

    # 5. 计算年平均度 (Annual Average Degree)
    print("\n--- 计算年平均度 (Time Slicing) ---")
    avg_deg_records = []

    # 按年遍历
    years = range(TIME_START, TIME_END + 1)

    for year in years:
        # 切片数据
        df_slice = df_all[df_all[year_col] == year]
        if df_slice.empty:
            continue

        # 构建当年的基础网络
        auth_ids_s, paper_ids_s, df_idx_s = build_index_maps(df_slice)
        edges_s = build_network_matrix(METHOD, auth_ids_s, paper_ids_s, df_idx_s)
        G_slice_base = nx.Graph()
        # 注意：这里不需要权重，计算度只需要拓扑结构，或者如果newman加权度也行，通常平均度指无权度
        # 这里为了计算平均度 (Average Degree = 2E/N)，使用无权边即可
        for r in edges_s.itertuples(index=False):
            G_slice_base.add_edge(r.src_author_id, r.dst_author_id)

        # 针对三个网络分别计算
        # 逻辑:
        # 1. 取出 Global 步骤中确定的该网络有效节点 (Global-First 策略)
        # 2. 从当年的 G_slice_base 中提取子图
        # 3. 移除孤立节点 (S1) - 因为不在当年发表论文或者切分后变孤立的不算在当年平均度分母中通常更合理，
        #    或者保留所有 Valid Nodes。通常 Global-First 后，我们统计的是在该切片中“活跃”的子图。
        #    这里采用：提取子图 -> 移除孤立点 -> 计算平均度

        record = {"year": year}

        for name in ["Overall", "Active", "Backbone"]:
            valid_nodes = valid_nodes_sets[name]

            # 提取子图
            # G_sub = G_slice_base.subgraph(valid_nodes).copy() # 这样包含所有valid nodes，即使当年度为0

            # 为了计算“平均度”，通常分母是当年的网络规模。
            # 如果仅仅 subgraph(valid_nodes)，分母可能是所有历史节点。
            # 建议逻辑：只保留当年有连边的节点 (S1)

            # 1. 仅保留在 Global Valid Set 中的节点
            nodes_in_slice = [n for n in G_slice_base.nodes() if n in valid_nodes]
            G_sub = G_slice_base.subgraph(nodes_in_slice).copy()

            # 2. 移除度为0的节点 (可选，但通常平均度统计的是连通部分或活跃部分)
            # 如果不移除，分母会很大（包含所有 Global Valid 但当年不活跃的人），平均度会极低。
            # 按照惯例，通常计算的是 "Active Subgraph" 的属性。
            G_sub = filter_isolates_s1(G_sub)

            n_nodes = G_sub.number_of_nodes()
            n_edges = G_sub.number_of_edges()

            if n_nodes > 0:
                avg_k = (2.0 * n_edges) / n_nodes
            else:
                avg_k = 0.0

            record[f"avg_degree_{name}"] = avg_k
            record[f"node_count_{name}"] = n_nodes  # 顺便记录当年的节点数

        avg_deg_records.append(record)
        # print(f"    Year {year} done.")

    df_avg_deg = pd.DataFrame(avg_deg_records)
    out_avg_path = os.path.join(PATH_OUTDIR, "network_annual_avg_degree.csv")
    df_avg_deg.to_csv(out_avg_path, index=False, encoding="utf-8-sig")
    print(f"    [Saved] Annual Avg Degree: {out_avg_path}")


if __name__ == "__main__":
    main()
