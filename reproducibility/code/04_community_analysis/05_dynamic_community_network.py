# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
Master Thesis Dynamic Network V7 (Avg Citations Per Paper)
硕士毕业论文 - 动态多重图生成脚本 (V7 篇均引用版)

修正点：
1. 社区引用算法升级：先在社区内对 paper_id 去重，再计算 (总被引 / 总篇数)，即“篇均被引”。
2. 保留所有 V6 功能：S5/S2过滤、双模式阈值、多重边动态图。
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import infomap
import scipy.sparse as sp

# =========================
# 1. 全局配置 (Configuration)
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "04_community_analysis" / "dynamic_community_network")

START_YEAR = 2006
END_YEAR = 2025

# --- 阈值模式设置 ---
# 'count' (人数) 或 'percent' (比例)
THRESHOLD_MODE = 'percent'
THRESHOLD_VAL = 0.1

# Infomap 参数
INFOMAP_ARGS = "--markov-time 1.1 -N 1000 --seed 42"


# =========================
# 2. 基础工具函数
# =========================
def ensure_outdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path):
    print(f"[Data] Reading: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)

    year_col = next((c for c in df.columns if c in ["publication_year", "year", "pub_year", "PY"]), None)
    if not year_col:
        raise ValueError("Year column not found!")

    df = df.dropna(subset=["author_id", "paper_id"])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)

    # 引用次数列处理
    if "times_cited" in df.columns:
        df["times_cited"] = pd.to_numeric(df["times_cited"], errors='coerce').fillna(0).astype(int)
    else:
        print("Warning: 'times_cited' column not found. Setting to 0.")
        df["times_cited"] = 0

    # 仅保留研究区间
    df = df[(df[year_col] >= START_YEAR) & (df[year_col] <= END_YEAR)]

    # 去重
    df = df.drop_duplicates(subset=["author_id", "paper_id"])

    print(f"[Data] Loaded {len(df)} rows after drop_duplicates.")
    return df, year_col


def build_network_fast(df_slice):
    a_ids = df_slice["author_id"].unique()
    p_ids = df_slice["paper_id"].unique()
    if len(a_ids) == 0 or len(p_ids) == 0:
        return nx.Graph()

    aid2idx = {a: i for i, a in enumerate(a_ids)}
    pid2idx = {p: i for i, p in enumerate(p_ids)}

    row_ind = df_slice["author_id"].map(aid2idx).astype(int)
    col_ind = df_slice["paper_id"].map(pid2idx).astype(int)

    data = np.ones(len(df_slice), dtype=np.float32)
    A = sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(a_ids), len(p_ids)))

    n = np.asarray(A.sum(axis=0)).ravel()
    w = np.zeros_like(n, dtype=np.float32)
    mask = n > 1
    w[mask] = 1.0 / (n[mask] - 1.0)

    U = A @ sp.diags(w) @ A.T
    U.setdiag(0)
    U.eliminate_zeros()

    G = nx.from_scipy_sparse_array(U)
    mapping = {i: aid for i, aid in enumerate(a_ids)}
    G = nx.relabel_nodes(G, mapping)
    return G


def filter_like_single_year(G, paper_counts):
    rem = [n for n in G.nodes() if paper_counts.get(n, 0) < 2]
    if rem:
        G.remove_nodes_from(rem)

    if G.number_of_nodes() > 0 and nx.number_connected_components(G) > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


# =========================
# 3. 核心逻辑
# =========================
def get_infomap_level2_partition(df):
    print("[Phase 1] Detecting Communities (Infomap Level 2)...")
    df_full = df
    G_full = build_network_fast(df_full)
    paper_counts = df_full["author_id"].value_counts().to_dict()
    G_full = filter_like_single_year(G_full, paper_counts)

    print(f"  Network size after S5+S2: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges.")

    if G_full.number_of_nodes() == 0:
        return G_full, {}

    im = infomap.Infomap(INFOMAP_ARGS)
    nodes = list(G_full.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    for u, v, d in G_full.edges(data=True):
        im.add_link(node_to_int[u], node_to_int[v], float(d.get("weight", 1.0)))
    im.run()

    if hasattr(im, "get_multilevel_modules"):
        modules = im.get_multilevel_modules(states=False)
    else:
        modules = dict(im.multilevel_modules)

    partition = {}
    path_map = {}
    next_id = 1

    for node_int, path in modules.items():
        if path is None: continue
        if len(path) >= 2:
            level2_key = tuple(path[:2])
        elif len(path) == 1:
            level2_key = (path[0], 0)
        else:
            continue

        if level2_key not in path_map:
            path_map[level2_key] = next_id
            next_id += 1

        partition[int_to_node[int(node_int)]] = path_map[level2_key]

    print(f"  Detected {len(path_map)} Level-2 communities.")
    return G_full, partition


def calculate_start_year_flexible(yearly_sizes, final_size):
    if final_size == 0: return None
    if THRESHOLD_MODE == 'percent':
        target = max(1, final_size * THRESHOLD_VAL)
    elif THRESHOLD_MODE == 'count':
        target = THRESHOLD_VAL
    else:
        target = 1

    for year in sorted(yearly_sizes.keys()):
        if yearly_sizes[year] >= target:
            return float(year)
    return None


def generate_multigraph_gexf(df, G_full, partition, year_col, out_dir):
    print(f"[Phase 2] Generating GEXF (Mode: {THRESHOLD_MODE}, Val: {THRESHOLD_VAL})...")

    M = nx.MultiGraph()
    comm_ids = sorted(list(set(partition.values())))

    # -------------------------------------------------------------------------
    # 1. 计算【篇均文章被引次数】 (Avg_Paper_Citations)
    # -------------------------------------------------------------------------
    print("  Calculating Avg Paper Citations per Community...")

    # 建立映射: Author -> Community
    df_metrics = df[['author_id', 'paper_id', 'times_cited']].copy()
    df_metrics['comm_id'] = df_metrics['author_id'].map(partition)

    # 过滤掉不在社区划分中的作者（被S5/S2过滤掉的）
    df_metrics = df_metrics.dropna(subset=['comm_id'])

    comm_avg_citations = {}

    # 按社区分组计算
    for cid, group in df_metrics.groupby('comm_id'):
        # 【关键修正】：在社区内部对 paper_id 去重
        # 这样多人合著的论文只算一次引用，不会重复叠加
        unique_papers = group.drop_duplicates(subset=['paper_id'])

        total_cited = unique_papers['times_cited'].sum()
        total_papers = len(unique_papers)

        avg_cit = total_cited / total_papers if total_papers > 0 else 0.0
        comm_avg_citations[int(cid)] = avg_cit

    # -------------------------------------------------------------------------
    # 2. 计算平均度 (Avg Degree)
    # -------------------------------------------------------------------------
    print("  Calculating Avg Degree...")
    comm_members = {}
    for auth, cid in partition.items():
        comm_members.setdefault(cid, []).append(auth)

    # -------------------------------------------------------------------------
    # 3. 初始化节点并写入静态指标
    # -------------------------------------------------------------------------
    for cid in comm_ids:
        members = comm_members.get(cid, [])
        if not members:
            M.add_node(cid, label=f"Community {cid}", Avg_Degree_Final=0.0, Avg_Paper_Citations=0.0, Final_Size=0)
            continue

        # 计算平均度
        degrees = [G_full.degree(m, weight='weight') for m in members]
        avg_deg = float(np.mean(degrees)) if degrees else 0.0

        # 获取刚才计算好的篇均引用
        avg_cit = comm_avg_citations.get(cid, 0.0)

        # 写入节点属性
        M.add_node(cid,
                   label=f"Community {cid}",
                   Avg_Degree_Final=avg_deg,
                   Avg_Paper_Citations=avg_cit,  # 新增属性
                   Final_Size=int(len(members)))

    # -------------------------------------------------------------------------
    # 4. 动态演化计算
    # -------------------------------------------------------------------------
    df_lite = df[["author_id", "paper_id", year_col]].copy()
    df_lite["comm_id"] = df_lite["author_id"].map(partition)
    df_lite = df_lite.dropna(subset=["comm_id"])
    df_lite["comm_id"] = df_lite["comm_id"].astype(int)

    cumulative_authors = {cid: set() for cid in comm_ids}
    global_edge_counter = {}
    node_yearly_stats = {cid: {} for cid in comm_ids}

    groups = df_lite.groupby(year_col)

    for year in range(START_YEAR, END_YEAR + 1):
        if year in groups.groups:
            df_year = groups.get_group(year)

            for comm, group in df_year.groupby("comm_id"):
                cumulative_authors[int(comm)].update(group["author_id"])

            G_yr = build_network_fast(df_year)

            for u, v in G_yr.edges():
                c1, c2 = partition.get(u), partition.get(v)
                if c1 is None or c2 is None: continue
                if c1 != c2:
                    if c1 > c2: c1, c2 = c2, c1
                    key = (c1, c2)
                    global_edge_counter[key] = global_edge_counter.get(key, 0) + 1

        # A. 写入动态 Size
        for cid in comm_ids:
            size = len(cumulative_authors[cid])
            node_yearly_stats[cid][year] = size
            M.nodes[cid][f"Size_{year}"] = int(size)

        # B. 写入动态 Edge
        for (c1, c2), weight in global_edge_counter.items():
            if weight > 0:
                edge_key = f"{c1}_{c2}_{year}"
                M.add_edge(
                    c1, c2,
                    key=edge_key,
                    weight=float(weight),
                    start=float(year),
                    end=float(year + 1.0),
                    Year=int(year)
                )

    # -------------------------------------------------------------------------
    # 5. 后处理
    # -------------------------------------------------------------------------
    print("  Calculating Layout & Applying Thresholds...")

    G_layout = nx.Graph()
    for u, v, data in M.edges(data=True):
        if data.get('Year') == END_YEAR and data.get('weight', 0) > 1:
            if G_layout.has_edge(u, v):
                G_layout[u][v]['weight'] += data['weight']
            else:
                G_layout.add_edge(u, v, weight=data['weight'])

    pos = nx.spring_layout(G_layout, k=2.5 / np.sqrt(len(comm_ids) + 1), seed=42)
    scale = 1000

    for cid in list(comm_ids):
        if cid in M.nodes and cid in pos:
            M.nodes[cid]['viz'] = {
                'position': {'x': float(pos[cid][0] * scale), 'y': float(pos[cid][1] * scale), 'z': 0.0},
                'color': {'r': 100, 'g': 180, 'b': 255, 'a': 1.0}
            }

        final_size = node_yearly_stats.get(cid, {}).get(END_YEAR, 0)
        start_year = calculate_start_year_flexible(node_yearly_stats.get(cid, {}), final_size)

        if start_year is not None and cid in M.nodes:
            M.nodes[cid]['start'] = float(start_year)
            M.nodes[cid]['end'] = float(END_YEAR + 1.0)
        else:
            if cid in M.nodes:
                M.remove_node(cid)

    out_file = os.path.join(out_dir, f"dynamic_net_v7_avgcit.gexf")
    print(f"  Writing GEXF to {out_file}...")
    nx.write_gexf(M, out_file, version="1.2draft")
    print("[Success] Done.")


# =========================
# Main
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)
    df, year_col = load_data(PATH_INPUT)
    G_full, partition = get_infomap_level2_partition(df)
    generate_multigraph_gexf(df, G_full, partition, year_col, PATH_OUTDIR)


if __name__ == "__main__":
    main()
