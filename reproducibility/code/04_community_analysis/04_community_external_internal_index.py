# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
One-Stop Script: Community E-I Index Analysis (Based on Edge Counts/Raw Degree)
------------------------------------------------------------------------------
功能：
    从原始数据出发，构建2025年网络，划分社区，并计算基于"合作人数"的 EI 指数。
    EI = (External_Neighbors - Internal_Neighbors) / (Total_Neighbors)

    - EI -> -1 : 极其封闭 (只认识圈内人)
    - EI -> +1 : 极其开放 (只认识圈外人)

输出：
    stats_2025_community_ei_raw.csv
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import infomap
import scipy.sparse as sp
import matplotlib.pyplot as plt

# =========================
# 1. 配置 (Configuration)
# =========================
# 输入文件的绝对路径
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")

# 输出目录
PATH_OUTDIR = str(_OUTPUT_ROOT / "04_community_analysis" / "community_external_internal_index")

# 目标年份
TARGET_YEAR = 2025

# Infomap 参数
INFOMAP_ARGS = "--markov-time 1.1 -N 1000 --seed 42"


# =========================
# 2. 核心函数
# =========================

def ensure_outdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path):
    print(f"[1/6] Reading data from: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)

    # 查找年份列
    year_col = next((c for c in df.columns if c in ["publication_year", "year", "pub_year", "PY"]), None)
    if not year_col: raise ValueError("Year column not found.")

    # 清洗
    df = df.dropna(subset=["author_id", "paper_id"])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)

    return df, year_col


def build_network_structure(df_slice):
    """
    构建网络结构。
    虽然是加权投影，但 NetworkX 图结构本身包含了所有连边信息，
    足够我们后续统计“邻居数量”(Raw Degree)。
    """
    print("[3/6] Building network topology...")
    a_ids = df_slice["author_id"].unique()
    p_ids = df_slice["paper_id"].unique()

    aid2idx = {a: i for i, a in enumerate(a_ids)}
    pid2idx = {p: i for i, p in enumerate(p_ids)}

    row_ind = df_slice["author_id"].map(aid2idx).astype(int)
    col_ind = df_slice["paper_id"].map(pid2idx).astype(int)

    # 二部图矩阵 A
    A = sp.csr_matrix(
        (np.ones(len(df_slice), dtype=np.float32), (row_ind, col_ind)),
        shape=(len(a_ids), len(p_ids))
    )

    # 简单投影 (A * A.T) 得到邻接关系
    # 这里为了速度，还是用标准的加权投影逻辑，反正后面只看连通性
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


def run_infomap(G):
    print(f"[4/6] Detecting communities (Infomap {INFOMAP_ARGS})...")
    if len(G) == 0: return {}

    im = infomap.Infomap(INFOMAP_ARGS)
    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    for u, v, d in G.edges(data=True):
        im.add_link(node_to_int[u], node_to_int[v], float(d.get("weight", 1.0)))

    im.run()

    partition = {}
    path_to_id = {}
    next_id = 1

    # 兼容不同版本API
    if hasattr(im, "get_multilevel_modules"):
        modules = im.get_multilevel_modules(states=False)
    else:
        modules = dict(im.multilevel_modules)

    for node_int, path in modules.items():
        lvl2_path = tuple(path[:min(len(path), 2)])
        if lvl2_path not in path_to_id:
            path_to_id[lvl2_path] = next_id
            next_id += 1
        partition[int_to_node[int(node_int)]] = path_to_id[lvl2_path]

    return partition


def calculate_raw_ei(G, partition):
    """
    [核心] 计算基于边数（Raw Degree）的 EI 指数
    """
    print("[5/6] Calculating Raw EI Indices (based on co-author counts)...")

    author_stats = []

    for node in G.nodes():
        comm_id = partition.get(node)

        # 获取所有邻居（即合作过的作者）
        neighbors = list(G.neighbors(node))
        total_partners = len(neighbors)

        if total_partners == 0:
            continue

        # 统计圈内合作人数
        internal_partners = 0
        for nbr in neighbors:
            if partition.get(nbr) == comm_id:
                internal_partners += 1

        # 统计圈外合作人数
        external_partners = total_partners - internal_partners

        # 计算个人 EI
        # EI = (Ext - Inn) / (Ext + Inn)
        ei_val = (external_partners - internal_partners) / total_partners

        author_stats.append({
            'AuthorID': node,
            'CommunityID': comm_id,
            'Total_Partners': total_partners,  # E + I
            'Internal_Partners': internal_partners,  # I
            'External_Partners': external_partners,  # E
            'Author_EI': ei_val
        })

    return pd.DataFrame(author_stats)


# =========================
# 3. 主程序
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载数据
    df, year_col = load_data(PATH_INPUT)

    # 2. 筛选
    print(f"[2/6] Filtering data (Year <= {TARGET_YEAR})...")
    df_slice = df[df[year_col] <= TARGET_YEAR].drop_duplicates(subset=["author_id", "paper_id"])

    # S5 过滤 (计算发文量)
    paper_counts = df_slice["author_id"].value_counts().to_dict()

    # 3. 建网
    G = build_network_structure(df_slice)
    print(f"    -> Initial Nodes: {G.number_of_nodes()}")

    # 4. 执行 S5 & S2 过滤
    # S5
    rem = [n for n in G.nodes() if paper_counts.get(n, 0) < 2]
    G.remove_nodes_from(rem)
    # S2
    if len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    print(f"    -> Filtered Nodes (S5+S2): {G.number_of_nodes()}")

    # 5. 社区划分
    partition = run_infomap(G)
    print(f"    -> Communities found: {len(set(partition.values()))}")

    # 6. 计算 EI
    df_authors = calculate_raw_ei(G, partition)

    # 7. 社区聚合统计
    print("[6/6] Aggregating Community Stats...")
    comm_stats = df_authors.groupby('CommunityID').agg(
        Size=('AuthorID', 'count'),
        Sum_Internal=('Internal_Partners', 'sum'),
        Sum_External=('External_Partners', 'sum'),
        Avg_Author_EI=('Author_EI', 'mean'),
        Median_Author_EI=('Author_EI', 'median')
    ).reset_index()

    # 计算社区整体 EI (Macro EI)
    # EI = (Sum_Ext - Sum_Int) / (Sum_Ext + Sum_Int)
    comm_stats['Community_EI'] = (comm_stats['Sum_External'] - comm_stats['Sum_Internal']) / \
                                 (comm_stats['Sum_External'] + comm_stats['Sum_Internal'])

    # 排序
    comm_stats = comm_stats.sort_values('Size', ascending=False)

    # 导出
    out_path = os.path.join(PATH_OUTDIR, "stats_2025_community_ei_raw.csv")
    comm_stats.to_csv(out_path, index=False)

    # 打印结果
    print("\n" + "=" * 50)
    print("TOP 10 COMMUNITIES E-I INDEX (Raw Degree / Edge Count)")
    print("(-1 = Closed/Insular, +1 = Open/Bridge)")
    print("=" * 50)
    print(comm_stats[['CommunityID', 'Size', 'Community_EI', 'Avg_Author_EI']].head(10).to_string(index=False))
    print("-" * 50)
    print(f"Results saved to: {out_path}")

    # 简单绘图预览
    try:
        top10 = comm_stats.head(10)
        plt.figure(figsize=(10, 5), dpi=100)
        bars = plt.bar(top10['CommunityID'].astype(str), top10['Community_EI'],
                       color='#0172be', alpha=0.8, edgecolor='black')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.ylim(-1.1, 1.1)
        plt.ylabel("E-I Index (Raw)")
        plt.title(f"Top 10 Communities E-I Index ({TARGET_YEAR})")

        for bar in bars:
            height = bar.get_height()
            xy = (bar.get_x() + bar.get_width() / 2, height)
            text_y = height + 0.05 if height >= 0 else height - 0.15
            plt.text(xy[0], text_y, f"{height:.2f}", ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
