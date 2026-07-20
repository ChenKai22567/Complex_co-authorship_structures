# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
Network Data Export Script V4 (Rich Metrics + Community Details)

功能升级：
在原有基础上，新增了“逐年社区详细指标”的计算与导出。
统计指标包括：社区大小、平均度、平均加权度、社区内文章的篇均被引频次。

输出文件：
1. stats_evolution_summary_rich.csv (年度全网概览)
2. stats_evolution_comm_sizes.csv (仅社区大小，保留用于快速查看)
3. [NEW] stats_evolution_community_details.csv (包含每年每个社区的度、引用等详细指标)
4. stats_2025_author_metrics.csv (最终年作者详情)
5. stats_2025_community_metrics.csv (最终年社区详情)
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import infomap
from collections import Counter
import scipy.sparse as sp

# =========================
# 配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "04_community_analysis" / "temporal_community_metrics")

START_YEAR = 2006
END_YEAR = 2025

# Infomap 参数
INFOMAP_ARGS = "--markov-time 1.1 -N 1000 --seed 42"


# =========================
# 基础函数
# =========================
def ensure_outdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path):
    print(f"Reading data: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    year_col = next((c for c in df.columns if c in ["publication_year", "year", "pub_year", "PY"]), None)
    if year_col is None:
        raise ValueError("Cannot find year column.")

    df = df.dropna(subset=["author_id", "paper_id"])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)

    if "times_cited" in df.columns:
        df["times_cited"] = pd.to_numeric(df["times_cited"], errors='coerce').fillna(0).astype(int)
    else:
        df["times_cited"] = 0

    return df, year_col


def build_network_fast(df_slice):
    """构建加权网络"""
    a_ids = df_slice["author_id"].unique()
    p_ids = df_slice["paper_id"].unique()

    aid2idx = {a: i for i, a in enumerate(a_ids)}
    pid2idx = {p: i for i, p in enumerate(p_ids)}

    row_ind = df_slice["author_id"].map(aid2idx).astype(int)
    col_ind = df_slice["paper_id"].map(pid2idx).astype(int)

    A = sp.csr_matrix(
        (np.ones(len(df_slice), dtype=np.float32), (row_ind, col_ind)),
        shape=(len(a_ids), len(p_ids))
    )

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


def filter_pipeline_like_single_year(G, df_slice_dedup):
    """过滤：S5 + S2"""
    # S5
    paper_counts = df_slice_dedup["author_id"].value_counts().to_dict()
    rem = [n for n in G.nodes() if paper_counts.get(n, 0) < 2]
    if rem:
        G.remove_nodes_from(rem)

    # S2
    if len(G) > 0 and nx.number_connected_components(G) > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


def run_infomap_level2_like_single_year(G):
    """Infomap 社区划分"""
    if G.number_of_nodes() == 0:
        return {}

    im = infomap.Infomap(INFOMAP_ARGS)
    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    for u, v, d in G.edges(data=True):
        im.add_link(node_to_int[u], node_to_int[v], float(d.get("weight", 1.0)))

    im.run()

    if hasattr(im, "get_multilevel_modules"):
        modules = im.get_multilevel_modules(states=False)
    else:
        modules = dict(im.multilevel_modules)

    partition = {}
    path_to_id = {}
    next_id = 1

    for node_int, path_tuple in modules.items():
        if path_tuple is None: continue
        level2_path = tuple(path_tuple[:min(len(path_tuple), 2)])
        if level2_path not in path_to_id:
            path_to_id[level2_path] = next_id
            next_id += 1
        partition[int_to_node[int(node_int)]] = path_to_id[level2_path]

    return partition


def calculate_h_g_indices(citations_list):
    """计算 h/g 指数"""
    if not citations_list: return 0, 0
    citations = sorted(citations_list, reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    g = 0
    cumulative = 0
    for i, c in enumerate(citations):
        cumulative += c
        if cumulative >= (i + 1) ** 2:
            g = i + 1
        else:
            break
    return h, g


def calculate_network_rich_metrics(G, df_source):
    """计算全网聚合指标"""
    if len(G) == 0: return 0, 0, 0, 0, 0, 0

    nodes_in_graph = set(G.nodes())
    df_subset = df_source[df_source['author_id'].isin(nodes_in_graph)]

    auth_citations_map = df_subset.groupby('author_id')['times_cited'].apply(list).to_dict()
    paper_count_map = df_subset.groupby('author_id').size().to_dict()

    g_list, h_list, total_citations_list, paper_count_list = [], [], [], []

    for node in nodes_in_graph:
        cites = auth_citations_map.get(node, [])
        h, g = calculate_h_g_indices(cites)
        g_list.append(g)
        h_list.append(h)
        total_citations_list.append(sum(cites))
        paper_count_list.append(paper_count_map.get(node, 0))

    return (int(sum(total_citations_list)),
            float(np.mean(total_citations_list)) if total_citations_list else 0.0,
            float(np.mean(g_list)) if g_list else 0.0,
            int(np.max(g_list)) if g_list else 0,
            float(np.mean(h_list)) if h_list else 0.0,
            float(np.mean(paper_count_list)) if paper_count_list else 0.0)


# =========================
# 处理逻辑 (主要修改区)
# =========================
def process_evolution(df, year_col, out_dir):
    summary_rows = []
    size_rows = []

    # [NEW] 用于存储每年每个社区的详细指标
    community_detail_rows = []

    final_year_data = {}

    print("\n>>> Phase 1: Processing Yearly Evolution (with Community Details) <<<")

    for year in range(START_YEAR, END_YEAR + 1):
        # 1) 累计切片 + 去重
        df_slice = df[df[year_col] <= year].drop_duplicates(subset=["author_id", "paper_id"])

        if df_slice.empty:
            print(f"  Year {year}: Empty")
            continue

        # 2) 建网
        G = build_network_fast(df_slice)

        # 3) 过滤
        G = filter_pipeline_like_single_year(G, df_slice)
        num_nodes = G.number_of_nodes()

        if num_nodes == 0:
            print(f"  Year {year}: Empty after filtering")
            continue

        # 4) 社区划分
        partition = run_infomap_level2_like_single_year(G)

        # -------------------------------------------------------------
        # [NEW] 计算社区详细指标 (度、加权度、篇均被引)
        # -------------------------------------------------------------
        # 准备数据：给 df_slice 打上社区标签
        df_metrics = df_slice.copy()
        # 仅保留在图中的节点
        valid_nodes = set(G.nodes())
        df_metrics = df_metrics[df_metrics['author_id'].isin(valid_nodes)]
        df_metrics['comm_id'] = df_metrics['author_id'].map(partition)

        # 反转 Partition: CommID -> [Nodes]
        comm_to_nodes = {}
        for node, comm_id in partition.items():
            comm_to_nodes.setdefault(comm_id, []).append(node)

        # 遍历该年份的所有社区
        for comm_id, members in comm_to_nodes.items():
            # A. 拓扑指标 (Degree)
            # 1. 平均度 (Degree): 节点的连边数量
            degrees = [G.degree(n) for n in members]
            avg_deg = np.mean(degrees) if degrees else 0.0

            # 2. 平均加权度 (Weighted Degree): 节点的连边权重之和
            weighted_degrees = [G.degree(n, weight='weight') for n in members]
            avg_w_deg = np.mean(weighted_degrees) if weighted_degrees else 0.0

            # B. 引用指标 (Citation)
            # 筛选该社区的数据
            comm_df = df_metrics[df_metrics['comm_id'] == comm_id]

            # 【核心】：在社区内部对 paper_id 去重，计算真实的篇均被引
            unique_papers = comm_df.drop_duplicates(subset=['paper_id'])
            total_citations = unique_papers['times_cited'].sum()
            paper_count = len(unique_papers)
            avg_cit_per_paper = total_citations / paper_count if paper_count > 0 else 0.0

            # 记录详细数据
            community_detail_rows.append({
                "Year": year,
                "CommunityID": comm_id,
                "Size": len(members),
                "Avg_Degree": round(avg_deg, 4),
                "Avg_Weighted_Degree": round(avg_w_deg, 4),
                "Avg_Citations_Per_Paper": round(avg_cit_per_paper, 4),
                "Total_Citations": total_citations,
                "Paper_Count": paper_count
            })

            # 同时记录简单的 Size 用于 stats_evolution_comm_sizes.csv (保持旧逻辑)
            size_rows.append({"Year": year, "Size": len(members)})

        # -------------------------------------------------------------
        # 5) 全网汇总指标 (保持不变)
        # -------------------------------------------------------------
        sizes = list(Counter(partition.values()).values())
        num_comms = len(sizes)
        num_edges = G.number_of_edges()
        avg_degree_net = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # 全网深度指标
        tot_cit, avg_cit, avg_g, max_g, avg_h, avg_pap = calculate_network_rich_metrics(G, df_slice)

        summary_rows.append({
            "Year": year,
            "Nodes": num_nodes,
            "Edges": num_edges,
            "Communities": num_comms,
            "Avg_Degree": round(avg_degree_net, 4),
            "Total_Citations": tot_cit,
            "Avg_Citations": round(avg_cit, 2),
            "Avg_g_index": round(avg_g, 4),
            "Max_g_index": max_g,
            "Avg_h_index": round(avg_h, 4),
            "Avg_Paper_Count": round(avg_pap, 4)
        })

        print(f"  Year {year}: Nodes={num_nodes}, Comms={num_comms}, Data Extracted.")

        if year == END_YEAR:
            final_year_data = {"G": G, "Partition": partition, "DF": df_slice}

    # 导出文件
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "stats_evolution_summary_rich.csv"), index=False)
    pd.DataFrame(size_rows).to_csv(os.path.join(out_dir, "stats_evolution_comm_sizes.csv"), index=False)

    # [NEW] 导出社区详细指标表
    pd.DataFrame(community_detail_rows).to_csv(os.path.join(out_dir, "stats_evolution_community_details.csv"),
                                               index=False)

    print(f"Saved all evolution stats to {out_dir}")

    return final_year_data


def process_2025_metrics(final_data, out_dir):
    """保持不变：生成最终年的详细作者和社区报告"""
    if not final_data: return
    print("\n>>> Phase 2: Calculating 2025 Detailed Metrics <<<")
    G = final_data["G"]
    partition = final_data["Partition"]
    df_slice = final_data["DF"]

    df_auth_stats = df_slice.groupby("author_id")["times_cited"].apply(list).to_dict()
    df_paper_counts = df_slice.groupby("author_id").size().to_dict()

    author_rows = []
    for node in G.nodes():
        citations = df_auth_stats.get(node, [])
        h, g = calculate_h_g_indices(citations)
        author_rows.append({
            "AuthorID": node,
            "CommunityID": partition.get(node, 0),
            "Paper_Count": df_paper_counts.get(node, 0),
            "Total_Citations": sum(citations),
            "h_index": h, "g_index": g
        })

    df_authors = pd.DataFrame(author_rows)
    df_authors.to_csv(os.path.join(out_dir, f"stats_{END_YEAR}_author_metrics.csv"), index=False)

    comm_stats = df_authors.groupby("CommunityID").agg(
        Size=("AuthorID", "count"),
        Total_Citations=("Total_Citations", "sum"),
        Avg_g_index=("g_index", "mean"),
        Max_g_index=("g_index", "max"),
        Avg_h_index=("h_index", "mean"),
        Avg_Paper_Count=("Paper_Count", "mean")
    ).reset_index().sort_values(by="Size", ascending=False)

    comm_stats.to_csv(os.path.join(out_dir, f"stats_{END_YEAR}_community_metrics.csv"), index=False)
    print("Saved 2025 detailed metrics.")


# =========================
# 主程序
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)
    df, year_col = load_data(PATH_INPUT)
    final_data = process_evolution(df, year_col, PATH_OUTDIR)
    process_2025_metrics(final_data, PATH_OUTDIR)
    print("\nAll data exported successfully.")


if __name__ == "__main__":
    main()
