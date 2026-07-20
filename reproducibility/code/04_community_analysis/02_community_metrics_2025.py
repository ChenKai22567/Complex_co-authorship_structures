# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
2025 Network Statistics & Rich Metrics Extractor
------------------------------------------------
功能描述：
    1. 构建 2006-2025 的累计作者合作网络 (Co-authorship Network)。
    2. 执行严格的过滤流程 (S5:发文量>=2 -> S2:最大连通子图)。
    3. 运行 Infomap 算法获取二级社区划分 (Level 2)。
    4. [新增核心功能] 计算每个节点的“社区内加权度” (Internal Weighted Degree) 和 “总加权度”。
    5. 计算学术影响力指标 (g-index, h-index, Citations)。
    6. 导出包含上述所有指标的 CSV，用于后续绘制箱线图。

输出文件：
    - stats_2025_author_metrics_extended.csv: 每一行是一个作者，包含其社区ID、合作度指标、影响力指标。
    - stats_2025_community_summary.csv: 社区层面的汇总统计 (Size, Avg Metrics)。

依赖库：
    pip install pandas networkx infomap scipy numpy
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
# 输入数据路径
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
# 输出目录
PATH_OUTDIR = str(_OUTPUT_ROOT / "04_community_analysis" / "community_metrics_2025")

# 目标年份
TARGET_YEAR = 2025

# Infomap 参数 (保持一致性)
INFOMAP_ARGS = "--markov-time 1.1 -N 1000 --seed 42"  # -N 20 保证精度，单年跑一次很快


# =========================
# 2. 基础工具函数
# =========================
def ensure_outdir(path):
    """确保输出目录存在"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path):
    """加载原始数据并处理年份与引用列格式"""
    print(f"[Data] Reading raw csv: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)

    # 自动查找年份列
    year_col = next((c for c in df.columns if c in ["publication_year", "year", "pub_year", "PY"]), None)
    if year_col is None:
        raise ValueError("Cannot find year column.")

    # 清洗关键列
    df = df.dropna(subset=["author_id", "paper_id"])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)

    # 清洗引用列
    if "times_cited" in df.columns:
        df["times_cited"] = pd.to_numeric(df["times_cited"], errors='coerce').fillna(0).astype(int)
    else:
        df["times_cited"] = 0

    return df, year_col


def build_network_weighted(df_slice):
    """
    构建加权作者合作网络 (Newman Weighting Scheme)
    权重 w_uv = sum(1 / (n_p - 1))，其中 n_p 是论文 p 的作者数。
    """
    print("[Network] Building weighted projection...")

    # 获取唯一的作者和论文ID
    a_ids = df_slice["author_id"].unique()
    p_ids = df_slice["paper_id"].unique()

    # 建立 ID 到 索引 的映射
    aid2idx = {a: i for i, a in enumerate(a_ids)}
    pid2idx = {p: i for i, p in enumerate(p_ids)}

    # 构建稀疏矩阵坐标
    row_ind = df_slice["author_id"].map(aid2idx).astype(int)
    col_ind = df_slice["paper_id"].map(pid2idx).astype(int)

    # 创建二部图矩阵 A (Authors x Papers)
    # df_slice 必须是去重后的 (author, paper) 对，否则权重会计算错误
    A = sp.csr_matrix(
        (np.ones(len(df_slice), dtype=np.float32), (row_ind, col_ind)),
        shape=(len(a_ids), len(p_ids))
    )

    # 计算单篇论文作者数 n
    n = np.asarray(A.sum(axis=0)).ravel()

    # 计算Newman权重 w = 1/(n-1)
    w = np.zeros_like(n, dtype=np.float32)
    mask = n > 1
    w[mask] = 1.0 / (n[mask] - 1.0)

    # 矩阵乘法投影 U = A * diag(w) * A.T
    U = A @ sp.diags(w) @ A.T
    U.setdiag(0)  # 移除自环
    U.eliminate_zeros()

    # 转换为 NetworkX 图
    G = nx.from_scipy_sparse_array(U)
    # 还原节点 Label
    mapping = {i: aid for i, aid in enumerate(a_ids)}
    G = nx.relabel_nodes(G, mapping)

    return G


def run_infomap_level2(G):
    """
    运行 Infomap 并提取 Level 2 社区。
    返回: {node_id: community_id (int)}
    """
    print(f"[Community] Running Infomap ({INFOMAP_ARGS})...")
    if G.number_of_nodes() == 0:
        return {}

    im = infomap.Infomap(INFOMAP_ARGS)

    # 映射节点为整数ID供 Infomap 使用
    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    # 添加边
    for u, v, d in G.edges(data=True):
        im.add_link(node_to_int[u], node_to_int[v], float(d.get("weight", 1.0)))

    im.run()

    # 提取多层模块结构
    partition = {}
    path_to_id = {}
    next_id = 1

    # 兼容新旧版本 API
    if hasattr(im, "get_multilevel_modules"):
        modules = im.get_multilevel_modules(states=False)
    else:
        modules = dict(im.multilevel_modules)

    for node_int, path_tuple in modules.items():
        if path_tuple is None: continue
        # 截取前两层作为 Level 2 标识
        target_path = tuple(path_tuple[:min(len(path_tuple), 2)])

        # 将路径元组映射为连续整数 ID
        if target_path not in path_to_id:
            path_to_id[target_path] = next_id
            next_id += 1

        original_node = int_to_node[int(node_int)]
        partition[original_node] = path_to_id[target_path]

    print(f"  -> Found {len(path_to_id)} Level-2 communities.")
    return partition


def calculate_h_g_indices(citations):
    """计算单个学者的 h-index 和 g-index"""
    if not citations:
        return 0, 0
    # 降序排列引用
    citations = sorted(citations, reverse=True)

    # h-index
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break

    # g-index
    g = 0
    cumulative = 0
    for i, c in enumerate(citations):
        cumulative += c
        # g 是最大的数，满足前 g 篇论文的总引用 >= g^2
        if cumulative >= (i + 1) ** 2:
            g = i + 1
        else:
            break

    return h, g


# =========================
# 3. [核心新增] 加权度计算函数
# =========================
def calculate_advanced_network_metrics(G, partition):
    """
    计算网络结构指标，特别是社区内加权度。

    返回字典: {node_id: {'Total_Weighted_Degree': float, 'Internal_Weighted_Degree': float}}
    """
    print("[Metrics] Calculating Internal Weighted Degrees...")
    metrics = {}

    for node in G.nodes():
        # 获取该节点的社区 ID
        node_comm = partition.get(node)

        total_w_deg = 0.0
        internal_w_deg = 0.0

        # 遍历该节点的所有邻居和边权重
        # G[node] 返回邻居字典 {neighbor: {attr: val}}
        for neighbor, attr in G[node].items():
            weight = attr.get('weight', 1.0)
            total_w_deg += weight

            # 如果邻居在同一个社区，则计入内部加权度
            if partition.get(neighbor) == node_comm:
                internal_w_deg += weight

        metrics[node] = {
            'Total_Weighted_Degree': total_w_deg,  # 总合作强度
            'Internal_Weighted_Degree': internal_w_deg  # 社区内部合作强度 (抱团程度)
        }

    return metrics


# =========================
# 4. 主流程逻辑
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载数据
    df, year_col = load_data(PATH_INPUT)

    # 2. 筛选截止到目标年份的数据
    print(f"\n[Processing] Filtering data <= {TARGET_YEAR}...")
    # 必须先按(author, paper)去重，保证二部图构建正确
    df_slice = df[df[year_col] <= TARGET_YEAR].drop_duplicates(subset=["author_id", "paper_id"])

    # 3. 计算发文量 (用于 S5 过滤)
    paper_counts = df_slice["author_id"].value_counts().to_dict()

    # 4. 构建网络
    G = build_network_weighted(df_slice)
    print(f"  -> Raw Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 5. 执行过滤管道 (S5 -> S2)
    # S5: 发文量 >= 2
    rem_s5 = [n for n in G.nodes() if paper_counts.get(n, 0) < 2]
    G.remove_nodes_from(rem_s5)
    print(f"  -> After S5 (Papers>=2): {G.number_of_nodes()} nodes")

    # S2: 最大连通分量
    if len(G) > 0 and nx.number_connected_components(G) > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    print(f"  -> After S2 (Giant Component): {G.number_of_nodes()} nodes")

    if len(G) == 0:
        print("Error: Graph is empty after filtering.")
        return

    # 6. 社区划分
    partition = run_infomap_level2(G)

    # 7. [新增] 计算加权度指标 (Total & Internal)
    net_metrics = calculate_advanced_network_metrics(G, partition)

    # 8. 计算影响力指标 (Citations, h/g-index)
    print("[Metrics] Calculating Influence Metrics (Citations, h/g-index)...")
    # 准备引用数据字典: author -> [citations list]
    # 只取当前过滤后网络中的节点数据
    valid_nodes = set(G.nodes())
    df_subset = df_slice[df_slice['author_id'].isin(valid_nodes)]
    auth_citations_map = df_subset.groupby('author_id')['times_cited'].apply(list).to_dict()

    # 9. 整合所有数据并导出
    print("[Export] Generating final CSVs...")
    rows = []
    for node in G.nodes():
        # 获取网络指标
        nm = net_metrics.get(node, {'Total_Weighted_Degree': 0, 'Internal_Weighted_Degree': 0})

        # 获取引用指标
        cites = auth_citations_map.get(node, [])
        h, g = calculate_h_g_indices(cites)

        # 获取基本属性
        comm_id = partition.get(node, 0)
        p_count = paper_counts.get(node, 0)

        rows.append({
            "AuthorID": node,
            "CommunityID": comm_id,
            "Paper_Count": p_count,
            "Total_Citations": sum(cites),
            "h_index": h,
            "g_index": g,
            "Total_Weighted_Degree": round(nm['Total_Weighted_Degree'], 4),  # 合作广度
            "Internal_Weighted_Degree": round(nm['Internal_Weighted_Degree'], 4)  # 合作深度
        })

    df_output = pd.DataFrame(rows)

    # 导出作者明细表 (用于箱线图)
    out_path_authors = os.path.join(PATH_OUTDIR, f"stats_{TARGET_YEAR}_author_metrics_extended.csv")
    df_output.to_csv(out_path_authors, index=False)

    # 导出社区汇总表 (辅助查看 Top 10)
    df_comm_summary = df_output.groupby("CommunityID").agg(
        Size=("AuthorID", "count"),
        Avg_Internal_Degree=("Internal_Weighted_Degree", "mean"),
        Avg_g_index=("g_index", "mean"),
        Total_Citations=("Total_Citations", "sum")
    ).reset_index().sort_values("Size", ascending=False)

    out_path_comm = os.path.join(PATH_OUTDIR, f"stats_{TARGET_YEAR}_community_summary.csv")
    df_comm_summary.to_csv(out_path_comm, index=False)

    print(f"\nSuccess! Files saved:")
    print(f"  1. Author Metrics (for Boxplot): {out_path_authors}")
    print(f"  2. Community Summary: {out_path_comm}")


if __name__ == "__main__":
    main()
