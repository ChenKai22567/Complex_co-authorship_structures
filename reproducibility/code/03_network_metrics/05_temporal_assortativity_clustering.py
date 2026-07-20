# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_metrics_v30_active_jaccard.py

==============================================================================
【程序功能：综合指标计算 (累积拓扑 + 活跃窗口稳定性)】
==============================================================================
核心逻辑差异 (与 v28/v29 相比):
1. 拓扑指标 (Assortativity, Clustering):
   基于 [累积网络 G_cumulative] 计算。反映网络整体结构的演化。

2. 稳定性指标 (Jaccard Similarity):
   基于 [活跃窗口网络 G_active] 计算。
   即：计算 Year(t) 与 Year(t-1) 的活跃节点/边的重叠度。
   这能有效衡量“当前活跃群体”的固化程度，避免因历史数据堆积导致的 Jaccard 虚高。

输出文件：
   network_metrics_v30_active_jaccard.csv
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

# 尝试导入 SciPy 以加速矩阵构建
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    print("Warning: SciPy not found. Large network construction will be slow.")

# =========================
# 1. 全局配置区
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
# 输出目录
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "temporal_assortativity_clustering")

AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

METHOD = "newman"
S3_ALPHA = 0.1
MIN_PAPER_COUNT = 2
TIME_START = 2006
TIME_END = 2025

PIPELINES = {
    "S2": ["S2"],
    "S5-S2": ["S5", "S2"],
    "S5-S3-S2": ["S5", "S3", "S2"]
}


# =========================
# 2. 工具函数
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
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
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
# 3. 构图与过滤
# =========================

def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False);
    U.setdiag(0);
    U = U.tocsr(copy=False);
    U.eliminate_zeros();
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
    if not HAVE_SCIPY: raise RuntimeError("Need SciPy library.")
    A = sp.csr_matrix((np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
                      shape=(len(author_ids), len(paper_ids)))
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


def filter_isolates_s1(G):
    if G.number_of_nodes() == 0: return G
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    return G


def filter_lcc_s2(G):
    if G.number_of_nodes() == 0: return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


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
        k_u = degree[u]
        a_u = (max(0.0, 1.0 - w / strength[u]) ** (k_u - 1)) if k_u > 1 else 0.0
        k_v = degree[v]
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
        elif step == "S2":
            G_curr = filter_lcc_s2(G_curr)
        elif step == "S3":
            G_curr = filter_disparity_s3(G_curr, alpha_val)
        elif step == "S5":
            G_curr = filter_min_papers_s5(G_curr, paper_counts, min_paper_count)
    return G_curr


# =========================
# 4. 指标计算核心
# =========================

def calculate_topology_metrics(G):
    """计算拓扑指标 (基于累积网络)"""
    res = {
        'assortativity': np.nan,
        'avg_clustering': np.nan
    }
    n = len(G)
    if n < 2: return res

    # 1. 同配系数
    try:
        res['assortativity'] = nx.degree_assortativity_coefficient(G)
    except Exception:
        pass

    # 2. 平均聚类系数
    try:
        res['avg_clustering'] = nx.average_clustering(G)
    except Exception:
        pass

    return res


def calculate_active_jaccard(G_curr_active, G_prev_active):
    """
    计算基于[活跃窗口]的 Jaccard 相似度
    反映相邻年份活跃群体的重叠程度
    """
    res = {
        'active_jaccard_nodes': np.nan,
        'active_jaccard_edges': np.nan,
        'active_new_nodes': 0,
        'active_overlap_nodes': 0
    }

    nodes_curr = set(G_curr_active.nodes())
    edges_curr = set(tuple(sorted((u, v))) for u, v in G_curr_active.edges())

    if G_prev_active is None:
        # 第一年
        res['active_new_nodes'] = len(nodes_curr)
        return res

    nodes_prev = set(G_prev_active.nodes())
    edges_prev = set(tuple(sorted((u, v))) for u, v in G_prev_active.edges())

    # Node Jaccard
    n_intersect = len(nodes_curr.intersection(nodes_prev))
    n_union = len(nodes_curr.union(nodes_prev))
    if n_union > 0:
        res['active_jaccard_nodes'] = n_intersect / n_union
    else:
        res['active_jaccard_nodes'] = 0.0

    # Edge Jaccard
    e_intersect = len(edges_curr.intersection(edges_prev))
    e_union = len(edges_curr.union(edges_prev))
    if e_union > 0:
        res['active_jaccard_edges'] = e_intersect / e_union
    else:
        res['active_jaccard_edges'] = 0.0

    res['active_overlap_nodes'] = n_intersect
    res['active_new_nodes'] = len(nodes_curr - nodes_prev)

    return res


# =========================
# 5. 主程序
# =========================

def main():
    print(">>> 启动综合指标计算 (v30 - Active Window Jaccard) ...")
    ensure_outdir(PATH_OUTDIR)

    df_all, year_col = load_data(PATH_INPUT)
    if not year_col: return

    records = []

    # Phase 1: 构建全局骨干 (全量时间范围)
    print("\n--- Phase 1: Building Global Backbone ---")
    df_global_unique = df_all.drop_duplicates(subset=[AUTHOR_COL, PAPER_COL])
    auth_ids, paper_ids, df_idx = build_index_maps(df_global_unique)
    edges_g = build_network_matrix(METHOD, auth_ids, paper_ids, df_idx)

    G_global_base = nx.Graph()
    G_global_base.add_nodes_from(df_global_unique[AUTHOR_COL].unique())
    for r in edges_g.itertuples(index=False):
        G_global_base.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

    global_counts = df_global_unique[AUTHOR_COL].value_counts().to_dict()

    for pipe_name, pipeline in PIPELINES.items():
        print(f"\n>>> Processing Pipeline: {pipe_name}")

        G_global_filtered = run_filter_pipeline(G_global_base, pipeline, S3_ALPHA, global_counts, MIN_PAPER_COUNT)

        valid_nodes_set = set(G_global_filtered.nodes())
        valid_edges_set = set()
        for u, v in G_global_filtered.edges():
            valid_edges_set.add(tuple(sorted((u, v))))

        print(f"    Global Backbone: Nodes={len(valid_nodes_set)}")
        if not valid_nodes_set: continue

        # 变量初始化
        G_active_prev = None  # 用于存储上一年的活跃网络

        # Phase 2: 时间回溯
        years = range(TIME_START, TIME_END + 1)
        for year in years:
            print(f"    Year {year} ...", end="\r")

            # --- 数据准备 ---
            # 1. 累积数据 (<= year) -> 用于拓扑指标
            df_cum = df_all[df_all[year_col] <= year].drop_duplicates(subset=[AUTHOR_COL, PAPER_COL])
            # 2. 活跃数据 (== year) -> 用于 Jaccard 稳定性
            df_act = df_all[df_all[year_col] == year].drop_duplicates(subset=[AUTHOR_COL, PAPER_COL])

            # --- 构建 G_cumulative (用于拓扑) ---
            G_cum = nx.Graph()
            if not df_cum.empty:
                aid_c, pid_c, idx_c = build_index_maps(df_cum)
                edge_c = build_network_matrix(METHOD, aid_c, pid_c, idx_c)
                # 节点过滤
                nodes_in_year = set(df_cum[AUTHOR_COL].unique())
                valid_in_year = nodes_in_year.intersection(valid_nodes_set)
                G_cum.add_nodes_from(valid_in_year)
                # 边过滤
                for r in edge_c.itertuples(index=False):
                    if r.src_author_id in valid_in_year and r.dst_author_id in valid_in_year:
                        if tuple(sorted((r.src_author_id, r.dst_author_id))) in valid_edges_set:
                            G_cum.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)
                G_cum = filter_isolates_s1(G_cum)

            # --- 构建 G_active (用于 Jaccard) ---
            G_active = nx.Graph()
            if not df_act.empty:
                aid_a, pid_a, idx_a = build_index_maps(df_act)
                edge_a = build_network_matrix(METHOD, aid_a, pid_a, idx_a)
                # 节点过滤
                nodes_in_act = set(df_act[AUTHOR_COL].unique())
                valid_in_act = nodes_in_act.intersection(valid_nodes_set)
                G_active.add_nodes_from(valid_in_act)
                # 边过滤
                for r in edge_a.itertuples(index=False):
                    if r.src_author_id in valid_in_act and r.dst_author_id in valid_in_act:
                        if tuple(sorted((r.src_author_id, r.dst_author_id))) in valid_edges_set:
                            G_active.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)
                G_active = filter_isolates_s1(G_active)

            # --- 指标计算 ---

            # 1. 累积拓扑
            topo = calculate_topology_metrics(G_cum)

            # 2. 活跃 Jaccard (对比 G_active 和 G_active_prev)
            stab = calculate_active_jaccard(G_active, G_active_prev)

            rec = {
                "year": year,
                "pipeline": pipe_name,
                "cumulative_nodes": G_cum.number_of_nodes(),
                "cumulative_edges": G_cum.number_of_edges(),
                # Topology (Cumulative)
                "assortativity": topo['assortativity'],
                "avg_clustering": topo['avg_clustering'],
                # Stability (Active Window)
                "active_jaccard_nodes": stab['active_jaccard_nodes'],
                "active_jaccard_edges": stab['active_jaccard_edges'],
                "active_overlap_nodes": stab['active_overlap_nodes'],
                "active_current_nodes": G_active.number_of_nodes()
            }
            records.append(rec)

            # 更新状态
            G_active_prev = G_active.copy()

        print("")

    # 导出
    df_res = pd.DataFrame(records)
    out_path = os.path.join(PATH_OUTDIR, "network_metrics_v30_active_jaccard.csv")
    print("-" * 30)
    if not records:
        print("Warning: No records generated.")
    else:
        df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f">>> Success! File saved to:\n{os.path.abspath(out_path)}")
    print(">>> All Done.")


if __name__ == "__main__":
    main()
