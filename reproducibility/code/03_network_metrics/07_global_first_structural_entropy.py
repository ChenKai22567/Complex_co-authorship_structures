# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_global_first_metrics_entropy_pure.py

==============================================================================
【程序功能：纯净版结构熵计算 (Pure Structural Entropy)】
==============================================================================
核心逻辑：
1. Global-First: 提取 2006-2025 全量骨干。
2. Time-Backtracking: 逐年回溯。
3. Pure Entropy: 移除了包括平均路径在内的所有耗时指标，只计算熵。

修正点 (v25):
1. [极速优化] 移除了 nx.average_shortest_path_length 计算。
==============================================================================
"""

import os
import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

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
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "global_first_structural_entropy")

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
        strength[u] += w
        strength[v] += w
        degree[u] += 1
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
# 4. 指标计算 (Pure Entropy)
# =========================

def calculate_structural_entropy(G, delta=0.0001):
    """
    计算结构熵 (Structural Entropy)
    """
    try:
        n = len(G)
        if n <= 1: return 0.0, 0.0

        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        k_counts = Counter(degree_values)
        total_nodes = len(degree_values)

        # p(k) = count(k) / N
        p_k = {k: count / total_nodes for k, count in k_counts.items()}

        raw_importance = []
        for node, k in degrees.items():
            prob = p_k.get(k, 0)
            # I_i 核心公式
            val = (k + 1) * (1.0 - prob + delta)
            raw_importance.append(val)

        total_raw = sum(raw_importance)
        if total_raw == 0: return 0.0, 0.0

        # 归一化 I (分布概率)
        norm_importance = [val / total_raw for val in raw_importance]

        entropy_abs = 0.0
        for I in norm_importance:
            if I > 0: entropy_abs += -1 * I * math.log(I)  # ln(I)

        max_entropy = math.log(n)
        entropy_norm = entropy_abs / max_entropy if max_entropy > 0 else 0.0

        return entropy_abs, entropy_norm

    except Exception as e:
        print(f"    [Err] Entropy: {e}")
        return np.nan, np.nan


def calculate_metrics_all(G):
    res = {}
    n = len(G)
    res['node_count'] = n
    res['edge_count'] = G.number_of_edges()

    # [优化] 仅计算结构熵，移除平均路径计算
    ent_abs, ent_norm = calculate_structural_entropy(G, delta=0.0001)
    res['struct_entropy_abs'] = ent_abs
    res['struct_entropy_norm'] = ent_norm

    return res


# =========================
# 5. 主程序
# =========================

def main():
    print(">>> 启动 Global-First 纯净版结构熵计算 ...")
    ensure_outdir(PATH_OUTDIR)

    df_all, year_col = load_data(PATH_INPUT)
    if not year_col: return

    records = []

    # Phase 1: 构建全局骨干
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

        print(f"    Global Backbone: Nodes={len(valid_nodes_set)}, Edges={len(valid_edges_set)}")
        if not valid_nodes_set: continue

        # Phase 2: 时间回溯
        years = range(TIME_START, TIME_END + 1)
        for year in years:
            print(f"    Processing Year {year} ...", end="\r")
            df_cum = df_all[df_all[year_col] <= year].drop_duplicates(subset=[AUTHOR_COL, PAPER_COL])
            if df_cum.empty: continue

            auth_ids_y, paper_ids_y, df_idx_y = build_index_maps(df_cum)
            edges_y = build_network_matrix(METHOD, auth_ids_y, paper_ids_y, df_idx_y)

            G_cum = nx.Graph()
            nodes_in_year = set(df_cum[AUTHOR_COL].unique())
            valid_in_year = nodes_in_year.intersection(valid_nodes_set)
            G_cum.add_nodes_from(valid_in_year)

            for r in edges_y.itertuples(index=False):
                if r.src_author_id in valid_in_year and r.dst_author_id in valid_in_year:
                    edge_tuple = tuple(sorted((r.src_author_id, r.dst_author_id)))
                    if edge_tuple in valid_edges_set:
                        G_cum.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

            G_final = filter_isolates_s1(G_cum)

            # 计算
            m = calculate_metrics_all(G_final)

            rec = {
                "year": year, "pipeline": pipe_name,
                "node_count": m['node_count'], "edge_count": m['edge_count'],
                "struct_entropy_abs": m['struct_entropy_abs'],
                "struct_entropy_norm": m['struct_entropy_norm']
            }
            records.append(rec)
        print("")

    # 导出
    df_res = pd.DataFrame(records)
    out_path = os.path.join(PATH_OUTDIR, "network_metrics_entropy_pure.csv")
    print("-" * 30)
    if not records:
        print("Warning: No records generated.")
    else:
        df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f">>> Success! File saved to:\n{os.path.abspath(out_path)}")
    print(">>> All Done.")


if __name__ == "__main__":
    main()
