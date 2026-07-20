# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_global_first_metrics_final_normalized.py

==============================================================================
【程序功能：Global-First 策略累积网络指标计算 (归一化修复版)】
==============================================================================
核心逻辑：
1. Global-First: 提取 2006-2025 全量骨干。
2. Time-Backtracking: 逐年回溯，严格约束在骨干范围内。
3. Normalization: 所有指标均归一化到 [0, 1] 区间，支持跨年份、跨规模比较。

修正点 (v22):
1. [重要] 修正了紧密中心势 (Closeness Centralization) 的 Freeman 分母公式。
2. 修正了输出路径，确保文件生成在脚本所在目录。
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
# 输出目录路径
PATH_OUTDIR = str(_OUTPUT_ROOT / "03_network_metrics" / "global_first_normalized_metrics")

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
    strength = {n: 0.0 for n in G.nodes()};
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
# 4. 指标计算 (Normalized)
# =========================

def calculate_structural_entropy(G):
    """
    计算结构熵 (Structural Entropy) - 衡量网络异质性
    标准化方法: 通过除以总重要性使 sum(I) = 1
    """
    try:
        n = len(G)
        if n == 0: return 0.0
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        k_counts = Counter(degree_values)
        total_nodes = len(degree_values)
        p_k = {k: count / total_nodes for k, count in k_counts.items()}
        raw_importance = []
        for node, k in degrees.items():
            prob = p_k.get(k, 0)
            val = (k + 1) * (1.0 - prob)
            raw_importance.append(val)
        total_raw = sum(raw_importance)
        if total_raw == 0: return 0.0
        norm_importance = [val / total_raw for val in raw_importance]
        entropy = 0.0
        for I in norm_importance:
            if I > 0: entropy += -1 * I * math.log(I)
        return entropy
    except Exception as e:
        print(f"    [Err] Entropy: {e}")
        return np.nan


def calculate_centralization(G, c_type="degree"):
    """
    计算 Freeman Centralization (归一化到 0-1)
    公式: sum(Max - Val_i) / Max_Possible_Sum
    """
    n = len(G)
    if n < 3: return 0.0

    try:
        if c_type == "degree":
            # nx.degree_centrality 已经返回 normalized (d / (n-1))
            # 理论最大差值和 (Star Graph): (n-2)
            cent = nx.degree_centrality(G)
            c_vals = list(cent.values())
            max_c = max(c_vals)
            denom = n - 2
            return sum(max_c - c for c in c_vals) / denom if denom > 0 else 0.0

        elif c_type == "betweenness":
            # nx.betweenness 默认归一化
            # 理论最大差值和: (n-1)
            k_sample = int(n * 0.15) if n > 4000 else None
            cent = nx.betweenness_centrality(G, k=k_sample)
            c_vals = list(cent.values())
            max_c = max(c_vals)
            return sum(max_c - c for c in c_vals) / (n - 1)

        elif c_type == "closeness":
            # nx.closeness 默认归一化
            # 理论最大差值和 (Star Graph):
            # Center=1.0, Leaf=(n-1)/(2n-3)
            # Sum of Diff = (n-1) * [1 - (n-1)/(2n-3)] = (n-1)(n-2)/(2n-3)
            cent = nx.closeness_centrality(G)
            c_vals = list(cent.values())
            max_c = max(c_vals)

            # [修正] 之前漏掉了 (n-1) 因子
            numerator = sum(max_c - c for c in c_vals)
            denominator = ((n - 1) * (n - 2)) / (2 * n - 3)

            return numerator / denominator if denominator > 0 else 0.0

        elif c_type == "strength":
            # 强度中心势 (Weighted Degree)
            # 无法计算理论上限，因此使用"相对归一化"
            # 衡量当前网络内部的强度分布不均程度 (类似 Gini)
            strength = dict(G.degree(weight='weight'))
            s_vals = list(strength.values())
            max_s = max(s_vals) if s_vals else 0
            if max_s == 0: return 0.0

            # 先将所有强度归一化到 [0,1] (相对于当前最大值)
            norm_vals = [s / max_s for s in s_vals]
            max_norm = 1.0  # 既然除以了 max_s，最大值必然是 1

            # 然后计算 Freeman 分母 (n-1)
            return sum(max_norm - v for v in norm_vals) / (n - 1)

    except Exception as e:
        print(f"    [Err] Cent ({c_type}): {e}")
        return np.nan
    return 0.0


def calculate_metrics_all(G):
    res = {}
    n = len(G)
    res['node_count'] = n
    res['edge_count'] = G.number_of_edges()
    keys = ['avg_path_len', 'deg_centralization', 'str_centralization',
            'clo_centralization', 'bet_centralization', 'struct_entropy']
    for k in keys: res[k] = np.nan

    if n < 3: return res

    res['struct_entropy'] = calculate_structural_entropy(G)

    if nx.is_connected(G):
        G_lcc = G
    else:
        G_lcc = filter_lcc_s2(G)

    if len(G_lcc) > 1:
        try:
            res['avg_path_len'] = nx.average_shortest_path_length(G_lcc)
        except:
            pass
    else:
        res['avg_path_len'] = 0.0

    res['deg_centralization'] = calculate_centralization(G, 'degree')
    res['str_centralization'] = calculate_centralization(G, 'strength')
    res['clo_centralization'] = calculate_centralization(G, 'closeness')

    print(f"      [Metrics] Calculating Betweenness (N={n})...", end="\r")
    res['bet_centralization'] = calculate_centralization(G, 'betweenness')

    return res


# =========================
# 5. 主程序
# =========================

def main():
    print(">>> 启动 Global-First 拓扑指标计算 (Normalized Fix)...")
    ensure_outdir(PATH_OUTDIR)

    df_all, year_col = load_data(PATH_INPUT)
    if not year_col: return

    records = []

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

        years = range(TIME_START, TIME_END + 1)
        for year in years:
            print(f"    Processing Year {year} ...", end="\r")
            df_cum = df_all[df_all[year_col] <= year].drop_duplicates(subset=[AUTHOR_COL, PAPER_COL])
            if df_cum.empty: continue

            auth_ids_y, paper_ids_y, df_idx_y = build_index_maps(df_cum)
            edges_y = build_network_matrix(METHOD, auth_ids_y, paper_ids_y, df_idx_y)

            G_cum = nx.Graph()

            # Global-First Filter
            nodes_in_year = set(df_cum[AUTHOR_COL].unique())
            valid_in_year = nodes_in_year.intersection(valid_nodes_set)
            G_cum.add_nodes_from(valid_in_year)

            for r in edges_y.itertuples(index=False):
                if r.src_author_id in valid_in_year and r.dst_author_id in valid_in_year:
                    edge_tuple = tuple(sorted((r.src_author_id, r.dst_author_id)))
                    if edge_tuple in valid_edges_set:
                        G_cum.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

            G_final = filter_isolates_s1(G_cum)
            m = calculate_metrics_all(G_final)

            rec = {
                "year": year, "pipeline": pipe_name,
                "node_count": m['node_count'], "edge_count": m['edge_count'],
                "avg_path_len": m['avg_path_len'],
                "deg_centralization": m['deg_centralization'],
                "str_centralization": m['str_centralization'],
                "clo_centralization": m['clo_centralization'],
                "bet_centralization": m['bet_centralization'],
                "struct_entropy": m['struct_entropy']
            }
            records.append(rec)
        print("")

    df_res = pd.DataFrame(records)
    out_path = os.path.join(PATH_OUTDIR, "network_metrics_normalized_final2.csv")
    print("-" * 30)
    if not records:
        print("Warning: No records generated.")
    else:
        df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f">>> Success! File saved to:\n{os.path.abspath(out_path)}")
    print(">>> All Done.")


if __name__ == "__main__":
    main()
