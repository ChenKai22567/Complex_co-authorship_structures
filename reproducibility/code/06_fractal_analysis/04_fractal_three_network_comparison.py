from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# =========================
# 1. 配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
OUTPUT_DIR = str(_OUTPUT_ROOT / "06_fractal_analysis" / "fractal_three_network_comparison")
DATA_FILENAME = "fractal_data_continuous.csv"  # 改个名以示区别

TARGET_YEAR = 2025
METHOD = "newman"
S3_ALPHA = 0.1
MIN_PAPER_COUNT = 2


# =========================
# 2. 核心函数 (保持不变)
# =========================
def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()
    return U


def build_network_matrix(method, author_ids, paper_ids, df_idx):
    n_auth = len(author_ids)
    n_paper = len(paper_ids)
    data = np.ones(len(df_idx), dtype=np.int8)
    rows = df_idx["ai"].values
    cols = df_idx["pk"].values
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_auth, n_paper))

    if method == "newman":
        paper_degrees = np.array(A.sum(axis=0)).ravel()
        w = np.zeros_like(paper_degrees, dtype=float)
        mask = paper_degrees > 1
        w[mask] = 1.0 / (paper_degrees[mask] - 1.0)
        W = sp.diags(w)
        U = A @ W @ A.T
    else:
        U = A @ A.T
    return zero_diag_csr_inplace(U)


def sparse_to_nx(U, author_ids):
    G = nx.Graph()
    G.add_nodes_from(author_ids)
    coo = U.tocoo()
    mask = coo.row < coo.col
    edges = zip([author_ids[r] for r in coo.row[mask]],
                [author_ids[c] for c in coo.col[mask]],
                coo.data[mask])
    G.add_weighted_edges_from(edges, weight='weight')
    return G


# --- 过滤器 ---
def filter_isolates_s1(G):
    if len(G) == 0: return G
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def filter_lcc_s2(G):
    if len(G) == 0: return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def filter_min_papers_s5(G, paper_counts, min_count):
    if len(G) == 0: return G
    nodes_to_keep = [n for n in G.nodes() if paper_counts.get(n, 0) >= min_count]
    return G.subgraph(nodes_to_keep).copy()


def filter_disparity_s3(G, alpha=0.1):
    if len(G) == 0: return G
    strength = dict(G.degree(weight='weight'))
    degree = dict(G.degree())
    rem_edges = []
    for u, v, d in G.edges(data=True):
        k_u = degree[u]
        k_v = degree[v]
        if k_u > 1 and k_v > 1:
            w = d.get('weight', 0)
            alpha_u = (1 - w / strength[u]) ** (k_u - 1)
            alpha_v = (1 - w / strength[v]) ** (k_v - 1)
            if alpha_u >= alpha and alpha_v >= alpha:
                rem_edges.append((u, v))
    G_filtered = G.copy()
    G_filtered.remove_edges_from(rem_edges)
    return filter_isolates_s1(G_filtered)


def greedy_box_covering(G, lb):
    N = G.number_of_nodes()
    if N == 0: return 0
    if lb == 1: return N
    uncovered = set(G.nodes())
    count = 0
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    centers = [n for n, d in sorted_nodes]
    cutoff = lb - 1
    for center in centers:
        if center not in uncovered: continue
        count += 1
        neighbors = nx.single_source_shortest_path_length(G, center, cutoff=cutoff)
        for n in neighbors:
            if n in uncovered: uncovered.remove(n)
        if not uncovered: break
    return count


# =========================
# 3. 主程序 (修改了循环逻辑)
# =========================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f">>> [Step 1] Loading Data...")
    df = pd.read_csv(PATH_INPUT, dtype=str)

    y_cols = [c for c in df.columns if 'year' in c.lower() or 'PY' in c]
    if y_cols:
        y_col = y_cols[0]
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)
        df = df[df[y_col] <= TARGET_YEAR]
    df = df.dropna(subset=["author_id", "paper_id"])

    print(">>> [Step 1] Building Matrix...")
    auth_ids = df["author_id"].unique()
    paper_ids = df["paper_id"].unique()
    aid_map = {a: i for i, a in enumerate(auth_ids)}
    pid_map = {p: i for i, p in enumerate(paper_ids)}
    df_idx = pd.DataFrame({
        "ai": df["author_id"].map(aid_map),
        "pk": df["paper_id"].map(pid_map)
    })

    U = build_network_matrix(METHOD, auth_ids, paper_ids, df_idx)
    G_base = sparse_to_nx(U, auth_ids)
    paper_counts = df["author_id"].value_counts().to_dict()

    print(">>> [Step 1] Generating Sub-networks...")
    networks = {}
    networks["Overall"] = filter_lcc_s2(G_base)

    G_s5 = filter_min_papers_s5(G_base, paper_counts, MIN_PAPER_COUNT)
    networks["Active"] = filter_lcc_s2(G_s5)

    G_s3 = filter_disparity_s3(G_s5, alpha=S3_ALPHA)
    networks["Backbone"] = filter_lcc_s2(G_s3)

    print(f">>> [Step 1] Calculating Continuous Fractal Dimensions...")
    data_records = []

    for name, G in networks.items():
        print(f"    Processing {name} (Nodes: {len(G)})...")

        # 【修改核心】: 使用 while 循环，从 2 开始逐一增加，直到 nb <= 1
        lb = 2
        while True:
            nb = greedy_box_covering(G, lb)

            data_records.append({
                "Network": name,
                "BoxSize": lb,
                "MinBoxes": nb
            })

            # 调试输出，防止看着像卡死了
            if lb % 5 == 0:
                print(f"       -> lb={lb}, nb={nb}")

            if nb <= 1:
                print(f"       -> Finished at lb={lb}")
                break

            lb += 1  # 步长为1，连续计算

    out_df = pd.DataFrame(data_records)
    save_path = os.path.join(OUTPUT_DIR, DATA_FILENAME)
    out_df.to_csv(save_path, index=False)
    print(f"\n>>> [Success] Data saved to: {save_path}")


if __name__ == "__main__":
    main()
