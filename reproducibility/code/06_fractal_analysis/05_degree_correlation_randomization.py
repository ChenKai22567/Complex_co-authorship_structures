from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

import os
import numpy as np
import pandas as pd
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

# =========================
# 1. 配置 (Configuration)
# =========================
# 【修正】这里必须指向包含 author_id 和 paper_id 的原始数据！
PATH_RAW_DATA = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
OUTPUT_DIR = str(_OUTPUT_ROOT / "06_fractal_analysis" / "degree_correlation_randomization")
DATA_FILENAME = "figure_c_heatmap_data.npz"

TARGET_YEAR = 2025
MIN_PAPER_COUNT = 2
HEATMAP_BINS = 35
NUM_RANDOMIZATIONS = 10


# =========================
# 2. 数据加载函数
# =========================
def load_network(path, year):
    print(f">>> Loading Network: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None

    df = pd.read_csv(path, dtype=str)

    # 检查列名是否存在
    if "paper_id" not in df.columns or "author_id" not in df.columns:
        print("Error: CSV must contain 'paper_id' and 'author_id' columns!")
        print(f"Found columns: {df.columns.tolist()}")
        return None

    y_col = next((c for c in df.columns if 'year' in c.lower() or 'PY' in c), None)
    if y_col:
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)
        df = df[df[y_col] <= year]

    G = nx.Graph()
    groups = df.groupby("paper_id")["author_id"].apply(list)
    edges = []
    for authors in groups:
        authors = list(set(authors))
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    edges.append((authors[i], authors[j]))
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    counts = df["author_id"].value_counts().to_dict()
    active = [n for n in G.nodes() if counts.get(n, 0) >= MIN_PAPER_COUNT]
    G_active = G.subgraph(active)

    if len(G_active) > 0:
        G_lcc = G_active.subgraph(max(nx.connected_components(G_active), key=len)).copy()
        print(f"    Nodes: {len(G_lcc)}, Edges: {G_lcc.number_of_edges()}")
        return G_lcc
    return None


# =========================
# 3. 核心计算函数 (整数对齐分箱)
# =========================
def get_smart_bins(data_array, target_bins=30):
    """
    生成'智能'分箱边界：低度数强制整数对齐，高度数对数增长
    """
    min_val = max(1, data_array.min())
    max_val = data_array.max()

    # logspace 生成对数间隔
    raw_bins = np.logspace(np.log10(min_val), np.log10(max_val + 1), target_bins + 1)

    # 向下取整并去重 -> 保证整数边界
    bins = np.unique(np.floor(raw_bins))

    # 确保包含最大值
    if bins[-1] <= max_val:
        bins = np.append(bins, max_val + 1)

    return bins


def get_joint_prob_matrix(G, fixed_bins=None, target_bin_num=30):
    k1_list, k2_list = [], []
    deg = dict(G.degree())
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        k1_list.extend([du, dv])
        k2_list.extend([dv, du])

    k1 = np.array(k1_list)
    k2 = np.array(k2_list)

    if fixed_bins is None:
        bins_edges = get_smart_bins(k1, target_bins=target_bin_num)
    else:
        bins_edges = fixed_bins

    H, xedges, yedges = np.histogram2d(k1, k2, bins=bins_edges)

    total_edges = len(k1)
    P = H / total_edges if total_edges > 0 else H * 0
    return P, xedges, yedges


def calculate_averaged_ratio_profile(G, bins=30, n_random=10):
    print(">>> 1. Calculating P_real...")
    P_real, bin_edges, _ = get_joint_prob_matrix(G, target_bin_num=bins)

    print(f"    -> Generated {len(bin_edges) - 1} bins. Low bins: {bin_edges[:8]}")

    print(f">>> 2. Generating {n_random} Randomized Networks...")
    degree_seq = [d for n, d in G.degree()]
    P_rand_sum = np.zeros_like(P_real)

    for i in range(n_random):
        print(f"    -> Simulation {i + 1}/{n_random}...", end="\r")
        G_rand = nx.configuration_model(degree_seq, create_using=nx.Graph)
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        P_rand_i, _, _ = get_joint_prob_matrix(G_rand, fixed_bins=bin_edges)
        P_rand_sum += P_rand_i

    P_rand_avg = P_rand_sum / n_random

    print("\n>>> 3. Calculating Ratio R...")
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.divide(P_real, P_rand_avg)

    R[np.isnan(R)] = 0
    R[P_rand_avg == 0] = 0

    return R.T, bin_edges, bin_edges


# =========================
# 4. 主程序
# =========================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    G = load_network(PATH_RAW_DATA, TARGET_YEAR)
    if G is None: return

    R, xedges, yedges = calculate_averaged_ratio_profile(G, bins=HEATMAP_BINS, n_random=NUM_RANDOMIZATIONS)

    save_path = os.path.join(OUTPUT_DIR, DATA_FILENAME)
    print(f">>> Saving data to: {save_path}")
    np.savez(save_path, R=R, xedges=xedges, yedges=yedges)
    print(">>> Done. Now run Step 2 to plot.")


if __name__ == "__main__":
    main()
