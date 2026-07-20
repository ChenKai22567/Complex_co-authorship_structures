# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
fractal_analysis_final_inset.py

【程序功能】
科研合作网络分形特征全维度分析工具 (Ultimate Version with Inset)。
生成论文 4.2.2 节所需的完整证据链：

1. [图 a] 盒覆盖维数 (Fractal Dimension): 验证宏观分形标度。
2. [图 b] MST 骨架分支率 (Skeleton Branching): 验证微观层级生长机制 (Kim et al. 2004)。
3. [图 c] 度-度相关性 (Hub Correlation): 验证"富人俱乐部"捷径机制。
4. [图 e] 重整化群流 (RG Flow): 验证分形结构的物理稳定性 (Rozenfeld et al. 2010)。
   * [内嵌图 d] 自相似性验证 (Self-Similarity): 展示原始网络与重整化网络(lB=3)的度分布重合。

"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import defaultdict
import warnings
import random

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置参数 (User Configuration)
# ==========================================

# 输入数据路径 (请修改为您本地的真实路径)
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")

# 截止年份
TARGET_YEAR = 2025

# 活跃作者筛选阈值 (S5 标准)
MIN_PAPER_COUNT = 2

# Kim骨架提取的近似程度
# None = 精确计算 (极慢); 1000 = 近似计算 (快)
APPROX_K = 1000

# 尺度序列
BOX_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]


# ==========================================
# 2. 基础算法模块
# ==========================================

def greedy_box_covering(G, lb):
    """贪婪盒覆盖算法 (Song et al.)"""
    N = G.number_of_nodes()
    if N == 0: return 0, {}
    if lb == 1: return N, {n: i for i, n in enumerate(G.nodes())}

    uncovered = set(G.nodes())
    count = 0
    node_to_box = {}

    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    centers = [n for n, d in sorted_nodes]
    cutoff = lb - 1

    for center in centers:
        if center not in uncovered: continue
        current_box_id = count
        count += 1
        neighbors = nx.single_source_shortest_path_length(G, center, cutoff=cutoff)
        for n in neighbors:
            if n in uncovered:
                uncovered.remove(n)
                node_to_box[n] = current_box_id
        if not uncovered: break

    return count, node_to_box


def get_degree_dist_prob(G):
    """计算度分布概率 P(k)"""
    degs = [d for n, d in G.degree()]
    if not degs: return [], []
    counts = defaultdict(int)
    for d in degs: counts[d] += 1
    total = len(degs)
    x = sorted(counts.keys())
    y = [counts[k] / total for k in x]
    return x, y


# ==========================================
# 3. 分析模块 A: 分形维数
# ==========================================

def analyze_fractal_dimension(G):
    print("  [Step 1/5] Calculating Fractal Dimension...")
    lb_vals = []
    nb_vals = []
    for lb in BOX_SIZES:
        nb, _ = greedy_box_covering(G, lb)
        if nb <= 1: break
        lb_vals.append(lb)
        nb_vals.append(nb)

    if len(lb_vals) < 3: return None
    slope, intercept, r_val, _, _ = linregress(np.log(lb_vals), np.log(nb_vals))
    return {
        'dB': -slope, 'R2': r_val ** 2,
        'x': lb_vals, 'y': nb_vals,
        'fit_y': np.exp(intercept) * (np.array(lb_vals, dtype=float) ** slope)
    }


# ==========================================
# 4. 分析模块 B: Kim骨架分支率
# ==========================================

def extract_kim_skeleton(G, k_approx=None):
    print(f"    -> Extracting Skeleton (Kim et al.)... Mode: {'Approx' if k_approx else 'Exact'}")
    if k_approx and len(G) > k_approx:
        ebc = nx.edge_betweenness_centrality(G, k=k_approx, normalized=True)
    else:
        ebc = nx.edge_betweenness_centrality(G, normalized=True)

    G_weighted = G.copy()
    for u, v in G.edges():
        w = ebc.get((u, v), ebc.get((v, u), 0))
        G_weighted[u][v]['weight'] = w

    try:
        skeleton = nx.maximum_spanning_tree(G_weighted, weight='weight')
    except:
        # 兼容旧版
        for u, v, d in G_weighted.edges(data=True): d['weight'] = -d['weight']
        skeleton = nx.minimum_spanning_tree(G_weighted, weight='weight')
    return skeleton


def analyze_skeleton_branching(G):
    print("  [Step 2/5] Analyzing Skeleton Structure...")
    if len(G) < 10: return None
    skeleton = extract_kim_skeleton(G, k_approx=APPROX_K)

    # 找根节点
    root = max(dict(G.degree()).items(), key=lambda x: x[1])[0]
    if root not in skeleton:
        root = max(dict(skeleton.degree()).items(), key=lambda x: x[1])[0]

    try:
        lengths = nx.single_source_shortest_path_length(skeleton, root)
    except:
        return None

    dist_map = defaultdict(list)
    skel_deg = dict(skeleton.degree())
    for node, dist in lengths.items():
        if node == root:
            b = skel_deg[node]
        else:
            b = max(0, skel_deg[node] - 1)
        dist_map[dist].append(b)

    x, y = [], []
    for d in range(max(dist_map.keys()) + 1):
        vals = dist_map.get(d, [])
        if vals:
            y.append(np.mean(vals))
            x.append(d)
    return {'x': x, 'y': y}


# ==========================================
# 5. 分析模块 C: Hub 相关性
# ==========================================

def analyze_hub_correlation(G):
    print("  [Step 3/5] Analyzing Hub Correlation...")
    if len(G) < 10: return None
    knn = nx.average_neighbor_degree(G)
    deg = dict(G.degree())
    x, y = [], []
    for n in G.nodes():
        if deg[n] > 0 and n in knn:
            x.append(deg[n])
            y.append(knn[n])

    if len(x) > 5:
        slope, _, _, _, _ = linregress(np.log(x), np.log(y))
    else:
        slope = 0.0

    df = pd.DataFrame({'k': x, 'knn': y})
    df = df[df['k'] <= 200]
    df_bin = df.groupby('k').mean().reset_index()
    return {'slope': slope, 'x': df_bin['k'], 'y': df_bin['knn']}


# ==========================================
# 6. 分析模块 E & Inset D: RG Flow + 自相似性
# ==========================================

def analyze_rg_flow_with_inset(G):
    print("  [Step 4/5] Analyzing RG Flow...")
    N0 = G.number_of_nodes()
    x_vals, z_vals = [], []

    # --- 1. 计算 RG Flow ---
    for lb in BOX_SIZES:
        nb, node_to_box = greedy_box_covering(G, lb)
        if nb <= 1: break

        super_edges = set()
        for u, v in G.edges():
            b1, b2 = node_to_box.get(u), node_to_box.get(v)
            if b1 is not None and b2 is not None and b1 != b2:
                if b1 > b2: b1, b2 = b2, b1
                super_edges.add((b1, b2))

        zb = 2 * len(super_edges) / nb
        xb = N0 / nb
        x_vals.append(xb)
        z_vals.append(zb)

    if len(x_vals) < 3: return None
    slope, intercept, r_val, _, _ = linregress(np.log(x_vals), np.log(z_vals))

    # --- 2. 计算自相似性 (Inset Data) ---
    print("  [Step 5/5] Calculating Renormalized Distribution (Inset)...")
    # 原始分布
    x0, y0 = get_degree_dist_prob(G)

    # 重整化分布 (取 lB=3)
    _, node_to_box_3 = greedy_box_covering(G, 3)
    Gb_3 = nx.Graph()
    edges_3 = set()
    for u, v in G.edges():
        b1, b2 = node_to_box_3.get(u), node_to_box_3.get(v)
        if b1 is not None and b2 is not None and b1 != b2:
            if b1 > b2: b1, b2 = b2, b1
            edges_3.add((b1, b2))
    Gb_3.add_edges_from(edges_3)
    x3, y3 = get_degree_dist_prob(Gb_3)

    return {
        'rg': {'lambda': slope, 'R2': r_val ** 2, 'x': x_vals, 'y': z_vals,
               'fit_y': np.exp(intercept) * (np.array(x_vals) ** slope)},
        'inset': {'x0': x0, 'y0': y0, 'x3': x3, 'y3': y3}
    }


# ==========================================
# 7. 数据加载
# ==========================================

def load_data(path, year):
    print(f">>> Loading Data: {path}")
    if not os.path.exists(path): return None
    df = pd.read_csv(path, dtype=str)
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
        print(f"    Loaded Active LCC: N={len(G_lcc)}, E={G_lcc.number_of_edges()}")
        return G_lcc
    return None


# ==========================================
# 8. 主绘图程序
# ==========================================

def main():
    G = load_data(PATH_INPUT, TARGET_YEAR)
    if G is None: return

    # 运行分析
    res_a = analyze_fractal_dimension(G)
    res_b = analyze_skeleton_branching(G)
    res_c = analyze_hub_correlation(G)
    res_e_all = analyze_rg_flow_with_inset(G)

    # 设置画布
    print("\n>>> Plotting Combined Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    color = '#1f77b4'

    # --- 图 (a): 分形维数 ---
    ax = axes[0, 0]
    if res_a:
        ax.loglog(res_a['x'], res_a['y'], 'o', color=color, alpha=0.6, ms=8)
        ax.loglog(res_a['x'], res_a['fit_y'], '-', color=color, lw=2,
                  label=f"$d_B={res_a['dB']:.2f}$ ($R^2={res_a['R2']:.2f}$)")
        ax.legend(fontsize=10)
    ax.set_title("(a) Fractal Dimension Analysis", fontsize=12)
    ax.set_xlabel(r"Box Size ($l_B$)", fontsize=11)
    ax.set_ylabel(r"Min Boxes ($N_B$)", fontsize=11)
    ax.grid(alpha=0.2)

    # --- 图 (b): MST 骨架分支率 ---
    ax = axes[1, 0]
    if res_b:
        ax.plot(res_b['x'], res_b['y'], 'o-', color=color, alpha=0.8, ms=5)
        ax.axhline(1.0, color='gray', ls='--', label='Critical (y=1)')
        ax.legend()
    ax.set_title("(b) MST Skeleton Branching (Kim et al.)", fontsize=12)
    ax.set_xlabel(r"Distance from Hub ($d$)", fontsize=11)
    ax.set_ylabel(r"Mean Branching Rate", fontsize=11)
    ax.grid(alpha=0.2)

    # --- 图 (c): Hub 相关性 ---
    ax = axes[0, 1]
    if res_c:
        ax.loglog(res_c['x'], res_c['y'], '.-', color=color, alpha=0.6,
                  label=f"Slope = {res_c['slope']:.2f}")
        ax.text(0.05, 0.85, "Rich-Club / Assortative", transform=ax.transAxes,
                bbox=dict(fc='white', alpha=0.8))
        ax.legend()
    ax.set_title("(c) Hub Correlation (Rich-Club)", fontsize=12)
    ax.set_xlabel(r"Degree ($k$)", fontsize=11)
    ax.set_ylabel(r"Avg. Neighbor Degree ($k_{nn}$)", fontsize=11)
    ax.grid(alpha=0.2)

    # --- 图 (e): RG Flow (带嵌入图) ---
    ax = axes[1, 1]
    if res_e_all:
        rg = res_e_all['rg']
        # 主图: RG Flow
        ax.loglog(rg['x'], rg['y'], 'o', color=color, alpha=0.8, ms=8)
        ax.loglog(rg['x'], rg['fit_y'], 'k--', lw=2,
                  label=fr"$\lambda={rg['lambda']:.2f}$")
        status = "Unstable (Small-World)" if rg['lambda'] > 0 else "Stable (Fractal)"
        ax.text(0.05, 0.1, f"Phase: {status}", transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))
        ax.legend(loc='lower left')
        ax.set_title("(e) RG Flow Analysis & Self-Similarity", fontsize=12)
        ax.set_xlabel(r"Renormalization Factor ($x_b$)", fontsize=11)
        ax.set_ylabel(r"Renormalized Degree ($z_b$)", fontsize=11)
        ax.grid(alpha=0.2)

        # --- 嵌入图 (Inset): 自相似性 ---
        # 在右上角创建嵌入轴 [left, bottom, width, height] (相对于父坐标轴)
        ax_inset = ax.inset_axes([0.52, 0.52, 0.45, 0.45])
        ins = res_e_all['inset']
        ax_inset.loglog(ins['x0'], ins['y0'], '-', color='gray', lw=1, alpha=0.5, label='Original')
        ax_inset.loglog(ins['x3'], ins['y3'], 'o', color=color, ms=4, alpha=0.8, label='$l_B=3$')

        # 嵌入图设置
        ax_inset.set_xlabel(r"$k$", fontsize=8)
        ax_inset.set_ylabel(r"$P(k)$", fontsize=8)
        ax_inset.set_title("(d) Inset: Self-Similarity", fontsize=9)
        # 去掉嵌入图的网格以防太乱，或者设得很淡
        ax_inset.grid(False)
        ax_inset.tick_params(axis='both', which='major', labelsize=7)
        # 简单的图例
        # ax_inset.legend(fontsize=7, loc='upper right', frameon=False)

    plt.suptitle("Fractal & Topological Analysis: Robust Assortative Structure", fontsize=14, y=0.98)
    print(">>> Done. Displaying plot.")
    plt.show()


if __name__ == "__main__":
    main()
