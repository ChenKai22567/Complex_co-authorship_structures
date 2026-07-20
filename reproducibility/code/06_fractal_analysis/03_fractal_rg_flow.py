# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
step1_calc_rg_flow.py

【程序功能】
RG Flow 分析计算脚本。
1. 加载网络并提取 LCC。
2. 执行全尺度重整化，计算平均度变化流。
3. 提取原始度序列用于后续分箱绘图。
4. 保存所有数据到 .npz。

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
# 【请修改为您本地的真实路径】
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")

# 输出目录
OUTPUT_DIR = str(_OUTPUT_ROOT / "06_fractal_analysis" / "fractal_rg_flow")
DATA_FILENAME = "rg_flow_data.npz"

TARGET_YEAR = 2025
MIN_PAPER_COUNT = 2
# 动态尺度: 从 2 开始，自动算到网络塌缩
BOX_SIZES = list(range(2, 100))


# ==========================================
# 2. 数据加载
# ==========================================
def load_data_lcc(path, year):
    print(f">>> Loading Data: {path}")
    if not os.path.exists(path):
        print("Error: File not found!")
        return None

    try:
        df = pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding='gbk')

    y_col = next((c for c in df.columns if 'year' in c.lower() or 'PY' in c), None)
    if y_col:
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int)
        df = df[df[y_col] <= year]

    G = nx.Graph()
    if "paper_id" not in df.columns or "author_id" not in df.columns:
        print("Error: CSV must contain 'paper_id' and 'author_id'")
        return None

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
    G_active = G.subgraph(active).copy()

    if len(G_active) > 0:
        G_lcc = G_active.subgraph(max(nx.connected_components(G_active), key=len)).copy()
        print(f"    Loaded LCC: N={len(G_lcc)}, E={G_lcc.number_of_edges()}")
        return G_lcc
    return None


# ==========================================
# 3. 严格算法模块
# ==========================================
def greedy_box_covering_strict(G, lb):
    """严格盒覆盖: 盒直径 < lb"""
    N = G.number_of_nodes()
    if N == 0: return 0, {}

    # 核心半径公式
    cutoff = (lb - 1) // 2

    uncovered = set(G.nodes())
    count = 0
    node_to_box = {}

    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    centers = [n for n, d in sorted_nodes]

    for center in centers:
        if center not in uncovered: continue
        current_box_id = count
        count += 1

        if cutoff == 0:
            neighbors = {center: 0}
        else:
            neighbors = nx.single_source_shortest_path_length(G, center, cutoff=cutoff)

        for n in neighbors:
            if n in uncovered:
                uncovered.remove(n)
                node_to_box[n] = current_box_id
        if not uncovered: break

    return count, node_to_box


def renormalize_network_strict(G, lb):
    """严格重整化: 包含孤立盒子"""
    nb, node_to_box = greedy_box_covering_strict(G, lb)
    G_b = nx.Graph()

    if nb > 0:
        G_b.add_nodes_from(range(nb))  # 显式添加所有盒子

    super_edges = set()
    for u, v in G.edges():
        b1 = node_to_box.get(u)
        b2 = node_to_box.get(v)
        if b1 is not None and b2 is not None and b1 != b2:
            if b1 > b2: b1, b2 = b2, b1
            super_edges.add((b1, b2))

    G_b.add_edges_from(super_edges)
    return G_b, nb


# ==========================================
# 4. 分析逻辑
# ==========================================
def analyze_rg_flow_smart(G):
    print(">>> Analyzing RG Flow (Smart Mode)...")
    N0 = G.number_of_nodes()
    z0 = 2 * G.number_of_edges() / N0
    print(f"    Original z0 = {z0:.4f}")

    raw_data = []

    for lb in BOX_SIZES:
        G_b, Nb = renormalize_network_strict(G, lb)

        # 终止条件
        if Nb <= 1:
            print(f"    lb={lb}: Network collapsed (Nb=1). Loop end.")
            break

        zb = 2 * G_b.number_of_edges() / Nb if Nb > 0 else 0
        xb = N0 / Nb
        diff = zb - z0

        raw_data.append({'lb': lb, 'xb': xb, 'zb': zb, 'diff': diff})
        print(f"    lb={lb}: Nb={Nb}, xb={xb:.2f}, zb={zb:.4f}")

    if not raw_data: return None

    # 判断相位
    # 稳定相特征: zb < z0 (diff < 0)
    neg_count = sum(1 for d in raw_data if d['diff'] < 0)
    is_stable = neg_count > (len(raw_data) / 2)

    x_fit = []
    y_fit = []

    if is_stable:
        print("    -> Phase: STABLE (Fractal). Fitting 'zb'.")
        mode = 'stable'
        # 稳定相: 拟合 zb (向下)
        for d in raw_data:
            if d['zb'] > 0:
                x_fit.append(d['xb'])
                y_fit.append(d['zb'])
    else:
        print("    -> Phase: UNSTABLE (Small-World). Fitting 'zb - z0'.")
        mode = 'unstable'
        # 不稳定相: 拟合 diff (向上)
        for d in raw_data:
            if d['diff'] > 1e-6:
                x_fit.append(d['xb'])
                y_fit.append(d['diff'])

    # 拟合 Lambda
    lambda_val = None
    r2_val = 0
    fit_line = []

    if len(x_fit) >= 3:
        slope, intercept, r_val, _, _ = linregress(np.log(x_fit), np.log(y_fit))
        lambda_val = slope
        r2_val = r_val ** 2
        fit_line = np.exp(intercept) * (np.array(x_fit) ** slope)
        print(f"    -> Slope (Lambda) = {slope:.2f}, R2 = {r2_val:.2f}")

    return {
        'mode': mode,
        'x': x_fit,
        'y': y_fit,
        'lambda': lambda_val,
        'R2': r2_val,
        'fit_line': fit_line,
        'z0': z0
    }


# ==========================================
# 5. 主程序
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载 LCC
    G = load_data_lcc(PATH_INPUT, TARGET_YEAR)
    if G is None: return

    # 2. 计算 RG Flow
    rg_res = analyze_rg_flow_smart(G)

    # 3. 提取 Raw Data 用于 Inset 分箱
    print(">>> Extracting Raw Degree Sequences (for Log-Binning)...")
    k0_raw = np.array([d for n, d in G.degree()])

    # 重整化网络 (lB=3) 的度序列
    Gb3, _ = renormalize_network_strict(G, 3)
    k3_raw = np.array([d for n, d in Gb3.degree()])

    # 4. 保存数据
    save_path = os.path.join(OUTPUT_DIR, DATA_FILENAME)
    print(f">>> Saving full data to {save_path}...")

    np.savez(save_path,
             mode=rg_res['mode'],
             rg_x=rg_res['x'],
             rg_y=rg_res['y'],
             rg_lambda=rg_res['lambda'] if rg_res['lambda'] is not None else np.nan,
             rg_R2=rg_res['R2'],
             rg_fit_line=rg_res['fit_line'],
             z0=rg_res['z0'],
             # 保存原始序列
             k0_raw=k0_raw,
             k3_raw=k3_raw
             )

    print(">>> Step 1 Complete. Please run 'step2_plot_rg_flow.py'.")


if __name__ == "__main__":
    main()
