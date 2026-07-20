# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
add_node_closeness_curves.py

用途：
- 读取你的合著网络（gexf）
- 计算 Closeness / Weighted Closeness（精确版）
- 按照“重要 -> 不重要”的顺序移除节点
- 生成两条攻击曲线 CSV，并可直接合并入你现有绘图流程

输出列（与 robustness_results_v1 其它策略保持一致）：
removed_frac, lcc_ratio, num_components, eff_ratio, cap_in_lcc_ratio
"""

import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# =========================
# 0) 路径配置：按你的实际路径修改
# =========================
# 你的网络文件（LCC网络）
GEXF_PATH = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025.gexf")

# 你现有鲁棒性结果目录（robustness_results_v1）
OUT_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "closeness_attack_curves")
os.makedirs(OUT_DIR, exist_ok=True)

# 用来“对齐 removed_frac 步长”的参考曲线文件（优先读取你已算好的 node_degree_curve.csv）
REF_CURVE = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "exact_network_robustness" / "node_degree_curve_exact.csv")

# =========================
# 1) 指标口径设置（为了和你现有 v1 结果一致）
# =========================
# “容量”字段：用于 cap_in_lcc_ratio（优先 papers_count；没有则 total_cites；还没有则 1）
CAPACITY_ATTR_CANDIDATES = ["papers_count", "total_cites", "g_index"]

# 效率计算：为了与 v1 保持一致（通常 v1 用抽样近似）
EFF_MODE = "approx"       # "approx" 或 "exact"
EFF_SAMPLE_SOURCES = 400  # approx 模式下每个 step 抽样多少个源点（越大越准越慢）
EFF_RANDOM_SEED = 2026    # 保持可复现

# =========================
# 2) 小工具：计时器
# =========================
class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        print(f"[START] {self.name} ...")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        t1 = time.time()
        print(f"[END]   {self.name} | time={t1 - self.t0:.3f}s\n")

# =========================
# 3) 读取 removed_frac（与现有曲线对齐）
# =========================
def load_removed_frac_reference(n_nodes: int) -> np.ndarray:
    """
    优先读取你已生成的 node_degree_curve.csv 的 removed_frac 列，
    保证新增策略曲线与现有曲线 x 轴完全对齐。
    若找不到，则默认生成 0~1 的 101 个点（步长 0.01）。
    """
    if os.path.exists(REF_CURVE):
        df = pd.read_csv(REF_CURVE)
        if "removed_frac" in df.columns:
            x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).to_numpy()
            # 简单清洗：限制到 [0,1]
            x = np.clip(x, 0.0, 1.0)
            # 必须从0开始
            if x[0] != 0.0:
                x[0] = 0.0
            return x

    # fallback：默认 101 点
    return np.linspace(0.0, 1.0, 101)

# =========================
# 4) capacity（cap_in_lcc_ratio 用）
# =========================
def detect_capacity_attr(G: nx.Graph) -> str:
    """
    在图的节点属性中找一个最合适的“容量”字段。
    """
    for attr in CAPACITY_ATTR_CANDIDATES:
        ok = True
        for n in list(G.nodes())[:50]:
            if attr not in G.nodes[n]:
                ok = False
                break
        if ok:
            return attr
    return ""

def get_node_capacity(G: nx.Graph, attr: str) -> dict:
    """
    返回 node -> capacity（float）
    """
    if not attr:
        return {n: 1.0 for n in G.nodes()}
    cap = {}
    for n in G.nodes():
        v = G.nodes[n].get(attr, 0)
        try:
            cap[n] = float(v)
        except Exception:
            cap[n] = 0.0
    return cap

# =========================
# 5) efficiency：exact / approx
# =========================
def global_efficiency_exact_unweighted(G: nx.Graph) -> float:
    """
    NetworkX 的全局效率（无权图，精确）
    注意：对大图每一步都算会很慢，因此默认不用。
    """
    if G.number_of_nodes() <= 1:
        return 0.0
    from networkx.algorithms.efficiency_measures import global_efficiency
    return float(global_efficiency(G))

def global_efficiency_approx_unweighted(G: nx.Graph, sample_sources: int, seed: int) -> float:
    """
    用“抽样源点 + BFS”近似全局效率（无权图）。
    口径：平均 1/d_ij（只对可达对计入），归一到 [0,1] 附近。
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    nodes = list(G.nodes())
    rng = random.Random(seed)
    s = min(sample_sources, n)
    sources = rng.sample(nodes, s)

    denom = s * (n - 1)
    total = 0.0

    for u in sources:
        lengths = nx.single_source_shortest_path_length(G, u)
        # lengths 包含 u 自身 0
        for v, d in lengths.items():
            if v == u:
                continue
            if d > 0:
                total += 1.0 / float(d)

    return total / float(denom)

def compute_efficiency(G: nx.Graph, eff_mode: str, sample_sources: int, seed: int) -> float:
    if eff_mode == "exact":
        return global_efficiency_exact_unweighted(G)
    return global_efficiency_approx_unweighted(G, sample_sources, seed)

# =========================
# 6) 计算 closeness（精确）与加权 closeness（精确）
# =========================
def ensure_dist_on_edges(G: nx.Graph, weight_attr="weight", dist_attr="dist"):
    """
    将边权 weight 转为距离 dist = 1/(1+weight)，供 weighted closeness 使用。
    """
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr, 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        d[dist_attr] = 1.0 / (1.0 + w)

def compute_closeness_scores(G: nx.Graph, weighted: bool) -> dict:
    """
    计算 closeness centrality：
    - weighted=False：无权紧密中心性（精确）
    - weighted=True ：加权紧密中心性（精确，使用 dist 作为 edge length）
    返回：node -> score（越大越重要）
    """
    if G.number_of_nodes() == 0:
        return {}

    # 确保连通（closeness 在连通图上更合理）
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if not weighted:
        # NetworkX closeness_centrality：精确（对每个节点做 BFS）
        with Timer("Compute closeness (exact, unweighted)"):
            scores = nx.closeness_centrality(G)  # 精确
        return scores

    # weighted closeness：需要 dist
    ensure_dist_on_edges(G, weight_attr="weight", dist_attr="dist")
    with Timer("Compute closeness (exact, weighted by dist=1/(1+w))"):
        scores = nx.closeness_centrality(G, distance="dist")  # 精确 Dijkstra
    return scores

# =========================
# 7) 仿真：按给定 order 移除节点，记录曲线
# =========================
def simulate_node_attack_curve(
    G0: nx.Graph,
    removal_order: list,
    removed_frac: np.ndarray,
    capacity: dict,
    eff_mode: str,
    eff_sample_sources: int,
    eff_seed: int
) -> pd.DataFrame:
    """
    使用 removed_frac 对齐步长：
    - 第 k 个点要求累计移除 round(removed_frac[k]*N0) 个节点
    - 每一步按 removal_order 顺序移除
    """
    G = G0.copy()
    n0 = G0.number_of_nodes()

    # 初始效率与总容量
    with Timer("Compute baseline efficiency E0"):
        E0 = compute_efficiency(G, eff_mode, eff_sample_sources, eff_seed)

    total_cap0 = sum(capacity.get(n, 0.0) for n in G0.nodes())
    if total_cap0 <= 0:
        total_cap0 = 1.0

    rows = []
    removed_so_far = 0

    # 预先把 order 变成可索引数组
    order = [n for n in removal_order if n in G0]
    if len(order) < n0:
        # 理论上应包含全部节点；少了就补齐剩余（保持确定性）
        rest = [n for n in G0.nodes() if n not in set(order)]
        order.extend(rest)

    # 逐步移除
    for k in tqdm(range(len(removed_frac)), desc="Simulating closeness attack curve"):
        target_removed = int(round(float(removed_frac[k]) * n0))
        step_remove = max(0, target_removed - removed_so_far)

        if step_remove > 0 and G.number_of_nodes() > 0:
            # 取下一段要移除的节点
            to_remove = order[removed_so_far: removed_so_far + step_remove]
            G.remove_nodes_from(to_remove)
            removed_so_far += len(to_remove)

        # ---- 结构指标：LCC + 组件数 ----
        if G.number_of_nodes() == 0:
            lcc_ratio = 0.0
            num_components = 0
            cap_in_lcc_ratio = 0.0
            eff_ratio = 0.0
        else:
            comps = list(nx.connected_components(G))
            num_components = len(comps)
            lcc_nodes = max(comps, key=len)
            lcc_ratio = len(lcc_nodes) / float(n0)

            # ---- 功能指标：效率（与 v1 保持一致建议用 approx）----
            E = compute_efficiency(G, eff_mode, eff_sample_sources, eff_seed)
            eff_ratio = (E / E0) if E0 > 0 else 0.0

            # ---- LCC 内容量占比 ----
            cap_lcc = sum(capacity.get(n, 0.0) for n in lcc_nodes)
            cap_in_lcc_ratio = cap_lcc / float(total_cap0)

        rows.append({
            "removed_frac": float(removed_frac[k]),
            "lcc_ratio": float(lcc_ratio),
            "num_components": int(num_components),
            "eff_ratio": float(eff_ratio),
            "cap_in_lcc_ratio": float(cap_in_lcc_ratio),
        })

    return pd.DataFrame(rows)

# =========================
# 8) 主程序
# =========================
def main():
    with Timer("Load graph"):
        G0 = nx.read_gexf(GEXF_PATH)
        # 保证无向图
        if nx.is_directed(G0):
            G0 = G0.to_undirected()

        # 确保在 LCC 上做攻击仿真（与你现有结果一致）
        if not nx.is_connected(G0):
            largest_cc = max(nx.connected_components(G0), key=len)
            G0 = G0.subgraph(largest_cc).copy()

        print(f"[OK] Loaded graph (LCC): n={G0.number_of_nodes()}, m={G0.number_of_edges()}")

    # 读取对齐用 removed_frac
    removed_frac = load_removed_frac_reference(G0.number_of_nodes())
    print(f"[OK] removed_frac points = {len(removed_frac)} (aligned to existing curves if available)")

    # capacity 字段
    cap_attr = detect_capacity_attr(G0)
    print(f"[OK] capacity attribute used = {cap_attr or '1.0 (fallback)'}")
    capacity = get_node_capacity(G0, cap_attr)

    # ---- 1) closeness order（越大越重要，移除从大到小）----
    scores_close = compute_closeness_scores(G0, weighted=False)
    order_close = sorted(scores_close.keys(), key=lambda n: scores_close.get(n, 0.0), reverse=True)

    # ---- 2) weighted closeness order（越大越重要，移除从大到小）----
    scores_wclose = compute_closeness_scores(G0, weighted=True)
    order_wclose = sorted(scores_wclose.keys(), key=lambda n: scores_wclose.get(n, 0.0), reverse=True)

    # ---- 仿真：closeness ----
    with Timer("Simulate node attack: Closeness"):
        df_curve = simulate_node_attack_curve(
            G0=G0,
            removal_order=order_close,
            removed_frac=removed_frac,
            capacity=capacity,
            eff_mode=EFF_MODE,
            eff_sample_sources=EFF_SAMPLE_SOURCES,
            eff_seed=EFF_RANDOM_SEED
        )
    out1 = os.path.join(OUT_DIR, "node_closeness_curve.csv")
    df_curve.to_csv(out1, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out1}")

    # ---- 仿真：weighted closeness ----
    with Timer("Simulate node attack: Weighted Closeness"):
        df_curve2 = simulate_node_attack_curve(
            G0=G0,
            removal_order=order_wclose,
            removed_frac=removed_frac,
            capacity=capacity,
            eff_mode=EFF_MODE,
            eff_sample_sources=EFF_SAMPLE_SOURCES,
            eff_seed=EFF_RANDOM_SEED
        )
    out2 = os.path.join(OUT_DIR, "node_w_closeness_curve.csv")
    df_curve2.to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out2}")

    print("\n[OK] Done. Now update your plotting script to include the two new CSVs.")

if __name__ == "__main__":
    main()
