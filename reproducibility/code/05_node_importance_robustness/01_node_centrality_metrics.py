# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
compute_centralities_exact_timed_v2.py

读取 GEXF 合著网络（无向有权），计算中心性/结构洞指标（尽量 exact）并统计运行时间。

计算指标（10项）：
1) degree_centrality
2) strength (weighted degree)
3) constraint_w (structural holes)
4) effective_size_w (structural holes)
5) katz_w (weighted)
6) leaderrank_w (weighted)
7) betweenness (unweighted, exact)
8) betweenness_w_exact (weighted, exact; uses dist)
9) closeness (unweighted)
10) closeness_w_exact (weighted; uses dist)

输出：
- node_centrality_metrics.csv
- node_centrality_timing.csv

依赖：
pip install networkx pandas numpy scipy
"""

import os
import time
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import networkx as nx

try:
    from scipy.sparse.linalg import eigsh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =======================
# 1) 路径配置（改这里）
# =======================
GEXF_PATH = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025.gexf")
OUT_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "node_centrality_metrics")

# =======================
# 2) 算法/收敛参数
# =======================
# Katz
KATZ_TOL = 1e-8
KATZ_MAXITER = 5000

# LeaderRank
LR_TOL = 1e-12
LR_MAXITER = 5000

# Strong-to-distance transform (for shortest-path centralities)
DIST_MODE = "inv_eps"   # "inv" / "inv_eps" / "inv1p"
DIST_EPS  = 1e-12       # only used in inv_eps

# =======================
# 3) 通用计时工具
# =======================
def timed(fn, *args, **kwargs) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =======================
# 4) 读取与预处理
# =======================
def load_graph_gexf(path: str) -> nx.Graph:
    G = nx.read_gexf(path)

    # MultiGraph -> Graph（合并平行边权重）
    if isinstance(G, nx.MultiGraph):
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0) or 1.0)
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        # 带上节点（GEXF 有时节点属性很多）
        for n, attrs in G.nodes(data=True):
            if n not in H:
                H.add_node(n, **attrs)
            else:
                H.nodes[n].update(attrs)
        G = H

    # 有向 -> 无向
    if G.is_directed():
        G = G.to_undirected(as_view=False)

    # 统一 weight 为 float，缺失则 1.0
    for u, v, d in G.edges(data=True):
        d["weight"] = float(d.get("weight", 1.0) or 1.0)

    # 确保无自环（一般没有，但防一下）
    loops = list(nx.selfloop_edges(G))
    if loops:
        G.remove_edges_from(loops)

    return G

def add_distance_attr(G: nx.Graph, weight_attr="weight", dist_attr="dist",
                      mode="inv_eps", eps=1e-12):
    """
    将“强度/相似度”权重转换为“距离/成本”，用于最短路中心性。
    NetworkX 最短路中心性把 weight 解释为距离/成本。为语义一致，需要转换。 参考：NetworkX 文档。

    mode:
      - "inv"      : dist = 1/weight（常见文献做法）
      - "inv_eps"  : dist = 1/(weight+eps)（防止极小/0）
      - "inv1p"    : dist = 1/(1+weight)（有界平滑）
    """
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight_attr, 1.0))
        if mode == "inv":
            if w <= 0:
                raise ValueError(f"Non-positive weight on edge ({u},{v}) = {w}, cannot use inv.")
            d[dist_attr] = 1.0 / w
        elif mode == "inv_eps":
            d[dist_attr] = 1.0 / (w + eps)
        elif mode == "inv1p":
            d[dist_attr] = 1.0 / (1.0 + w)
        else:
            raise ValueError("Unknown distance transform mode")

# =======================
# 5) LeaderRank（带权）实现
# =======================
def leaderrank_weighted(G: nx.Graph, weight_attr="weight",
                        tol=1e-12, max_iter=2000) -> Dict[Any, float]:
    """
    Weighted LeaderRank implementation:
    - Add a ground node connected to all nodes (weight=1)
    - Perform random walk power iteration using weighted transitions
    - Redistribute ground score evenly to all nodes
    """
    ground = "__GROUND__"
    H = G.copy()
    H.add_node(ground)

    # connect ground to all nodes
    for n in G.nodes():
        H.add_edge(ground, n, **{weight_attr: 1.0})

    nodes = list(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    x = np.ones(n, dtype=float)
    x[idx[ground]] = 0.0

    # out_strength for each node
    out_strength = np.zeros(n, dtype=float)
    for u in nodes:
        su = 0.0
        for _, v, d in H.edges(u, data=True):
            su += float(d.get(weight_attr, 1.0))
        out_strength[idx[u]] = su if su > 0 else 1.0

    # power iteration
    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=float)
        for u, v, d in H.edges(data=True):
            w = float(d.get(weight_attr, 1.0))
            iu, iv = idx[u], idx[v]
            # undirected: treat as two directed transitions
            x_new[iv] += x[iu] * (w / out_strength[iu])
            x_new[iu] += x[iv] * (w / out_strength[iv])

        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            break
        x = x_new

    gscore = x[idx[ground]]
    share = gscore / G.number_of_nodes()

    lr = {v: float(x[idx[v]] + share) for v in G.nodes()}
    return lr

# =======================
# 6) Katz alpha 自动选择（保证收敛）
# =======================
def choose_katz_alpha(G: nx.Graph, weight_attr="weight") -> float:
    """
    Katz requires alpha < 1/lambda_max for convergence.
    Use sparse largest eigenvalue if SciPy available; else conservative 1/deg_max.
    """
    if HAVE_SCIPY:
        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=weight_attr, dtype=float, format="csr")
        lam = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
        if lam <= 0:
            return 0.1
        return 0.9 / lam
    dmax = max(dict(G.degree()).values()) if G.number_of_nodes() else 1
    return 1.0 / max(dmax, 1)

# =======================
# 7) 主计算流程
# =======================
def main():
    ensure_dir(OUT_DIR)

    print("Loading graph:", GEXF_PATH)
    G, t_load = timed(load_graph_gexf, GEXF_PATH)
    print(f"Loaded: n={G.number_of_nodes()}, m={G.number_of_edges()}  (time={t_load:.3f}s)")

    print(f"Distance transform mode: {DIST_MODE} (eps={DIST_EPS})")
    add_distance_attr(G, "weight", "dist", mode=DIST_MODE, eps=DIST_EPS)

    res = pd.DataFrame(index=pd.Index(list(G.nodes()), name="author_id"))
    timings = []

    # 1) Degree centrality
    out, dt = timed(nx.degree_centrality, G)
    res["degree_centrality"] = pd.Series(out)
    timings.append(("degree_centrality", dt))

    # 2) Strength (weighted degree)
    out, dt = timed(lambda: dict(G.degree(weight="weight")))
    res["strength"] = pd.Series(out)
    timings.append(("strength", dt))

    # 3) Structural holes: constraint (weighted)
    out, dt = timed(nx.constraint, G, weight="weight")
    res["constraint_w"] = pd.Series(out)
    timings.append(("constraint_w", dt))

    # 4) Structural holes: effective_size (weighted)
    out, dt = timed(nx.effective_size, G, weight="weight")
    res["effective_size_w"] = pd.Series(out)
    timings.append(("effective_size_w", dt))

    # 5) Katz centrality (weighted)
    alpha = choose_katz_alpha(G, "weight")
    print(f"Katz alpha chosen = {alpha:.6g}")
    out, dt = timed(nx.katz_centrality, G,
                    alpha=alpha, beta=1.0, weight="weight",
                    tol=KATZ_TOL, max_iter=KATZ_MAXITER,
                    normalized=True)
    res["katz_w"] = pd.Series(out)
    timings.append(("katz_w", dt))

    # 6) LeaderRank (weighted)
    out, dt = timed(leaderrank_weighted, G, "weight", LR_TOL, LR_MAXITER)
    res["leaderrank_w"] = pd.Series(out)
    timings.append(("leaderrank_w", dt))

    # 7) Betweenness (unweighted, exact)
    out, dt = timed(nx.betweenness_centrality, G,
                    k=None, normalized=True, weight=None, endpoints=False)
    res["betweenness"] = pd.Series(out)
    timings.append(("betweenness_exact", dt))

    # 8) Betweenness (weighted by dist, exact)
    out, dt = timed(nx.betweenness_centrality, G,
                    k=None, normalized=True, weight="dist", endpoints=False)
    res["betweenness_w_exact"] = pd.Series(out)
    timings.append(("betweenness_w_exact(dist)", dt))

    # 9) Closeness (unweighted)
    out, dt = timed(nx.closeness_centrality, G, distance=None)
    res["closeness"] = pd.Series(out)
    timings.append(("closeness", dt))

    # 10) Closeness (weighted by dist)
    out, dt = timed(nx.closeness_centrality, G, distance="dist")
    res["closeness_w_exact"] = pd.Series(out)
    timings.append(("closeness_w_exact(dist)", dt))

    # 输出
    out_csv = os.path.join(OUT_DIR, "node_centrality_metrics.csv")
    res.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")

    timing_df = pd.DataFrame(timings, columns=["metric", "seconds"]).sort_values("seconds", ascending=False)
    timing_path = os.path.join(OUT_DIR, "node_centrality_timing.csv")
    timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")

    print("\nSaved centralities:", out_csv)
    print("Saved timings:", timing_path)
    print("\nTiming summary (top 10):")
    print(timing_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
