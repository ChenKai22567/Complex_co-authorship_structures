# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
viralrank_full_exact_and_corehd.py

1) ViralRank (Iannelli et al.) on FULL graph, UNWEIGHTED (weight=None), using sparse direct LU:
   - compute viralrank_v (distance-like; smaller -> more central)
   - compute viralrank_score = -viralrank_v (larger -> more central; matches GitHub output sign)
   - this is a direct/“exact” solve approach (within floating point), but computationally heavy.

2) CoreHD (Core High Degree) for LCC-minimization CNP heuristic:
   - remove k nodes
   - output removed order and LCC curve

Outputs:
- viralrank_full.csv
- corehd_removed_nodes.csv
- corehd_lcc_curve.csv
- timing_report.csv

Requirements:
pip install networkx numpy pandas scipy tqdm
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import splu

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# =========================
# 1) PATHS (edit here)
# =========================
GEXF_PATH = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025.gexf")
OUT_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "viralrank_corehd_metrics")

# =========================
# 2) ViralRank settings (UNWEIGHTED)
# =========================
INVERSE_TEMP = 1e-4          # lambda > 0
BLOCK_SIZE   = 32            # number of RHS columns solved per block (adjust for memory/speed)
LOG_EPS      = 1e-15         # for numerical stability in log/clip

# =========================
# 3) CoreHD settings (LCC minimization)
# =========================
COREHD_K = 1700              # delete budget
RECORD_LCC_EVERY_STEP = True
RECORD_INTERVAL = 10

# =========================
# timing helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tic():
    return time.perf_counter()

def toc(t0):
    return time.perf_counter() - t0

def lcc_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))

# =========================
# load graph (undirected; keep weights for CoreHD selection only)
# ViralRank computation uses weight=None by design.
# =========================
def load_graph_gexf(path: str) -> nx.Graph:
    G = nx.read_gexf(path)

    # MultiGraph -> Graph merge weights
    if isinstance(G, nx.MultiGraph):
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0) or 1.0)
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        for n, attrs in G.nodes(data=True):
            if n not in H:
                H.add_node(n, **attrs)
            else:
                H.nodes[n].update(attrs)
        G = H

    if G.is_directed():
        G = G.to_undirected(as_view=False)

    for u, v, d in G.edges(data=True):
        d["weight"] = float(d.get("weight", 1.0) or 1.0)

    loops = list(nx.selfloop_edges(G))
    if loops:
        G.remove_edges_from(loops)

    # Ensure connected (use LCC if needed)
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()

    return G

# =========================
# ViralRank (FULL, UNWEIGHTED, direct sparse LU)
# =========================
def viralrank_full_unweighted_exact(G: nx.Graph, inversetemp: float, block_size: int = 32) -> pd.DataFrame:
    """
    Compute ViralRank on FULL graph with unweighted adjacency (weight=None).
    Uses sparse direct LU factorization of M = I - qP, q = exp(-lambda),
    then solves for blocks of RHS to avoid storing NxN.

    Accumulates:
      row_sum[i] = sum_j D_ij
      col_sum[j] = sum_i D_ij

    viralrank_v     = (row_sum + col_sum)/N   (distance-like, smaller is better)
    viralrank_score = -viralrank_v           (score-like, larger is better)
    """
    assert nx.is_connected(G), "ViralRank requires connected graph."
    assert inversetemp > 0, "inversetemp must be > 0"

    nodes = list(G.nodes())
    n = len(nodes)
    node_index = pd.Index(nodes, name="author_id")

    # Unweighted adjacency (CSR)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, dtype=np.float64, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    if np.any(deg <= 0):
        raise ValueError("Graph has isolates; ViralRank requires connected graph with no isolates.")

    inv_deg = 1.0 / deg
    Dinv = sp.diags(inv_deg, offsets=0, format="csr")

    # Row-stochastic transition matrix P = D^-1 * A
    P = (Dinv @ A).tocsr()

    # q = exp(-lambda)
    q = float(np.exp(-inversetemp))

    # M = I - qP (CSC for splu)
    M = (sp.eye(n, format="csc", dtype=np.float64) - (q * P).tocsc())

    # Sparse LU factorization (direct)
    t_lu = tic()
    lu = splu(M)  # may be memory heavy; direct exact approach
    dt_lu = toc(t_lu)

    # Accumulators
    row_sum = np.zeros(n, dtype=np.float64)
    col_sum = np.zeros(n, dtype=np.float64)

    # Solve blocks of RHS: M X = B, where B has columns e_j
    # Then X = M^{-1} B gives columns of N = (I - qP)^{-1} for selected j.
    t_solve = tic()
    for start in tqdm(range(0, n, block_size), desc="ViralRank solving blocks"):
        end = min(start + block_size, n)
        b = end - start
        js = np.arange(start, end, dtype=int)

        # Build dense RHS block B (n x b) with identity columns at js
        B = np.zeros((n, b), dtype=np.float64)
        B[js, np.arange(b)] = 1.0

        # Solve for X (n x b)
        X = lu.solve(B)  # dense output

        # diag elements for these columns: N_jj = X[j, col]
        diag = X[js, np.arange(b)]
        diag = np.maximum(diag, LOG_EPS)

        # Z_ij = N_ij / N_jj (column normalization)
        Z = X / diag.reshape(1, -1)
        Z = np.clip(Z, LOG_EPS, None)

        D = -np.log(Z)

        # accumulate sums
        row_sum += D.sum(axis=1)
        col_sum[js] = D.sum(axis=0)

    dt_solve = toc(t_solve)

    viralrank_v = (row_sum + col_sum) / n
    viralrank_score = -viralrank_v

    out = pd.DataFrame({
        "author_id": node_index,
        "viralrank_v": viralrank_v,
        "viralrank_score": viralrank_score,
    })

    # Provide ranks to avoid sign confusion
    out["rank_by_v_asc"] = out["viralrank_v"].rank(method="min", ascending=True).astype(int)
    out["rank_by_score_desc"] = out["viralrank_score"].rank(method="min", ascending=False).astype(int)

    meta = pd.DataFrame([
        {"component": "viralrank_sparse_lu_factorization", "seconds": dt_lu},
        {"component": "viralrank_block_solves_and_accumulation", "seconds": dt_solve},
        {"component": "viralrank_total", "seconds": dt_lu + dt_solve},
    ])
    return out, meta

# =========================
# CoreHD (LCC minimization heuristic)
# =========================
def corehd_dismantle(G: nx.Graph, k_remove: int,
                     record_every_step: bool = True,
                     record_interval: int = 10):
    H = G.copy()
    n0 = H.number_of_nodes()
    removed = []
    curve = []

    def snapshot(step):
        s = lcc_size(H) if H.number_of_nodes() else 0
        curve.append({
            "step": step,
            "removed": len(removed),
            "lcc_size": s,
            "lcc_ratio": s / n0 if n0 > 0 else 0.0
        })

    snapshot(step=0)

    for step in tqdm(range(1, k_remove + 1), desc="CoreHD removing"):
        if H.number_of_nodes() == 0:
            break

        core = nx.k_core(H, k=2)
        if core.number_of_nodes() == 0:
            v = max(H.nodes(), key=lambda x: H.degree(x))
        else:
            v = max(core.nodes(), key=lambda x: core.degree(x))

        H.remove_node(v)
        removed.append(v)

        if record_every_step:
            snapshot(step)
        else:
            if step % record_interval == 0 or step == k_remove:
                snapshot(step)

    return removed, pd.DataFrame(curve)

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)
    timing_rows = []

    # ---- load graph ----
    t0 = tic()
    G = load_graph_gexf(GEXF_PATH)
    dt = toc(t0)
    timing_rows.append({"task": "load_graph_lcc", "seconds": dt})
    print(f"[OK] Loaded graph (LCC): n={G.number_of_nodes()}, m={G.number_of_edges()} | time={dt:.3f}s")

    # ---- ViralRank FULL (UNWEIGHTED) ----
    print("\n[START] ViralRank FULL (UNWEIGHTED, direct sparse LU). This is very resource-intensive.")
    t1 = tic()
    vr_df, vr_meta = viralrank_full_unweighted_exact(G, INVERSE_TEMP, block_size=BLOCK_SIZE)
    dt_vr = toc(t1)
    timing_rows.append({"task": "viralrank_full_total", "seconds": dt_vr})

    vr_path = os.path.join(OUT_DIR, "viralrank_full.csv")
    vr_df.to_csv(vr_path, index=False, encoding="utf-8-sig")
    print(f"[OK] ViralRank saved: {vr_path} | time={dt_vr:.3f}s")

    # print internal breakdown
    print("\nViralRank internal timing breakdown:")
    print(vr_meta.sort_values("seconds", ascending=False).to_string(index=False))

    # show top-20 (two consistent views)
    print("\nTop-20 by ViralRank v (ascending; smaller is more central):")
    print(vr_df.sort_values("viralrank_v", ascending=True).head(20)[["author_id", "viralrank_v", "viralrank_score"]])

    print("\nTop-20 by ViralRank score=-v (descending; larger is more central):")
    print(vr_df.sort_values("viralrank_score", ascending=False).head(20)[["author_id", "viralrank_v", "viralrank_score"]])

    # ---- CoreHD (CNP: minimize LCC) ----
    print(f"\n[START] CoreHD on FULL graph (k={COREHD_K}) ...")
    t2 = tic()
    removed, curve_df = corehd_dismantle(G, COREHD_K, RECORD_LCC_EVERY_STEP, RECORD_INTERVAL)
    dt_core = toc(t2)
    timing_rows.append({"task": "corehd_full", "seconds": dt_core})

    rem_path = os.path.join(OUT_DIR, "corehd_removed_nodes.csv")
    pd.DataFrame({"rank": range(1, len(removed) + 1), "author_id": removed}).to_csv(
        rem_path, index=False, encoding="utf-8-sig"
    )
    curve_path = os.path.join(OUT_DIR, "corehd_lcc_curve.csv")
    curve_df.to_csv(curve_path, index=False, encoding="utf-8-sig")

    print(f"[OK] CoreHD saved removed list: {rem_path}")
    print(f"[OK] CoreHD saved LCC curve:    {curve_path}")
    print(f"[OK] CoreHD time={dt_core:.3f}s | removed={len(removed)} | final LCC ratio={float(curve_df.tail(1)['lcc_ratio'].iloc[0]) if not curve_df.empty else None}")

    # ---- timing report ----
    timing_df = pd.DataFrame(timing_rows)
    timing_path = os.path.join(OUT_DIR, "timing_report.csv")
    timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Timing report saved: {timing_path}")
    print("\nOverall timing summary:")
    print(timing_df.sort_values("seconds", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
