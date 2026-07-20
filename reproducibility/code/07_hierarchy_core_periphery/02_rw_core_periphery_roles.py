# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
Random-walker core–periphery profiling (Della Rossa, Dercole, Piccardi 2013)
+ CommunityID role framework (z + P, weighted)
+ (NEW) Role-level summary table you requested

THIS VERSION:
- Output THREE layers: periphery / semi_core / core (from RW profile)
- Keep unweighted (binary) profiling for comparison (curve + coreness + optional layer agreement)
- Optional node filtering: keep only nodes in TOP-N largest communities (by CommunityID)
- Output role summary table:
    1) 7 roles node counts + proportion
    2) per-role counts in cp_layer: core / semi_core / periphery
    3) per-role mean: degree(unweighted), strength(weighted), g_index, Paper_Count
    4) per-role unique #communities

Dependencies:
  pip install pandas numpy scipy matplotlib
"""

# =======================
# 0) PARAMETERS (ALL HERE)
# =======================

# --- Input paths ---
EDGELIST_PATH = str(_OUTPUT_ROOT / "07_hierarchy_core_periphery" / "nested_weighted_dcsbm" / "hierarchy_filtered_edges.csv")
METRICS_PATH = str(_OUTPUT_ROOT / "04_community_analysis" / "community_metrics_2025" / "community_author_metrics_2025.csv")
METRICS_AUTHOR_COL = "AuthorID"
METRICS_COMM_COL   = "CommunityID"

# Metrics columns you want to summarize by role (must exist in metrics csv)
METRICS_PAPER_COL = "Paper_Count"
METRICS_GINDEX_COL = "g_index"

# --- Node filtering by community size ---
ENABLE_TOP_COMMUNITY_FILTER = True
TOP_COMMUNITIES_N = 135
MIN_COMMUNITY_SIZE = None
DROP_NODES_WITH_MISSING_COMMUNITY = True
DROP_ISOLATED_NODES_AFTER_FILTER = True  # induced subgraph + drop isolates

# --- Random-walker CP profiling parameters ---
RANDOM_SEED = 42
RW_SEED_CHOICE = "min_strength"   # "min_strength" (recommended) or "random"
RW_HEAP_EPS = 1e-12               # heap stale-key tolerance

# --- Compare with unweighted (binary) profile ---
PLOT_UNWEIGHTED_PROFILE = True

# --- 3-layer boundary detection on profile alpha_k ---
LAYER_SOURCE = "weighted"   # main cp_layer from: "weighted" or "unweighted"
LAYER_MODE = "fixed"        # "auto" or "fixed"

# Auto plateau + slope detection
PLATEAU_ALPHA_EPS = 1e-6
PLATEAU_CONSEC = 80
SMOOTH_WINDOW = 151
CORE_TAIL_MIN_FRAC = 0.60
CORE_TAIL_MAX_FRAC = 0.97
CORE_SLOPE_FRAC = 0.25

# Fixed fractions
PERIPH_FRACTION_FIXED = 0.35  # first 35% in peripheral->core order
CORE_FRACTION_FIXED   = 0.15  # last 15% in peripheral->core order

# --- Role framework (CommunityID as module) ---
Z_HUB = 2.5
P_ULTRA_PERIPHERAL = 0.05
P_PERIPHERAL = 0.62
P_CONNECTOR = 0.80
P_PROVINCIAL_HUB = 0.30
P_CONNECTOR_HUB = 0.75

# --- Outputs / plots ---
N_BINS = 300
LOG_HEATMAP = True
SHOW_FLIERS = False
LOG_BOXPLOT_Y = True
PROFILE_X_AS_FRACTION = True


# =======================
# 1) IMPORTS
# =======================
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz
import matplotlib.pyplot as plt


# =======================
# 2) HELPERS
# =======================
def read_and_collapse_edgelist(path: str) -> pd.DataFrame:
    """Read u,v,weight; clean; undirected; sum duplicates (u<v)."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    req = {"u", "v", "weight"}
    if not req.issubset(df.columns):
        raise ValueError(f"Edgelist must contain columns {req}, got {set(df.columns)}")

    df = df.dropna(subset=["u", "v", "weight"]).copy()
    df["u"] = df["u"].astype(str).str.strip()
    df["v"] = df["v"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).copy()
    df = df[df["weight"] > 0].copy()
    df = df[df["u"] != df["v"]].copy()

    uv = np.sort(df[["u", "v"]].to_numpy(), axis=1)
    df["a"] = uv[:, 0]
    df["b"] = uv[:, 1]
    df2 = df.groupby(["a", "b"], as_index=False)["weight"].sum()
    df2 = df2.rename(columns={"a": "u", "b": "v"})
    return df2


def load_metrics(metrics_path: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if METRICS_AUTHOR_COL not in df.columns or METRICS_COMM_COL not in df.columns:
        raise ValueError(f"Metrics file must contain {METRICS_AUTHOR_COL} and {METRICS_COMM_COL}")
    df = df.drop_duplicates(subset=[METRICS_AUTHOR_COL]).copy()
    df["author"] = df[METRICS_AUTHOR_COL].astype(str).str.strip()
    df["community"] = df[METRICS_COMM_COL].astype(str).str.strip()

    # ensure these columns are numeric if present
    for col in [METRICS_PAPER_COL, METRICS_GINDEX_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_sparse_adjacency_from_edges(df_uv_w: pd.DataFrame):
    """Build symmetric CSR adjacency and mapping from u,v,weight (u,v as strings)."""
    nodes = pd.Index(pd.unique(df_uv_w[["u", "v"]].values.ravel()))
    node2id = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    row = df_uv_w["u"].map(node2id).to_numpy(dtype=int)
    col = df_uv_w["v"].map(node2id).to_numpy(dtype=int)
    w = df_uv_w["weight"].to_numpy(dtype=float)

    A = coo_matrix((w, (row, col)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.eliminate_zeros()
    return A, nodes, node2id


def binarize_adjacency(A: csr_matrix) -> csr_matrix:
    A_bin = A.copy()
    A_bin.data = np.ones_like(A_bin.data)
    A_bin.eliminate_zeros()
    return A_bin


def get_top_communities(metrics_sub: pd.DataFrame, top_n: int, min_size=None):
    vc = metrics_sub["community"].value_counts()
    comm_size = vc.rename("n_nodes").reset_index().rename(columns={"index": "community"})
    kept = set()

    if top_n is not None and top_n > 0:
        kept.update(comm_size.head(top_n)["community"].tolist())

    if min_size is not None:
        kept.update(comm_size.loc[comm_size["n_nodes"] >= int(min_size), "community"].tolist())

    kept = list(kept)
    kept = sorted(kept, key=lambda c: (-int(vc.get(c, 0)), str(c)))
    return kept, comm_size


def compute_strengths_and_Pz(n, A: csr_matrix, comm_of_node):
    """
    Compute:
    - strength_total
    - strength_internal (same community)
    - strength_external
    - weighted participation coefficient P
    - within-community z-score based on internal strength
    """
    indptr, indices, data = A.indptr, A.indices, A.data
    strength_total = np.asarray(A.sum(axis=1)).ravel()
    strength_internal = np.zeros(n, dtype=float)
    comm_w = [dict() for _ in range(n)]

    for u in range(n):
        cu = comm_of_node[u]
        start, end = indptr[u], indptr[u + 1]
        for v, w in zip(indices[start:end], data[start:end]):
            cv = comm_of_node[v]
            if (cu is not None) and (cu == cv):
                strength_internal[u] += w
            if cv is not None:
                comm_w[u][cv] = comm_w[u].get(cv, 0.0) + w

    strength_external = strength_total - strength_internal
    strength_external[strength_external < 0] = 0.0

    P = np.zeros(n, dtype=float)
    for i in range(n):
        ki = strength_total[i]
        if ki <= 0:
            P[i] = 0.0
            continue
        s = 0.0
        for wsum in comm_w[i].values():
            frac = wsum / ki
            s += frac * frac
        P[i] = 1.0 - s

    z = np.zeros(n, dtype=float)
    comm_to_nodes = {}
    for i, c in enumerate(comm_of_node):
        if c is None:
            continue
        comm_to_nodes.setdefault(c, []).append(i)

    for c, idxs in comm_to_nodes.items():
        vals = strength_internal[idxs]
        mu = float(vals.mean()) if len(vals) else 0.0
        sd = float(vals.std(ddof=0)) if len(vals) else 0.0
        if sd <= 1e-12:
            z[idxs] = 0.0
        else:
            z[idxs] = (vals - mu) / sd

    return strength_total, strength_internal, strength_external, P, z


def assign_role(z, P):
    if z < Z_HUB:
        if P <= P_ULTRA_PERIPHERAL:
            return "ultra_peripheral"
        elif P <= P_PERIPHERAL:
            return "peripheral"
        elif P <= P_CONNECTOR:
            return "connector"
        else:
            return "kinless"
    else:
        if P <= P_PROVINCIAL_HUB:
            return "provincial_hub"
        elif P <= P_CONNECTOR_HUB:
            return "connector_hub"
        else:
            return "kinless_hub"


# -------- Random-walker CP profiling --------
def random_walker_core_periphery_profile(A: csr_matrix, seed_choice="min_strength", eps=1e-12, seed=42):
    """
    Greedy construction of periphery sets S by adding node v minimizing alpha(S ∪ {v}).

    For undirected weighted graphs:
      alpha(S) = 2 * W_in(S) / vol(S)
    where W_in(S) is total UNDIRECTED internal weight in S,
          vol(S) = sum_{i in S} strength(i).

    Returns:
      order_periphery: nodes added from most peripheral -> most core
      alpha_k: alpha after each step k (length n)
      rw_coreness: per-node coreness = alpha at the step it was added
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    strength = np.asarray(A.sum(axis=1)).ravel()

    if np.any(strength <= 0):
        raise RuntimeError("Graph contains isolated nodes (strength=0). Drop isolates before profiling.")

    if seed_choice == "min_strength":
        seed_node = int(np.argmin(strength))
    elif seed_choice == "random":
        seed_node = int(rng.integers(0, n))
    else:
        raise ValueError("seed_choice must be 'min_strength' or 'random'")

    indptr, indices, data = A.indptr, A.indices, A.data
    inS = np.zeros(n, dtype=bool)
    w_to_S = np.zeros(n, dtype=float)

    order = np.empty(n, dtype=int)
    alpha_k = np.zeros(n, dtype=float)

    inS[seed_node] = True
    order[0] = seed_node
    W_in = 0.0
    vol = float(strength[seed_node])
    alpha_k[0] = 0.0

    # update neighbors
    start, end = indptr[seed_node], indptr[seed_node + 1]
    for v, w in zip(indices[start:end], data[start:end]):
        if not inS[v]:
            w_to_S[v] += w

    # heap: (alpha_candidate, node)
    heap = []
    for v in range(n):
        if v == seed_node:
            continue
        a = 2.0 * (W_in + w_to_S[v]) / (vol + strength[v])
        heapq.heappush(heap, (a, v))

    for k in range(1, n):
        while True:
            a_old, v = heapq.heappop(heap)
            if inS[v]:
                continue
            a_new = 2.0 * (W_in + w_to_S[v]) / (vol + strength[v])
            if abs(a_new - a_old) <= eps:
                break
            heapq.heappush(heap, (a_new, v))

        inS[v] = True
        order[k] = v

        W_in += float(w_to_S[v])
        vol += float(strength[v])
        alpha_k[k] = 2.0 * W_in / vol

        start, end = indptr[v], indptr[v + 1]
        for u, wuv in zip(indices[start:end], data[start:end]):
            if not inS[u]:
                w_to_S[u] += wuv
                a_u = 2.0 * (W_in + w_to_S[u]) / (vol + strength[u])
                heapq.heappush(heap, (a_u, u))

    rw_coreness = np.zeros(n, dtype=float)
    for step, node in enumerate(order):
        rw_coreness[node] = alpha_k[step]

    return order, alpha_k, rw_coreness


# -------- 3-layer boundary detection --------
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return y.astype(float, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, max(3, len(y) - 1 if (len(y) % 2 == 0) else len(y)))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    ypad = np.pad(y.astype(float), (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def detect_boundaries_auto(alpha_k: np.ndarray,
                           plateau_eps: float,
                           plateau_consec: int,
                           smooth_window: int,
                           tail_min_frac: float,
                           tail_max_frac: float,
                           slope_frac: float):
    n = len(alpha_k)
    if n < 10:
        k1 = max(1, int(0.3 * n))
        k2 = max(k1 + 1, int(0.8 * n))
        return k1, k2

    sm = moving_average(alpha_k, smooth_window)

    consec = max(5, min(int(plateau_consec), n // 5))
    k1_rank = None
    for i in range(0, n - consec):
        if np.all(sm[i:i + consec] >= plateau_eps):
            k1_rank = i
            break
    if k1_rank is None:
        idx = np.argmax(sm >= plateau_eps)
        if sm[idx] >= plateau_eps:
            k1_rank = int(idx)
        else:
            k1_rank = int(0.35 * n)

    d1 = np.diff(sm)
    r0 = int(max(0, min(n - 3, tail_min_frac * n)))
    r1 = int(max(r0 + 2, min(n - 2, tail_max_frac * n)))
    tail_slice = d1[r0:r1] if r1 > r0 else d1
    max_slope = float(np.max(tail_slice)) if len(tail_slice) else float(np.max(d1))
    slope_thr = slope_frac * max_slope

    k2_rank = None
    start_i = max(int(k1_rank), r0)
    for i in range(start_i, min(len(d1), r1)):
        if d1[i] >= slope_thr:
            k2_rank = i + 1
            break

    if k2_rank is None:
        d2 = np.diff(d1)
        c0 = max(0, r0)
        c1 = min(len(d2), r1)
        if c1 <= c0:
            c0, c1 = int(0.6 * len(d2)), int(0.95 * len(d2))
        curv = np.abs(d2[c0:c1]) if c1 > c0 else np.abs(d2)
        if len(curv) > 0:
            j = int(np.argmax(curv)) + c0 + 1
            k2_rank = max(j, k1_rank + 1)
        else:
            k2_rank = max(k1_rank + 1, int(0.8 * n))

    k1_rank = int(np.clip(k1_rank, 1, n - 2))
    k2_rank = int(np.clip(k2_rank, k1_rank + 1, n - 1))
    return k1_rank, k2_rank


def boundaries_fixed(n: int, periph_frac: float, core_frac: float):
    periph_frac = float(np.clip(periph_frac, 0.01, 0.95))
    core_frac = float(np.clip(core_frac, 0.01, 0.95))
    k1 = int(np.floor(periph_frac * n))
    k2 = int(np.floor((1.0 - core_frac) * n))
    k1 = int(np.clip(k1, 1, n - 2))
    k2 = int(np.clip(k2, k1 + 1, n - 1))
    return k1, k2


def assign_layers_from_rank(rank: np.ndarray, k1_rank: int, k2_rank: int):
    layer = np.empty(rank.shape[0], dtype=object)
    layer[rank < k1_rank] = "periphery"
    layer[(rank >= k1_rank) & (rank < k2_rank)] = "semi_core"
    layer[rank >= k2_rank] = "core"
    return layer


# -------- plots --------
def save_profile_plot_compare(alpha_w, alpha_bin, out_png, out_csv,
                              k1_w=None, k2_w=None, k1_b=None, k2_b=None,
                              x_as_fraction=True):
    n = len(alpha_w)
    k = np.arange(1, n + 1)
    x = (k / n) if x_as_fraction else k

    df = pd.DataFrame({
        "k": k,
        "k_over_n": k / n,
        "alpha_weighted": alpha_w,
        "alpha_unweighted": alpha_bin if alpha_bin is not None else np.nan
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure()
    plt.plot(x, alpha_w, label="weighted")
    if alpha_bin is not None:
        plt.plot(x, alpha_bin, label="unweighted (binary)")

    if (k1_w is not None) and (k2_w is not None):
        xv1 = (k1_w / n) if x_as_fraction else k1_w
        xv2 = (k2_w / n) if x_as_fraction else k2_w
        plt.axvline(xv1, linewidth=1.2)
        plt.axvline(xv2, linewidth=1.2)

    if (k1_b is not None) and (k2_b is not None):
        xv1b = (k1_b / n) if x_as_fraction else k1_b
        xv2b = (k2_b / n) if x_as_fraction else k2_b
        plt.axvline(xv1b, linestyle="--", linewidth=1.2)
        plt.axvline(xv2b, linestyle="--", linewidth=1.2)

    plt.ylim(-0.02, 1.02)
    plt.xlabel("k / n" if x_as_fraction else "k")
    plt.ylabel("core–periphery profile α_k")
    plt.title("Random-walker core–periphery profile (weighted vs unweighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_binned_heatmap_with_two_lines(A: csr_matrix, idx_sorted, out_png, out_csv,
                                       n_bins=300, log1p=True, cut1=None, cut2=None):
    n = A.shape[0]
    pos = np.empty(n, dtype=int)
    pos[idx_sorted] = np.arange(n)

    bin_id = np.minimum((pos * n_bins) // max(1, n), n_bins - 1)
    B = np.zeros((n_bins, n_bins), dtype=float)

    indptr, indices, data = A.indptr, A.indices, A.data
    for u in range(n):
        bu = int(bin_id[u])
        start, end = indptr[u], indptr[u + 1]
        for v, w in zip(indices[start:end], data[start:end]):
            bv = int(bin_id[v])
            B[bu, bv] += w

    pd.DataFrame(B).to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 8))
    M = np.log1p(B) if log1p else B
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.title(f"Binned adjacency ({n_bins}x{n_bins})" + (" [log1p]" if log1p else ""))
    plt.xlabel("Bin (permuted order)")
    plt.ylabel("Bin (permuted order)")
    plt.colorbar()

    def _draw_cut(cut, linestyle="-"):
        if cut is None:
            return
        cut_bin = int(np.minimum((cut * n_bins) // max(1, n), n_bins - 1))
        plt.axhline(cut_bin - 0.5, linewidth=1.2, linestyle=linestyle)
        plt.axvline(cut_bin - 0.5, linewidth=1.2, linestyle=linestyle)

    _draw_cut(cut1, linestyle="-")    # core|semi_core
    _draw_cut(cut2, linestyle="--")   # semi_core|periphery

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -------- NEW: role summary table --------
def add_unweighted_degree_column(A: csr_matrix, node_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Add unweighted degree (binary degree = #neighbors) as 'degree_unweighted'.
    For symmetric CSR without self-loops, nnz-per-row equals degree.
    """
    deg = np.diff(A.indptr).astype(int)
    out = node_tbl.copy()
    out["degree_unweighted"] = deg
    return out


def _nunique_communities_excluding_na(series: pd.Series) -> int:
    s = series.astype(str)
    s = s.replace({"NA": np.nan, "nan": np.nan, "None": np.nan})
    return int(pd.Series(s).dropna().nunique())


def build_role_summary_table(node_tbl: pd.DataFrame,
                             comm_col: str = "community",
                             role_col: str = "role_label",
                             layer_col: str = "cp_layer",
                             degree_col: str = "degree_unweighted",
                             strength_col: str = "strength_total",
                             paper_col: str = METRICS_PAPER_COL,
                             gindex_col: str = METRICS_GINDEX_COL) -> pd.DataFrame:
    """
    Output table required by you:
      - role counts + proportion
      - role x cp_layer counts (periphery/semi_core/core)
      - per-role mean degree/strength/g_index/paper_count
      - per-role unique #communities
    """
    df = node_tbl.copy()

    for c in [role_col, layer_col, comm_col, degree_col, strength_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column in node_tbl: {c}")

    # ensure numeric
    df[degree_col] = pd.to_numeric(df[degree_col], errors="coerce")
    df[strength_col] = pd.to_numeric(df[strength_col], errors="coerce")
    if paper_col not in df.columns:
        df[paper_col] = np.nan
    if gindex_col not in df.columns:
        df[gindex_col] = np.nan
    df[paper_col] = pd.to_numeric(df[paper_col], errors="coerce")
    df[gindex_col] = pd.to_numeric(df[gindex_col], errors="coerce")

    # counts + proportion
    role_counts = df[role_col].value_counts(dropna=False)
    total_n = len(df)
    role_prop = role_counts / total_n

    # role x layer counts
    role_layer = pd.crosstab(df[role_col], df[layer_col])
    for col in ["periphery", "semi_core", "core"]:
        if col not in role_layer.columns:
            role_layer[col] = 0
    role_layer = role_layer[["periphery", "semi_core", "core"]]

    # means + unique communities
    means = df.groupby(role_col, dropna=False).agg(
        mean_degree=(degree_col, "mean"),
        mean_strength=(strength_col, "mean"),
        mean_g_index=(gindex_col, "mean"),
        mean_paper_count=(paper_col, "mean"),
        n_unique_communities=(comm_col, _nunique_communities_excluding_na),
    )

    summary = pd.DataFrame({"n_nodes": role_counts, "proportion": role_prop}) \
        .join(role_layer).join(means)

    # within-role layer fractions (helpful)
    denom = summary["n_nodes"].replace(0, np.nan)
    summary["periphery_frac_in_role"] = summary["periphery"] / denom
    summary["semi_core_frac_in_role"] = summary["semi_core"] / denom
    summary["core_frac_in_role"] = summary["core"] / denom

    # stable role order (if present)
    role_order = [
        "ultra_peripheral", "peripheral", "connector", "kinless",
        "provincial_hub", "connector_hub", "kinless_hub"
    ]
    existing = [r for r in role_order if r in summary.index]
    others = [r for r in summary.index if r not in existing]
    summary = summary.loc[existing + others]

    return summary.reset_index().rename(columns={"index": "role_label"})


# =======================
# 3) MAIN
# =======================
def main():
    np.random.seed(RANDOM_SEED)
    out_dir = _OUTPUT_ROOT / "07_hierarchy_core_periphery" / "rw_core_periphery_roles"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/13] Reading + collapsing edgelist ...")
    df_edges = read_and_collapse_edgelist(EDGELIST_PATH)
    print(f"  undirected unique edges: {len(df_edges):,}")

    print("[2/13] Loading metrics (CommunityID mapping) ...")
    metrics_df = load_metrics(METRICS_PATH)
    comm_map = dict(zip(metrics_df["author"], metrics_df["community"]))

    print("[3/13] Applying community-based node filtering (optional) ...")
    graph_nodes = pd.Index(pd.unique(df_edges[["u", "v"]].values.ravel()))
    msub = metrics_df[metrics_df["author"].isin(graph_nodes)].copy()

    if DROP_NODES_WITH_MISSING_COMMUNITY:
        msub = msub.dropna(subset=["community"])
        msub = msub[msub["community"].astype(str).str.lower() != "nan"].copy()

    if ENABLE_TOP_COMMUNITY_FILTER:
        kept_comms, comm_size = get_top_communities(
            metrics_sub=msub,
            top_n=TOP_COMMUNITIES_N,
            min_size=MIN_COMMUNITY_SIZE
        )
        kept_comms_set = set(kept_comms)

        comm_size.to_csv(out_dir / "community_sizes_in_graph.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"kept_communities": kept_comms}).to_csv(out_dir / "kept_communities.csv", index=False, encoding="utf-8-sig")
        print(f"  kept communities: {len(kept_comms)} (saved community_sizes_in_graph.csv & kept_communities.csv)")

        kept_authors = set(msub.loc[msub["community"].isin(kept_comms_set), "author"].tolist())

        before_e = len(df_edges)
        df_edges_f = df_edges[df_edges["u"].isin(kept_authors) & df_edges["v"].isin(kept_authors)].copy()
        after_e = len(df_edges_f)

        if DROP_ISOLATED_NODES_AFTER_FILTER:
            used = pd.Index(pd.unique(df_edges_f[["u", "v"]].values.ravel()))
            kept_authors = set(used.tolist())
            df_edges_f = df_edges_f[df_edges_f["u"].isin(kept_authors) & df_edges_f["v"].isin(kept_authors)].copy()

        print(f"  edges: {before_e:,} -> {after_e:,} after filtering")
        print(f"  nodes kept (by edges): {len(kept_authors):,}")
        df_edges = df_edges_f
    else:
        comm_size = msub["community"].value_counts().rename("n_nodes").reset_index().rename(columns={"index": "community"})
        comm_size.to_csv(out_dir / "community_sizes_in_graph.csv", index=False, encoding="utf-8-sig")
        print("  filtering disabled; use all nodes/edges.")

    if len(df_edges) == 0:
        raise RuntimeError("After filtering, no edges remain. Lower TOP_COMMUNITIES_N or MIN_COMMUNITY_SIZE.")

    print("[4/13] Building sparse adjacency (filtered graph) ...")
    A, nodes, node2id = build_sparse_adjacency_from_edges(df_edges)
    n = A.shape[0]
    print(f"  n_nodes={n:,}  n_edges(undirected)={len(df_edges):,}  nnz={A.nnz:,}")

    # drop isolated nodes if any remain
    strength = np.asarray(A.sum(axis=1)).ravel()
    iso = np.where(strength <= 0)[0]
    if len(iso) > 0:
        keep = np.setdiff1d(np.arange(n), iso)
        A = A[keep, :][:, keep].tocsr()
        nodes = nodes[keep]
        n = A.shape[0]
        print(f"  WARNING: removed isolates after build: {len(iso)} -> new n={n}")

    print("[5/13] Preparing community labels per node ...")
    comm_of_node = []
    missing = 0
    for a in nodes:
        c = comm_map.get(str(a), None)
        if c is None or (isinstance(c, float) and np.isnan(c)) or str(c).lower() == "nan":
            missing += 1
            comm_of_node.append(None)
        else:
            comm_of_node.append(str(c))
    print(f"  community missing in filtered graph: {missing:,}/{n:,}")

    metrics_df = metrics_df.drop_duplicates(subset=[METRICS_AUTHOR_COL]).copy()

    print("[6/13] Random-walker profiling on WEIGHTED network ...")
    order_w, alpha_w, coreness_w = random_walker_core_periphery_profile(
        A, seed_choice=RW_SEED_CHOICE, eps=RW_HEAP_EPS, seed=RANDOM_SEED
    )
    rank_w = np.empty(n, dtype=int)
    rank_w[order_w] = np.arange(n)

    alpha_b = None
    coreness_b = None
    rank_b = None
    layer_b = None
    k1_b = None
    k2_b = None

    print("[7/13] Random-walker profiling on UNWEIGHTED (binary) network ...")
    if PLOT_UNWEIGHTED_PROFILE:
        A_bin = binarize_adjacency(A)
        order_b, alpha_b, coreness_b = random_walker_core_periphery_profile(
            A_bin, seed_choice=RW_SEED_CHOICE, eps=RW_HEAP_EPS, seed=RANDOM_SEED
        )
        rank_b = np.empty(n, dtype=int)
        rank_b[order_b] = np.arange(n)
        print("  computed unweighted profile.")
    else:
        print("  unweighted profile disabled.")

    print("[8/13] Detecting 3-layer boundaries (weighted + optional unweighted) ...")
    if LAYER_MODE == "auto":
        k1_w, k2_w = detect_boundaries_auto(
            alpha_w,
            plateau_eps=PLATEAU_ALPHA_EPS,
            plateau_consec=PLATEAU_CONSEC,
            smooth_window=SMOOTH_WINDOW,
            tail_min_frac=CORE_TAIL_MIN_FRAC,
            tail_max_frac=CORE_TAIL_MAX_FRAC,
            slope_frac=CORE_SLOPE_FRAC
        )
        if alpha_b is not None:
            k1_b, k2_b = detect_boundaries_auto(
                alpha_b,
                plateau_eps=PLATEAU_ALPHA_EPS,
                plateau_consec=PLATEAU_CONSEC,
                smooth_window=SMOOTH_WINDOW,
                tail_min_frac=CORE_TAIL_MIN_FRAC,
                tail_max_frac=CORE_TAIL_MAX_FRAC,
                slope_frac=CORE_SLOPE_FRAC
            )
    elif LAYER_MODE == "fixed":
        k1_w, k2_w = boundaries_fixed(n, PERIPH_FRACTION_FIXED, CORE_FRACTION_FIXED)
        if alpha_b is not None:
            k1_b, k2_b = boundaries_fixed(n, PERIPH_FRACTION_FIXED, CORE_FRACTION_FIXED)
    else:
        raise ValueError("LAYER_MODE must be 'auto' or 'fixed'")

    layer_w = assign_layers_from_rank(rank_w, k1_w, k2_w)
    if rank_b is not None and (k1_b is not None) and (k2_b is not None):
        layer_b = assign_layers_from_rank(rank_b, k1_b, k2_b)

    if LAYER_SOURCE == "weighted":
        cp_layer = layer_w
        k1_main, k2_main = k1_w, k2_w
    elif LAYER_SOURCE == "unweighted":
        if layer_b is None:
            raise RuntimeError("LAYER_SOURCE='unweighted' but unweighted profile/layers are unavailable.")
        cp_layer = layer_b
        k1_main, k2_main = k1_b, k2_b
    else:
        raise ValueError("LAYER_SOURCE must be 'weighted' or 'unweighted'")

    cp_class = np.where(cp_layer == "core", "core", "noncore")

    boundary_summary = {
        "n_nodes": int(n),
        "LAYER_MODE": LAYER_MODE,
        "LAYER_SOURCE": LAYER_SOURCE,
        "k1_weighted_rank": int(k1_w),
        "k2_weighted_rank": int(k2_w),
        "periphery_frac_weighted": float(k1_w / n),
        "core_frac_weighted": float((n - k2_w) / n),
        "semi_core_frac_weighted": float((k2_w - k1_w) / n),
        "k1_unweighted_rank": int(k1_b) if k1_b is not None else None,
        "k2_unweighted_rank": int(k2_b) if k2_b is not None else None,
        "periphery_frac_unweighted": float(k1_b / n) if k1_b is not None else None,
        "core_frac_unweighted": float((n - k2_b) / n) if k2_b is not None else None,
        "semi_core_frac_unweighted": float((k2_b - k1_b) / n) if (k1_b is not None and k2_b is not None) else None,
    }
    pd.DataFrame([boundary_summary]).to_csv(out_dir / "RWCP_3layer_boundary_summary.csv", index=False, encoding="utf-8-sig")

    profile_png = out_dir / "RWCP_profile_alpha_k_weighted_vs_unweighted_with_boundaries.png"
    profile_csv = out_dir / "RWCP_profile_alpha_k_weighted_vs_unweighted.csv"
    save_profile_plot_compare(
        alpha_w, alpha_b,
        out_png=profile_png, out_csv=profile_csv,
        k1_w=k1_w, k2_w=k2_w,
        k1_b=k1_b, k2_b=k2_b,
        x_as_fraction=PROFILE_X_AS_FRACTION
    )
    print(f"  saved: {profile_png}")
    print(f"  saved: {profile_csv}")
    print(f"  saved: {out_dir / 'RWCP_3layer_boundary_summary.csv'}")

    print("[9/13] Computing strengths + participation P + within-community z ...")
    strength_total, strength_internal, strength_external, P, z = compute_strengths_and_Pz(n, A, comm_of_node)
    roles = [assign_role(float(z[i]), float(P[i])) for i in range(n)]

    print("[10/13] Building node table + merging metrics ...")
    node_tbl = pd.DataFrame({
        "author": nodes.astype(str),
        "node_id": np.arange(n, dtype=int),
        "community": [c if c is not None else "NA" for c in comm_of_node],

        "rw_coreness_weighted": coreness_w,
        "rw_rank_weighted": rank_w,
        "cp_layer_weighted": layer_w,

        "rw_coreness_unweighted": coreness_b if coreness_b is not None else np.nan,
        "rw_rank_unweighted": rank_b if rank_b is not None else np.nan,
        "cp_layer_unweighted": layer_b if layer_b is not None else np.nan,

        "cp_layer": cp_layer,
        "cp_class": cp_class,

        "strength_total": strength_total,
        "strength_internal": strength_internal,
        "strength_external": strength_external,
        "P_weighted": P,
        "z_internal_strength": z,
        "role_label": roles,
    })

    metrics_df["author"] = metrics_df[METRICS_AUTHOR_COL].astype(str).str.strip()
    node_tbl = node_tbl.merge(metrics_df, on="author", how="left", suffixes=("", "_metrics"))

    node_out = out_dir / "RWCP_3layer_node_roles_with_metrics.csv"
    node_tbl.to_csv(node_out, index=False, encoding="utf-8-sig")
    print(f"  saved: {node_out}")

    print("[11/13] Summary tables (3-layer × role, and optional weighted vs unweighted layer agreement) ...")
    ct = pd.crosstab(node_tbl["cp_layer"], node_tbl["role_label"])
    ct_out = out_dir / "RWCP_3layer_vs_role_counts.csv"
    ct.to_csv(ct_out, encoding="utf-8-sig")

    prop = ct.div(ct.sum(axis=1), axis=0)
    prop_out = out_dir / "RWCP_3layer_vs_role_proportions.csv"
    prop.to_csv(prop_out, encoding="utf-8-sig")

    if layer_b is not None:
        agree = pd.crosstab(node_tbl["cp_layer_weighted"], node_tbl["cp_layer_unweighted"])
        agree_out = out_dir / "RWCP_layer_agreement_weighted_vs_unweighted.csv"
        agree.to_csv(agree_out, encoding="utf-8-sig")
        print(f"  saved: {agree_out}")

    stats = node_tbl.groupby("cp_layer")[[
        "rw_coreness_weighted", "strength_total", "strength_internal", "strength_external", "P_weighted", "z_internal_strength"
    ]].agg(["mean", "median", "std", "count"])
    stats_out = out_dir / "RWCP_3layer_numeric_stats.csv"
    stats.to_csv(stats_out, encoding="utf-8-sig")

    print(f"  saved: {ct_out}")
    print(f"  saved: {prop_out}")
    print(f"  saved: {stats_out}")

    print("[12/13] Adjacency outputs + plots ...")
    tmp = node_tbl.copy()
    layer_order = {"core": 0, "semi_core": 1, "periphery": 2}
    tmp["layer_sort"] = tmp["cp_layer"].map(layer_order).astype(int)
    tmp["comm_sort"] = tmp["community"].astype(str)
    tmp = tmp.sort_values(
        by=["layer_sort", "rw_coreness_weighted", "comm_sort", "strength_total"],
        ascending=[True, False, True, False],
    )
    idx_sorted = tmp["node_id"].to_numpy(dtype=int)

    A_perm = A[idx_sorted, :][:, idx_sorted].tocsr()
    perm_npz = out_dir / "RWCP_3layer_adjacency_permuted.npz"
    save_npz(perm_npz, A_perm)
    print(f"  saved: {perm_npz}")

    n_core = int((node_tbl["cp_layer"] == "core").sum())
    n_semi = int((node_tbl["cp_layer"] == "semi_core").sum())
    cut_core_end = n_core
    cut_semi_end = n_core + n_semi

    binned_csv = out_dir / "RWCP_3layer_adjacency_binned_matrix.csv"
    binned_png = out_dir / "RWCP_3layer_adjacency_binned_heatmap.png"
    plot_binned_heatmap_with_two_lines(
        A, idx_sorted,
        out_png=binned_png, out_csv=binned_csv,
        n_bins=N_BINS, log1p=LOG_HEATMAP,
        cut1=cut_core_end,
        cut2=cut_semi_end
    )
    print(f"  saved: {binned_csv}")
    print(f"  saved: {binned_png}")

    plt.figure(figsize=(8, 5))
    groups = ["periphery", "semi_core", "core"]
    data = [node_tbl.loc[node_tbl["cp_layer"] == g, "strength_total"].to_numpy() for g in groups]
    plt.boxplot(data, tick_labels=groups, showfliers=SHOW_FLIERS)
    plt.title("Strength (total weighted degree) by CP layer")
    plt.ylabel("Strength")
    if LOG_BOXPLOT_Y:
        plt.yscale("log")
        plt.ylabel("Strength [log]")
    plt.tight_layout()
    box1 = out_dir / "RWCP_3layer_strength_boxplot_by_layer.png"
    plt.savefig(box1, dpi=220)
    plt.close()
    print(f"  saved: {box1}")

    role_counts = node_tbl["role_label"].value_counts()
    roles_order = role_counts.index.tolist()
    plt.figure(figsize=(10, 5))
    data2 = [node_tbl.loc[node_tbl["role_label"] == r, "strength_total"].to_numpy() for r in roles_order]
    plt.boxplot(data2, tick_labels=roles_order, showfliers=SHOW_FLIERS)
    plt.title("Strength (total) by role_label")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Strength")
    if LOG_BOXPLOT_Y:
        plt.yscale("log")
        plt.ylabel("Strength [log]")
    plt.tight_layout()
    box2 = out_dir / "RWCP_strength_boxplot_by_role.png"
    plt.savefig(box2, dpi=220)
    plt.close()
    print(f"  saved: {box2}")

    print("[13/13] NEW: Building 7-role summary table (counts/proportions + layer split + means + community coverage) ...")
    node_tbl2 = add_unweighted_degree_column(A, node_tbl)

    role_summary = build_role_summary_table(
        node_tbl2,
        comm_col="community",
        role_col="role_label",
        layer_col="cp_layer",
        degree_col="degree_unweighted",
        strength_col="strength_total",
        paper_col=METRICS_PAPER_COL,
        gindex_col=METRICS_GINDEX_COL
    )

    role_summary_out = out_dir / "RWCP_role_7class_summary.csv"
    role_summary.to_csv(role_summary_out, index=False, encoding="utf-8-sig")
    print(f"  saved: {role_summary_out}")

    summary = {
        "n_nodes": int(n),
        "n_edges_undirected": int(len(df_edges)),
        "unweighted_profile_enabled": int(PLOT_UNWEIGHTED_PROFILE),
        "LAYER_MODE": LAYER_MODE,
        "LAYER_SOURCE": LAYER_SOURCE,
        "k1_main_rank": int(k1_main),
        "k2_main_rank": int(k2_main),
        "periphery_size": int((node_tbl["cp_layer"] == "periphery").sum()),
        "semi_core_size": int((node_tbl["cp_layer"] == "semi_core").sum()),
        "core_size": int((node_tbl["cp_layer"] == "core").sum()),
        "missing_community_nodes": int(missing),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")

    print("\nDONE.")
    print(f"Outputs root: {out_dir}")
    print("Key outputs:")
    print(" - RWCP_3layer_node_roles_with_metrics.csv")
    print(" - RWCP_role_7class_summary.csv  (NEW)")
    print(" - RWCP_profile_alpha_k_weighted_vs_unweighted_with_boundaries.png (+ csv)")
    print(" - RWCP_3layer_boundary_summary.csv")
    print(" - RWCP_3layer_vs_role_counts.csv / proportions.csv")
    print(" - RWCP_3layer_adjacency_permuted.npz")
    print(" - RWCP_3layer_adjacency_binned_heatmap.png (+ csv)")
    print(" - RWCP_3layer_strength_boxplot_by_layer.png")
    print(" - RWCP_strength_boxplot_by_role.png")


if __name__ == "__main__":
    main()
