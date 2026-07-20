# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
KM_config global + strict qstest + strong cross-community profile
+ NEW: pair membership counts (n_nodes per pair) with permutation intervals
+ NEW: pair-level block heatmaps (pair x pair, and (pair,role) x (pair,role))
+ NEW: binned heatmap with pair boundary lines

Dependencies:
  pip install pandas numpy scipy networkx matplotlib cpnet
"""

# =======================
# 0) PARAMETERS (ALL HERE)
# =======================

# --- Input paths ---
EDGELIST_PATH = str(_OUTPUT_ROOT / "07_hierarchy_core_periphery" / "nested_weighted_dcsbm" / "hierarchy_filtered_edges.csv")

# Your community+metrics file (Infomap result + author indicators)
METRICS_PATH = str(_OUTPUT_ROOT / "04_community_analysis" / "community_metrics_2025" / "community_author_metrics_2025.csv")
METRICS_AUTHOR_COL = "AuthorID"
METRICS_COMM_COL = "CommunityID"

# --- KM_config ---
KM_NUM_RUNS = 50  # multi-starts for stability

# --- qstest (strict significance) ---
ALPHA = 0.01
NUM_RAND_NET = 5  # 你日志里是 10/10；更严格建议 200/500/1000（更慢）

# --- Post-qstest strict structural filters ---
USE_SIZE_FILTER = True
MIN_PAIR_SIZE = 10      # 你网络平均pair很小(≈5)，建议从5起
MIN_CORE_SIZE = 2

USE_COMM_COVERAGE_FILTER = True
MIN_VALID_COMM_FRAC = 0.80  # 你的 coverage=1.0，可保留该约束

# --- Cross-community profile ---
TOPK_COMM = 3
NORMALIZE_ENTROPY = True

# --- Visualization / adjacency outputs ---
N_BINS = 900
LOG_HEATMAP = True
LOG_BOXPLOT_Y = True
SHOW_FLIERS = False
RANDOM_SEED = 42

# --- NEW: block heatmap settings ---
# pair block heatmap 是否加入“OTHER”（非 kept pairs 的所有节点合并成一个额外组）
INCLUDE_OTHER_BLOCK = True

# 如果 kept_pairs=0，是否回退使用所有 significant pairs 来画 block heatmap
FALLBACK_TO_ALL_SIGNIFICANT_FOR_BLOCK = True


# =======================
# 1) IMPORTS
# =======================
import math
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, save_npz
import matplotlib.pyplot as plt

import cpnet


# =======================
# 2) HELPERS
# =======================
def read_and_collapse_edgelist(path: str) -> pd.DataFrame:
    """Read u,v,weight; clean; undirected; sum duplicates."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    required = {"u", "v", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Edgelist must contain columns {required}, got {set(df.columns)}")

    df = df.dropna(subset=["u", "v", "weight"]).copy()
    df["u"] = df["u"].astype(str).str.strip()
    df["v"] = df["v"].astype(str).str.strip()
    df["weight"] = df["weight"].astype(float)

    df = df[df["weight"] > 0].copy()
    df = df[df["u"] != df["v"]].copy()

    uv = np.sort(df[["u", "v"]].to_numpy(), axis=1)
    df["a"] = uv[:, 0]
    df["b"] = uv[:, 1]

    df2 = df.groupby(["a", "b"], as_index=False)["weight"].sum()
    df2 = df2.rename(columns={"a": "u", "b": "v"})
    return df2


def build_sparse_adjacency(df_uv_w: pd.DataFrame):
    """Build symmetric CSR adjacency and mapping."""
    nodes = pd.Index(pd.unique(df_uv_w[["u", "v"]].values.ravel()))
    node2id = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    row = df_uv_w["u"].map(node2id).to_numpy()
    col = df_uv_w["v"].map(node2id).to_numpy()
    w = df_uv_w["weight"].to_numpy(dtype=float)

    A = coo_matrix((w, (row, col)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.eliminate_zeros()
    return A, nodes, node2id, row, col, w


def build_nx_graph_from_edges(n: int, row: np.ndarray, col: np.ndarray, w: np.ndarray) -> nx.Graph:
    """qstest expects a NetworkX Graph internally."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = [(int(i), int(j), float(ww)) for i, j, ww in zip(row, col, w)]
    G.add_weighted_edges_from(edges, weight="weight")
    return G


def load_metrics(path: str, author_col: str, comm_col: str) -> pd.DataFrame:
    """Load your metrics file; standardize to columns: author, infomap_comm."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(p)
    if author_col not in df.columns or comm_col not in df.columns:
        raise ValueError(
            f"Cannot find required columns in {path}. "
            f"Need {author_col=} and {comm_col=}. Got: {df.columns.tolist()}"
        )

    df = df.copy()
    df["author"] = df[author_col].astype(str).str.strip()
    df["infomap_comm"] = df[comm_col].astype(str).str.strip()

    if "Total_Weighted_Degree" in df.columns and "Internal_Weighted_Degree" in df.columns:
        df["External_Weighted_Degree"] = df["Total_Weighted_Degree"] - df["Internal_Weighted_Degree"]
        df.loc[df["External_Weighted_Degree"] < 0, "External_Weighted_Degree"] = 0.0

    return df


def safe_pvalues_and_significance(p_values, significant):
    """Conservative fix: NaN/inf p-values -> p=1 and NOT significant."""
    p = np.array(p_values, dtype=float)
    sig = np.array(significant, dtype=bool)
    bad = ~np.isfinite(p)
    if bad.any():
        print(f"[qstest] WARNING: bad p-values (NaN/inf): {bad.sum()} / {len(p)} -> treat as NOT significant.")
        p[bad] = 1.0
        sig[bad] = False
    return p, sig


def shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def entropy_normalized(ent: float, k: int) -> float:
    if k <= 1:
        return 0.0
    return float(ent / np.log(k))


def topk_community_shares(vc: pd.Series, k: int):
    total = float(vc.sum())
    out = []
    for comm, cnt in vc.iloc[:k].items():
        out.append((str(comm), float(cnt) / total if total > 0 else 0.0))
    return out


def agg_metrics(sub: pd.DataFrame, metric_cols: list, prefix: str) -> dict:
    out = {}
    if len(sub) == 0:
        for c in metric_cols:
            out[f"{prefix}{c}_mean"] = np.nan
            out[f"{prefix}{c}_median"] = np.nan
        return out

    for c in metric_cols:
        if c not in sub.columns:
            out[f"{prefix}{c}_mean"] = np.nan
            out[f"{prefix}{c}_median"] = np.nan
            continue
        x = pd.to_numeric(sub[c], errors="coerce")
        out[f"{prefix}{c}_mean"] = float(x.mean())
        out[f"{prefix}{c}_median"] = float(x.median())
    return out


def plot_matrix_heatmap(M: np.ndarray, title: str, out_path: Path, log1p: bool = False,
                        draw_lines=None, figsize=(8, 7)):
    """
    draw_lines: list of ints (positions to draw grid lines) in matrix coordinates
    """
    plt.figure(figsize=figsize)
    X = np.log1p(M) if log1p else M
    plt.imshow(X, aspect="auto", interpolation="nearest")
    plt.title(title + (" [log1p]" if log1p else ""))
    plt.xlabel("Group index")
    plt.ylabel("Group index")
    plt.colorbar()

    if draw_lines:
        for p in draw_lines:
            plt.axhline(p - 0.5, linewidth=0.7)
            plt.axvline(p - 0.5, linewidth=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =======================
# 3) MAIN
# =======================
def main():
    np.random.seed(RANDOM_SEED)
    out_dir = _OUTPUT_ROOT / "07_hierarchy_core_periphery" / "km_core_periphery_block_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/9] Reading + collapsing edgelist ...")
    df_edges = read_and_collapse_edgelist(EDGELIST_PATH)
    print(f"  undirected unique edges: {len(df_edges):,}")

    print("[2/9] Building sparse adjacency ...")
    A, nodes, node2id, row_u, col_v, w = build_sparse_adjacency(df_edges)
    n = A.shape[0]
    print(f"  n_nodes={n:,}  n_edges(undirected)={len(df_edges):,}  nnz={A.nnz:,}")

    strength = np.asarray(A.sum(axis=1)).ravel()

    print("[3/9] Loading metrics (AuthorID/CommunityID + indicators) ...")
    metrics_df = load_metrics(METRICS_PATH, METRICS_AUTHOR_COL, METRICS_COMM_COL)

    exclude_cols = {METRICS_AUTHOR_COL, METRICS_COMM_COL, "author", "infomap_comm"}
    metric_cols = [c for c in metrics_df.columns if c not in exclude_cols]
    print(f"  metrics columns to aggregate: {metric_cols}")

    print("[4/9] Running KM_config (global) ...")
    alg = cpnet.KM_config(num_runs=KM_NUM_RUNS)
    alg.detect(A)

    pair_id_raw = alg.get_pair_id()      # dict: node -> pair
    coreness = alg.get_coreness()        # dict: node -> score

    # make pair ids contiguous 0..K-1
    unique_pids = sorted(set(pair_id_raw.values()))
    pid2new = {pid: i for i, pid in enumerate(unique_pids)}
    pair_id = {node: pid2new[pid] for node, pid in pair_id_raw.items()}
    K = len(unique_pids)
    print(f"  detected core-periphery pairs: {K:,}")

    print("[5/9] Strict significance filtering with qstest ...")
    G_nx = build_nx_graph_from_edges(n, row_u, col_v, w)

    sig_pair_id_dict, sig_coreness_dict, significant, p_values = cpnet.qstest(
        pair_id,
        coreness,
        G_nx,
        alg,
        significance_level=ALPHA,
        num_of_rand_net=NUM_RAND_NET,
    )

    p_values, significant = safe_pvalues_and_significance(p_values, significant)

    pair_sig = np.zeros(K, dtype=bool)
    pair_p = np.ones(K, dtype=float)
    L = min(K, len(significant), len(p_values))
    pair_sig[:L] = significant[:L]
    pair_p[:L] = p_values[:L]

    print(f"  significant pairs (qstest): {int(pair_sig.sum()):,} / {K:,}")

    print("[6/9] Building node_roles (merge metrics) + pair_summary ...")
    # Node roles (PAIR-level significance + KM coreness)
    records = []
    for i in range(n):
        pid = int(pair_id[i])
        x = float(coreness.get(i, np.nan))

        if pair_sig[pid]:
            sig_pid = pid
            sig_x = x
            role = "core" if sig_x >= 0.5 else "periphery"
        else:
            sig_pid = None
            sig_x = None
            role = "insignificant"

        records.append(
            {
                "author": nodes[i],
                "node_id": i,
                "pair_id": pid,
                "coreness": x,
                "sig_pair_id": sig_pid,
                "sig_coreness": sig_x,
                "role": role,
                "strength": float(strength[i]),
            }
        )

    node_roles = pd.DataFrame.from_records(records)

    # merge metrics
    metrics_unique = metrics_df.drop_duplicates(subset=["author"]).copy()
    node_roles = node_roles.merge(metrics_unique, on="author", how="left", suffixes=("", "_metrics"))

    comm_cov = float(node_roles["infomap_comm"].notna().mean())
    print(f"  community coverage in node_roles: {comm_cov:.3f}")

    # pair summary + keep_strict
    pair_rows = []
    for pid in range(K):
        sub = node_roles[node_roles["pair_id"] == pid]
        n_nodes = int(len(sub))
        n_core = int((sub["coreness"] >= 0.5).sum())
        n_peri = n_nodes - n_core
        valid_comm_frac = float(sub["infomap_comm"].notna().mean()) if n_nodes else 0.0

        pair_rows.append(
            {
                "pair_id": pid,
                "n_nodes": n_nodes,
                "p_value": float(pair_p[pid]),
                "significant": bool(pair_sig[pid]),
                "n_core": n_core if pair_sig[pid] else 0,
                "n_periphery": n_peri if pair_sig[pid] else 0,
                "valid_comm_frac": valid_comm_frac,
            }
        )

    pair_summary = pd.DataFrame(pair_rows)
    if USE_SIZE_FILTER:
        keep = (pair_summary["significant"] == True) & (pair_summary["n_nodes"] >= MIN_PAIR_SIZE) & (pair_summary["n_core"] >= MIN_CORE_SIZE)
    else:
        keep = (pair_summary["significant"] == True)

    if USE_COMM_COVERAGE_FILTER:
        keep = keep & (pair_summary["valid_comm_frac"] >= MIN_VALID_COMM_FRAC)

    pair_summary["keep_strict"] = keep
    pair_summary = pair_summary.sort_values(["keep_strict", "n_nodes"], ascending=[False, False]).reset_index(drop=True)

    kept_pairs = set(pair_summary.loc[pair_summary["keep_strict"], "pair_id"].tolist())
    print(f"  strict kept pairs: {len(kept_pairs):,} / {K:,}")

    # Save node roles + pair summary
    node_roles_path = out_dir / "KM_config_node_roles_with_metrics.csv"
    pair_summary_path = out_dir / "KM_config_pair_summary.csv"
    node_roles.to_csv(node_roles_path, index=False, encoding="utf-8-sig")
    pair_summary.to_csv(pair_summary_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {node_roles_path}")
    print(f"  saved: {pair_summary_path}")

    print("[7/9] Strong cross-community comparison (pair profile) ...")
    profile_pairs = sorted(kept_pairs)
    if len(profile_pairs) == 0 and FALLBACK_TO_ALL_SIGNIFICANT_FOR_BLOCK:
        profile_pairs = sorted(pair_summary.loc[pair_summary["significant"], "pair_id"].tolist())

    prof_rows = []
    for pid in profile_pairs:
        sub = node_roles[node_roles["pair_id"] == pid].copy()
        if len(sub) == 0:
            continue

        sub_core = sub[sub["coreness"] >= 0.5]
        sub_peri = sub[sub["coreness"] < 0.5]

        def comm_stats(comm_series: pd.Series):
            comm_series = comm_series.dropna().astype(str)
            if len(comm_series) == 0:
                return {"n_comm": 0, "purity": np.nan, "entropy": np.nan, "entropy_norm": np.nan, "topk": []}
            vc = comm_series.value_counts()
            counts = vc.to_numpy()
            k = int(len(vc))
            purity = float(counts.max() / counts.sum())
            ent = shannon_entropy(counts)
            entn = entropy_normalized(ent, k) if NORMALIZE_ENTROPY else ent
            topk = topk_community_shares(vc, TOPK_COMM)
            return {"n_comm": k, "purity": purity, "entropy": ent, "entropy_norm": entn, "topk": topk}

        s_all = comm_stats(sub["infomap_comm"])
        s_core = comm_stats(sub_core["infomap_comm"])
        s_peri = comm_stats(sub_peri["infomap_comm"])

        row = {
            "pair_id": pid,
            "p_value": float(pair_p[pid]),
            "significant": bool(pair_sig[pid]),
            "keep_strict": (pid in kept_pairs),

            "n_nodes": int(len(sub)),
            "n_core": int(len(sub_core)),
            "n_periphery": int(len(sub_peri)),
            "core_frac": float(len(sub_core) / len(sub)) if len(sub) else np.nan,

            "valid_comm_frac": float(sub["infomap_comm"].notna().mean()),

            "n_infomap_covered": s_all["n_comm"],
            "purity": s_all["purity"],
            "entropy": s_all["entropy"],
            "entropy_norm": s_all["entropy_norm"],

            "n_infomap_covered_core": s_core["n_comm"],
            "purity_core": s_core["purity"],
            "entropy_core": s_core["entropy"],
            "entropy_norm_core": s_core["entropy_norm"],

            "n_infomap_covered_periphery": s_peri["n_comm"],
            "purity_periphery": s_peri["purity"],
            "entropy_periphery": s_peri["entropy"],
            "entropy_norm_periphery": s_peri["entropy_norm"],
        }

        for t in range(TOPK_COMM):
            if t < len(s_all["topk"]):
                comm, frac = s_all["topk"][t]
                row[f"top{t+1}_comm"] = comm
                row[f"top{t+1}_frac"] = frac
            else:
                row[f"top{t+1}_comm"] = np.nan
                row[f"top{t+1}_frac"] = np.nan

        agg_cols = ["strength"] + metric_cols
        row.update(agg_metrics(sub, agg_cols, prefix="all_"))
        row.update(agg_metrics(sub_core, agg_cols, prefix="core_"))
        row.update(agg_metrics(sub_peri, agg_cols, prefix="peri_"))

        prof_rows.append(row)

    pair_profile = pd.DataFrame(prof_rows)
    if len(pair_profile) > 0:
        pair_profile = pair_profile.sort_values(["entropy_norm", "n_nodes"], ascending=[False, False]).reset_index(drop=True)

    pair_profile_path = out_dir / "KM_config_pair_cross_community_profile.csv"
    pair_profile.to_csv(pair_profile_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {pair_profile_path}")

    print("[8/9] Adjacency outputs + NEW pair-aware heatmaps ...")

    # ---- 8.1 Permuted adjacency order (kept pairs first -> pair_id -> core/peri -> coreness desc -> strength desc) ----
    role_order = {"core": 0, "periphery": 1, "insignificant": 2}
    tmp = node_roles.copy()
    tmp["is_kept_pair"] = tmp["pair_id"].apply(lambda x: int(int(x) in kept_pairs))
    tmp["is_sig_pair"] = tmp["pair_id"].apply(lambda x: int(pair_sig[int(x)]))

    tmp["keep_sort"] = -tmp["is_kept_pair"]
    tmp["sig_sort"] = -tmp["is_sig_pair"]
    tmp["pair_sort"] = tmp["pair_id"].astype(np.int64)
    tmp["role_sort"] = tmp["role"].map(role_order).astype(np.int64)
    tmp["core_sort"] = tmp["coreness"].fillna(-1.0)
    tmp["strength_sort"] = tmp["strength"]

    tmp = tmp.sort_values(
        by=["keep_sort", "sig_sort", "pair_sort", "role_sort", "core_sort", "strength_sort"],
        ascending=[True, True, True, True, False, False],
    )

    idx_sorted = tmp["node_id"].to_numpy(dtype=int)

    # Save permuted sparse adjacency
    A_perm = A[idx_sorted, :][:, idx_sorted].tocsr()
    perm_path = out_dir / "KM_config_adjacency_permuted.npz"
    save_npz(perm_path, A_perm)
    print(f"  saved: {perm_path}")

    # ---- NEW 8.2 Output pair membership counts + pair intervals in permuted order ----
    # Compute start/end positions for each pair in permuted order
    pos_in_perm = np.empty(n, dtype=int)
    pos_in_perm[idx_sorted] = np.arange(n)

    # For each pair, positions range
    pair_positions = {}
    for pid in range(K):
        node_ids = node_roles.loc[node_roles["pair_id"] == pid, "node_id"].to_numpy(dtype=int)
        if len(node_ids) == 0:
            continue
        ppos = pos_in_perm[node_ids]
        pair_positions[pid] = (int(ppos.min()), int(ppos.max()))

    pair_counts = pair_summary.copy()
    pair_counts["perm_start_idx"] = pair_counts["pair_id"].map(lambda x: pair_positions.get(int(x), (None, None))[0])
    pair_counts["perm_end_idx"] = pair_counts["pair_id"].map(lambda x: pair_positions.get(int(x), (None, None))[1])

    pair_counts_path = out_dir / "KM_config_pair_membership_counts.csv"
    pair_counts.to_csv(pair_counts_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {pair_counts_path}")

    # ---- 8.3 Original binned heatmap + NEW: draw pair boundary lines ----
    bin_id = np.minimum((pos_in_perm * N_BINS) // max(1, n), N_BINS - 1)

    B = np.zeros((N_BINS, N_BINS), dtype=float)
    for i, j, ww in zip(row_u, col_v, w):
        bi = int(bin_id[i])
        bj = int(bin_id[j])
        B[bi, bj] += ww
        B[bj, bi] += ww

    binned_csv_path = out_dir / "KM_config_adjacency_binned_matrix.csv"
    pd.DataFrame(B).to_csv(binned_csv_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {binned_csv_path}")

    # Pair boundary bins (only for kept pairs, to make lines meaningful)
    kept_pair_starts = []
    for pid in sorted(kept_pairs):
        start, end = pair_positions.get(pid, (None, None))
        if start is None:
            continue
        kept_pair_starts.append(int((start * N_BINS) // max(1, n)))
    kept_pair_starts = sorted(set([x for x in kept_pair_starts if 0 <= x < N_BINS]))

    heatmap_path = out_dir / "KM_config_adjacency_binned_heatmap.png"
    plot_matrix_heatmap(B, f"Binned adjacency ({N_BINS}x{N_BINS})", heatmap_path, log1p=LOG_HEATMAP)

    heatmap_lines_path = out_dir / "KM_config_adjacency_binned_heatmap_with_pair_lines.png"
    plot_matrix_heatmap(
        B,
        f"Binned adjacency ({N_BINS}x{N_BINS}) with kept-pair boundaries",
        heatmap_lines_path,
        log1p=LOG_HEATMAP,
        draw_lines=kept_pair_starts,
        figsize=(9, 8),
    )
    print(f"  saved: {heatmap_path}")
    print(f"  saved: {heatmap_lines_path}")

    # ---- NEW 8.4 Pair-level block heatmap (pair x pair) ----
    # Choose which pairs to visualize as blocks
    block_pairs = sorted(kept_pairs)
    if len(block_pairs) == 0 and FALLBACK_TO_ALL_SIGNIFICANT_FOR_BLOCK:
        block_pairs = sorted(pair_summary.loc[pair_summary["significant"], "pair_id"].tolist())

    # Map node -> block index
    pair_to_block = {pid: k for k, pid in enumerate(block_pairs)}
    other_idx = len(block_pairs) if INCLUDE_OTHER_BLOCK else None
    n_block = len(block_pairs) + (1 if INCLUDE_OTHER_BLOCK else 0)

    node_block = np.full(n, -1, dtype=int)
    for i in range(n):
        pid = int(pair_id[i])
        if pid in pair_to_block:
            node_block[i] = pair_to_block[pid]
        else:
            if INCLUDE_OTHER_BLOCK:
                node_block[i] = other_idx

    M_pair = np.zeros((n_block, n_block), dtype=float)
    for i, j, ww in zip(row_u, col_v, w):
        bi = int(node_block[i])
        bj = int(node_block[j])
        if bi < 0 or bj < 0:
            continue
        M_pair[bi, bj] += ww
        M_pair[bj, bi] += ww

    pair_block_csv = out_dir / "KM_config_pair_block_matrix.csv"
    pd.DataFrame(M_pair).to_csv(pair_block_csv, index=False, encoding="utf-8-sig")

    pair_block_heatmap = out_dir / "KM_config_pair_block_heatmap.png"
    plot_matrix_heatmap(
        M_pair,
        f"Pair-block adjacency ({len(block_pairs)} pairs" + (", +OTHER" if INCLUDE_OTHER_BLOCK else "") + ")",
        pair_block_heatmap,
        log1p=True,
        figsize=(8, 7),
    )
    print(f"  saved: {pair_block_csv}")
    print(f"  saved: {pair_block_heatmap}")

    # ---- NEW 8.5 Pair-role block heatmap ((pair,role) x (pair,role)) ----
    # For each kept/significant pair: split into core/periphery; optionally include OTHER as last group
    # group index: 2*block + role_bit (core=0, peri=1)
    # OTHER group: last index (if enabled) and we won't split role there (just one group)
    if INCLUDE_OTHER_BLOCK:
        n_role_block = 2 * len(block_pairs) + 1
        other_role_idx = 2 * len(block_pairs)
    else:
        n_role_block = 2 * len(block_pairs)
        other_role_idx = None

    node_role_block = np.full(n, -1, dtype=int)
    for i in range(n):
        pid = int(pair_id[i])
        x = float(coreness.get(i, np.nan))
        is_core = (x >= 0.5)

        if pid in pair_to_block:
            b = pair_to_block[pid]
            node_role_block[i] = 2 * b + (0 if is_core else 1)
        else:
            if INCLUDE_OTHER_BLOCK:
                node_role_block[i] = other_role_idx

    M_pair_role = np.zeros((n_role_block, n_role_block), dtype=float)
    for i, j, ww in zip(row_u, col_v, w):
        bi = int(node_role_block[i])
        bj = int(node_role_block[j])
        if bi < 0 or bj < 0:
            continue
        M_pair_role[bi, bj] += ww
        M_pair_role[bj, bi] += ww

    pair_role_block_csv = out_dir / "KM_config_pair_role_block_matrix.csv"
    pd.DataFrame(M_pair_role).to_csv(pair_role_block_csv, index=False, encoding="utf-8-sig")

    # draw lines between pairs (every 2 groups)
    lines = [2 * k for k in range(1, len(block_pairs))]
    if INCLUDE_OTHER_BLOCK:
        lines.append(other_role_idx)

    pair_role_block_heatmap = out_dir / "KM_config_pair_role_block_heatmap.png"
    plot_matrix_heatmap(
        M_pair_role,
        f"Pair-role block adjacency (core/periphery per pair" + (", +OTHER" if INCLUDE_OTHER_BLOCK else "") + ")",
        pair_role_block_heatmap,
        log1p=True,
        draw_lines=lines,
        figsize=(9, 8),
    )
    print(f"  saved: {pair_role_block_csv}")
    print(f"  saved: {pair_role_block_heatmap}")

    # ---- 8.6 Strength boxplot by role ----
    groups = ["core", "periphery", "insignificant"]
    data = [node_roles.loc[node_roles["role"] == g, "strength"].to_numpy() for g in groups]

    plt.figure()
    plt.boxplot(data, tick_labels=groups, showfliers=SHOW_FLIERS)
    plt.title("Weighted degree (strength) by role")
    plt.ylabel("Strength (sum of Newman weights)")
    if LOG_BOXPLOT_Y:
        plt.yscale("log")
        plt.ylabel("Strength (log scale)")
    plt.tight_layout()
    boxplot_path = out_dir / "KM_config_strength_boxplot.png"
    plt.savefig(boxplot_path, dpi=220)
    plt.close()
    print(f"  saved: {boxplot_path}")

    print("\nDONE.")
    print("Key outputs:")
    print(" - KM_config_node_roles_with_metrics.csv")
    print(" - KM_config_pair_summary.csv")
    print(" - KM_config_pair_membership_counts.csv   <-- NEW: 每个pair节点数+重排区间")
    print(" - KM_config_pair_cross_community_profile.csv")
    print(" - KM_config_adjacency_permuted.npz")
    print(" - KM_config_adjacency_binned_heatmap.png (+ csv)")
    print(" - KM_config_adjacency_binned_heatmap_with_pair_lines.png   <-- NEW")
    print(" - KM_config_pair_block_heatmap.png (+ matrix csv)          <-- NEW")
    print(" - KM_config_pair_role_block_heatmap.png (+ matrix csv)     <-- NEW")
    print(" - KM_config_strength_boxplot.png")


if __name__ == "__main__":
    main()
