# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
WDSBM_CoauthorCount_S5S2_Nested_SaveLevels0to4.py

1) 读取 author-paper 数据
2) Count 投影：U = A A^T （每篇论文对每对作者贡献 +1）
3) S5-S2：按发文量过滤 -> 最大连通子图
4) 导出 filtered edgelist
5) graph-tool nested WDSBM：deg_corr + discrete-poisson + MDL（多次重启取最优）
6) 保存 nested bs + level0-4 作者层块编号 + 每层块规模 + 层间映射

依赖：
- numpy, pandas, scipy, networkx
- graph-tool (Linux)

注意：
- 该脚本会重新拟合 nested SBM（你要求“重新运行并保存 level0-4 全部信息”）
"""

import os
import hashlib
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import graph_tool.all as gt


# =========================
# 1. 配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
OUT_DIR = str(_OUTPUT_ROOT / "07_hierarchy_core_periphery" / "nested_weighted_dcsbm")
os.makedirs(OUT_DIR, exist_ok=True)

AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"
YEAR_COL_CANDIDATES = ["publication_year", "year", "py", "pub_year", "publicationyear"]

TIME_START = 2006
TIME_END = 2025
MIN_PAPER_COUNT = 2

# 与 Windows 检查版一致
BINARIZE_INCIDENCE = True                  # incidence A 0/1（同一 author-paper 多行不重复计数）
COUNT_UNIQUE_PAPERS_FOR_S5 = True          # S5 发文量按 nunique(paper_id)

PIPELINE = ["S5", "S2"]  # 固定 S5-S2

# === nested WDSBM 配置 ===
RUN_WDSBM = True
REC_TYPE = "discrete-poisson"
DEG_CORR = True
N_RESTARTS = 1
RNG_SEED = 42

# 保存层数：level0..level4
SAVE_MAX_LEVEL = 4


# =========================
# 2. 工具函数
# =========================
def file_fingerprint(path: str, n_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if str(c).strip().lower() == str(cand).strip().lower():
                return c
    return ""


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="gbk")


def print_graph_stats(tag: str, G: nx.Graph):
    print(f"\n[{tag}] N={G.number_of_nodes()}, E={G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        return
    isolates = [n for n, d in G.degree() if d == 0]
    print(f"[{tag}] isolates(deg=0)={len(isolates)}")
    comps = list(nx.connected_components(G)) if G.number_of_edges() > 0 else []
    if comps:
        sizes = sorted([len(c) for c in comps], reverse=True)
        print(f"[{tag}] components={len(comps)}, largest={sizes[0]}, top5={sizes[:5]}")
    if G.number_of_edges() > 0:
        w = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
        print(f"[{tag}] weight stats: min={w.min():.6f}, median={np.median(w):.6f}, max={w.max():.6f}")


def to_int_array(x):
    """
    兼容 graph-tool 的 VertexPropertyMap / PropertyArray / numpy：
    - 有 .a 用 .a
    - 否则 np.asarray(x)
    """
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=int)
    return np.asarray(x, dtype=int)


# =========================
# 3. 数据加载 + 时间切片
# =========================
def load_data(csv_path: str):
    print(f"[LOAD] {csv_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    size = os.path.getsize(csv_path)
    fp = file_fingerprint(csv_path)
    print(f"[LOAD] File size={size} bytes, sha256(head1MB)={fp}")

    df = safe_read_csv(csv_path)
    print(f"[LOAD] Raw rows={len(df)}, cols={len(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].astype(str).str.strip() != "") & (df[PAPER_COL].astype(str).str.strip() != "")]
    after = len(df)
    print(f"[LOAD] Rows: {before} -> {after} after dropping empty author/paper")

    year_col = detect_column(df, YEAR_COL_CANDIDATES)
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
        y = df[year_col].values
        print(f"[LOAD] Detected year col: {year_col}")
        print(f"[LOAD] Year stats: min={y.min()}, median={int(np.median(y))}, max={y.max()}, <=0={(y<=0).sum()}")
    else:
        print("[LOAD] WARNING: No year column detected; will use full data.")

    return df, year_col


def slice_by_year(df: pd.DataFrame, year_col: str, t0: int, t1: int) -> pd.DataFrame:
    if not year_col:
        print("[SLICE] WARNING: no year column; using full data.")
        return df.copy()
    df2 = df[(df[year_col] >= t0) & (df[year_col] <= t1)].copy()
    print(f"[SLICE] Time slice {t0}..{t1}: rows={len(df2)}")
    return df2


# =========================
# 4. Count 投影构图（U = A A^T）
# =========================
def build_index_maps(df_slice: pd.DataFrame):
    author_ids = df_slice[AUTHOR_COL].unique()
    paper_ids = df_slice[PAPER_COL].unique()
    aid2idx = {a: i for i, a in enumerate(author_ids)}
    pid2idx = {p: i for i, p in enumerate(paper_ids)}

    df_idx = df_slice[[AUTHOR_COL, PAPER_COL]].copy()
    df_idx["ai"] = df_idx[AUTHOR_COL].map(aid2idx).astype(int)
    df_idx["pk"] = df_idx[PAPER_COL].map(pid2idx).astype(int)

    print(f"[INDEX] Unique authors={len(author_ids)}, unique papers={len(paper_ids)}")
    print(f"[INDEX] author-paper rows={len(df_idx)}")
    return author_ids, paper_ids, df_idx


def get_paper_counts_for_s5(df_slice: pd.DataFrame) -> dict:
    if COUNT_UNIQUE_PAPERS_FOR_S5:
        counts = df_slice.groupby(AUTHOR_COL)[PAPER_COL].nunique().to_dict()
        print(f"[S5] paper_counts=nunique(paper_id): authors={len(counts)}")
    else:
        counts = df_slice.groupby(AUTHOR_COL)[PAPER_COL].size().to_dict()
        print(f"[S5] paper_counts=size(rows): authors={len(counts)}")

    vals = np.array(list(counts.values()), dtype=float)
    print(f"[S5] count stats: min={vals.min():.0f}, median={np.median(vals):.0f}, max={vals.max():.0f}, >=2={(vals>=2).sum():.0f}")
    return counts


def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()
    return U


def build_count_projection(author_ids, paper_ids, df_idx):
    print("[COUNT] Building incidence matrix A (CSR)...")
    A = sp.coo_matrix(
        (np.ones(len(df_idx), dtype=np.float64), (df_idx["ai"].values, df_idx["pk"].values)),
        shape=(len(author_ids), len(paper_ids))
    )
    A.sum_duplicates()
    if BINARIZE_INCIDENCE:
        A.data[:] = 1.0
    A = A.tocsr()

    n_authors_per_paper = np.asarray(A.sum(axis=0)).ravel()
    print(f"[COUNT] A shape={A.shape}, nnz(A)={A.nnz}")
    print(f"[COUNT] paper author-count: min={n_authors_per_paper.min():.0f}, median={np.median(n_authors_per_paper):.0f}, max={n_authors_per_paper.max():.0f}")

    U = A @ A.T
    U = zero_diag_csr_inplace(U.tocsr())
    print(f"[COUNT] U shape={U.shape}, nnz(U)={U.nnz}")
    return U


def nx_from_sparse_upper(U_csr, author_ids):
    U = U_csr.tocoo()
    mask = U.row < U.col
    rows = U.row[mask]
    cols = U.col[mask]
    data = U.data[mask].astype(np.float64)

    G = nx.Graph()
    G.add_nodes_from(author_ids)

    for i, j, w in zip(rows, cols, data):
        if np.isfinite(w) and w > 0:
            ww = int(round(float(w)))
            if ww > 0:
                G.add_edge(author_ids[i], author_ids[j], weight=ww)
    return G


# =========================
# 5. S5-S2 过滤（与 Windows 检查版一致）
# =========================
def filter_min_papers_s5(G, paper_counts, min_count):
    G2 = G.copy()
    remove = [n for n in G2.nodes() if paper_counts.get(n, 0) < min_count]
    G2.remove_nodes_from(remove)
    print(f"[FILTER S5] removed={len(remove)} (papers<{min_count})")
    return G2


def filter_lcc_s2(G):
    if G.number_of_nodes() == 0:
        return G
    if G.number_of_edges() == 0:
        print("[FILTER S2] No edges; skip LCC.")
        return G
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()


def run_pipeline(G, paper_counts):
    Gc = G
    for step in PIPELINE:
        if step == "S5":
            Gc = filter_min_papers_s5(Gc, paper_counts, MIN_PAPER_COUNT)
            print_graph_stats("AFTER S5", Gc)
        elif step == "S2":
            Gc = filter_lcc_s2(Gc)
            print_graph_stats("AFTER S2 (LCC)", Gc)
        else:
            raise ValueError(f"Unsupported step: {step}")
    return Gc


# =========================
# 6. graph-tool：NX -> GT + nested WDSBM
# =========================
def nx_to_graphtool_count(G_nx: nx.Graph):
    g = gt.Graph(directed=False)
    vp_id = g.new_vp("string")
    ep_w = g.new_ep("int")  # 计数权重

    nx2gt = {}
    for n in G_nx.nodes():  # 依赖 insertion order（你前面已保证稳定）
        v = g.add_vertex()
        vp_id[v] = str(n)
        nx2gt[n] = v

    for u, v, d in G_nx.edges(data=True):
        w = int(d.get("weight", 0))
        if w <= 0:
            continue
        e = g.add_edge(nx2gt[u], nx2gt[v])
        ep_w[e] = w

    g.vp["author_id"] = vp_id
    g.ep["w_count"] = ep_w
    return g


def nested_B_levels(state):
    """返回每一层 BlockState 的 B（不调用 NestedBlockState.get_B）"""
    Bs = []
    for lev in state.levels:
        if hasattr(lev, "get_B"):
            Bs.append(int(lev.get_B()))
        else:
            # 兜底：unique(get_blocks)
            b = to_int_array(lev.get_blocks())
            Bs.append(int(np.unique(b).size))
    return Bs


def fit_nested_wdsbm_mdl_count(g: gt.Graph):
    """
    Nested SBM：minimize_nested_blockmodel_dl
    多次重启取 entropy 最小
    """
    np.random.seed(RNG_SEED)
    gt.seed_rng(RNG_SEED)

    state_args = dict(
        deg_corr=DEG_CORR,
        recs=[g.ep["w_count"]],
        rec_types=[REC_TYPE],
    )

    N = g.num_vertices()
    print(f"[WDSBM-NESTED] N={N}, rec_type={REC_TYPE}, deg_corr={DEG_CORR}, restarts={N_RESTARTS}")

    best_state = None
    best_dl = np.inf
    best_Bs = None

    for r in range(N_RESTARTS):
        seed = RNG_SEED + 999 * r
        np.random.seed(seed)
        gt.seed_rng(seed)

        # 兼容不同版本的参数形式
        try:
            st = gt.minimize_nested_blockmodel_dl(g, state_args=state_args)
        except TypeError:
            st = gt.minimize_nested_blockmodel_dl(g, **state_args)

        dl = float(st.entropy())
        Bs = nested_B_levels(st)
        print(f"[MDL] restart {r+1}/{N_RESTARTS}: B_levels={Bs}, B0={Bs[0] if Bs else -1}, DL={dl:.6f}")

        if dl < best_dl:
            best_dl = dl
            best_state = st
            best_Bs = Bs

    print(f"[MDL] BEST: B_levels={best_Bs}, B0={best_Bs[0] if best_Bs else -1}, DL={best_dl:.6f}")
    return best_state, best_dl, best_Bs


# =========================
# 7. 保存：nested bs + level0-4 作者层块编号 + 层间映射 + 块规模
# =========================
def compute_vertex_blocks_levels_0_to_K(state, K: int):
    """
    从 nested state.get_bs() 还原“作者层”的 level0..K 块编号：
      v0 = bs[0][vertex]
      v1 = bs[1][v0]
      v2 = bs[2][v1]
      ...
    """
    bs = state.get_bs()
    if not bs:
        raise ValueError("state.get_bs() is empty")

    b0 = to_int_array(bs[0])
    out = [b0]
    prev = b0

    maxL = min(K, len(bs) - 1)
    for L in range(1, maxL + 1):
        parent_map = to_int_array(bs[L])  # length = B_{L-1} (上一层块图的顶点数)
        if prev.max() >= len(parent_map) or prev.min() < 0:
            raise ValueError(
                f"Level mapping failed at L={L}: prev range [{prev.min()},{prev.max()}], "
                f"but map length={len(parent_map)}. (block labels not aligned)"
            )
        cur = parent_map[prev]
        out.append(cur)
        prev = cur

    # 不足的层用 -1 补齐
    while len(out) < (K + 1):
        out.append(np.full_like(out[0], fill_value=-1, dtype=int))

    return out, bs


def save_nested_all_levels(g: gt.Graph, state, best_dl: float, best_Bs: list, filtered_edgelist_path: str):
    # 1) 保存 nested bs（关键：以后不用重跑就能恢复 level2/3/4）
    bs = state.get_bs()
    bs_path = os.path.join(OUT_DIR, "nested_bs.pkl")
    with open(bs_path, "wb") as f:
        pickle.dump(bs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] nested_bs.pkl saved: {bs_path}")

    # 2) 保存 level0..4 的“作者层块编号”
    levels_arrs, bs_list = compute_vertex_blocks_levels_0_to_K(state, SAVE_MAX_LEVEL)
    # levels_arrs: [b0,b1,b2,b3,b4] each length = N(vertices)

    df = pd.DataFrame({
        "author_id": [g.vp["author_id"][v] for v in g.vertices()],
        "block_level0": levels_arrs[0],
        "block_level1": levels_arrs[1],
        "block_level2": levels_arrs[2],
        "block_level3": levels_arrs[3],
        "block_level4": levels_arrs[4],
    })
    out_levels = os.path.join(OUT_DIR, "author_blocks_levels_0_4.csv")
    df.to_csv(out_levels, index=False, encoding="utf-8-sig")
    print(f"[SAVE] author_blocks_levels_0_4.csv: {out_levels}")

    # 3) 每层块规模（按“作者数”统计）
    for L in range(SAVE_MAX_LEVEL + 1):
        arr = levels_arrs[L]
        if (arr < 0).all():
            continue
        # 让 block_id 连续到 max（便于对齐）
        B = int(arr.max()) + 1 if arr.size > 0 and arr.max() >= 0 else 0
        if B <= 0:
            continue
        counts = np.bincount(arr.astype(int), minlength=B)
        dfc = pd.DataFrame({"block_id": np.arange(B, dtype=int), "size": counts.astype(int)})
        dfc = dfc.sort_values("size", ascending=False)
        p = os.path.join(OUT_DIR, f"blocks_level{L}_sizes.csv")
        dfc.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[SAVE] blocks_level{L}_sizes.csv: {p}")

    # 4) 保存层间映射：level(L-1)块 -> level(L)块
    # 注意：bs_list[L] 的索引就是 level(L-1) 的块编号
    # 所以 mapping 直接是 child_block = 0..len(bs_list[L])-1, parent_block = bs_list[L][child]
    max_map_L = min(SAVE_MAX_LEVEL, len(bs_list) - 1)
    for L in range(1, max_map_L + 1):
        parent_map = to_int_array(bs_list[L])
        dfm = pd.DataFrame({
            f"block_level{L-1}": np.arange(len(parent_map), dtype=int),
            f"block_level{L}": parent_map.astype(int),
        })
        mp = os.path.join(OUT_DIR, f"map_level{L-1}_to_level{L}.csv")
        dfm.to_csv(mp, index=False, encoding="utf-8-sig")
        print(f"[SAVE] map_level{L-1}_to_level{L}.csv: {mp}")

    # 5) model info
    info_path = os.path.join(OUT_DIR, "model_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"PATH_INPUT={PATH_INPUT}\n")
        f.write(f"TIME_START={TIME_START}, TIME_END={TIME_END}\n")
        f.write("PROJECTION=COUNT(U=A*A^T)\n")
        f.write(f"PIPELINE={PIPELINE}\n")
        f.write(f"MIN_PAPER_COUNT={MIN_PAPER_COUNT}\n")
        f.write(f"BINARIZE_INCIDENCE={BINARIZE_INCIDENCE}\n")
        f.write(f"COUNT_UNIQUE_PAPERS_FOR_S5={COUNT_UNIQUE_PAPERS_FOR_S5}\n")
        f.write(f"nested_rec_type={REC_TYPE}\n")
        f.write(f"deg_corr={DEG_CORR}\n")
        f.write(f"restarts={N_RESTARTS}, seed={RNG_SEED}\n")
        f.write(f"B_levels(best)={best_Bs}\n")
        f.write(f"DL(entropy)={best_dl:.10f}\n")
        f.write(f"filtered_edgelist={filtered_edgelist_path}\n")
        f.write(f"nested_bs_pkl={bs_path}\n")
        f.write(f"saved_levels=0..{SAVE_MAX_LEVEL}\n")
    print(f"[SAVE] model_info.txt: {info_path}")


# =========================
# 8. 主程序
# =========================
def main():
    df, year_col = load_data(PATH_INPUT)
    df_slice = slice_by_year(df, year_col, TIME_START, TIME_END)
    if len(df_slice) == 0:
        raise ValueError("Time slice is empty. Check year parsing/time window.")

    author_ids, paper_ids, df_idx = build_index_maps(df_slice)
    paper_counts = get_paper_counts_for_s5(df_slice)

    # Count projection -> NX graph
    U = build_count_projection(author_ids, paper_ids, df_idx)
    G0 = nx_from_sparse_upper(U, author_ids)
    print_graph_stats("RAW GRAPH (COUNT projection)", G0)

    print("\n=== RUN PIPELINE: S5 -> S2 ===")
    Gf = run_pipeline(G0, paper_counts)

    print("\n=== FINAL CHECK ===")
    print_graph_stats("FINAL", Gf)

    # 导出过滤后边表供核验
    edges = [(u, v, int(d.get("weight", 0))) for u, v, d in Gf.edges(data=True)]
    df_edges = pd.DataFrame(edges, columns=["u", "v", "weight"])
    filtered_path = os.path.join(OUT_DIR, "filtered_graph_edgelist.csv")
    df_edges.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    print(f"[EXPORT] filtered edgelist saved to: {filtered_path}")

    if not RUN_WDSBM:
        print("[WDSBM] RUN_WDSBM=False, stop after filtering/export.")
        return

    g = nx_to_graphtool_count(Gf)
    print(f"[GT] Graph-tool: N={g.num_vertices()}, E={g.num_edges()}")

    state, best_dl, best_Bs = fit_nested_wdsbm_mdl_count(g)
    save_nested_all_levels(g, state, best_dl, best_Bs, filtered_path)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
