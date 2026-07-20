# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
ego_on_active_network_s5s2_gephi_with_seed_relative_node_attrs.py

中文：
- 构建 Newman 加权合著网络 -> S5(发文量) -> S2(最大连通分量) 得到活跃作者网络 G_active
- 在 G_active 上抽取 seed 的 ego 网络(1/2阶)
- 边属性(Edge attrs): newman_weight/weight, freq, avg_year, edge_first_year, edge_last_year, edge_duration
- 节点属性(Node attrs, seed-relative):
    * freq_with_seed:
        - collaborator node: freq(seed, node)
        - seed node: sum_v freq(seed, v)
    * avg_year_with_seed:
        - collaborator node: avg_year(seed, node)
        - seed node: avg publication year of seed
    * duration_with_seed:
        - collaborator node: duration(seed, node)
        - seed node: active span of seed (last_pub - first_pub)
- 输出：Gephi 可导入图文件(gexf/graphml) + 每个 ego 的边表 CSV + 汇总 CSV（绘图用）

English summary:
- Build active network (Newman weighted) with S5-S2 filtering
- Extract ego graphs; write seed-relative metrics as node attributes; keep Newman weight as edge attribute
- Export Gephi graph + per-ego edge CSV + consolidated plotting CSV
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =========================
# 配置 / Config
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "02_network_construction" / "ego_network_extraction")

AUTHOR_COL = "author_id"
PAPER_COL  = "paper_id"
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

# S5-S2：活跃作者网络
MIN_PAPER_COUNT = 2

# seeds + ego orders
SEED_AUTHORS = ["author_6690", "author_18007"]
EGO_ORDERS   = [1, 2]

# 时间窗（可选）
TIME_START = 2006
TIME_END   = 2025

# Gephi 输出格式
GRAPH_FORMAT = "gexf"  # "gexf" or "graphml"

# 是否只输出 seed-合作者 的星型边
EXPORT_STAR_EDGES_ONLY = False


# =========================
# Helpers
# =========================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == str(df_col).lower():
                return df_col
    return ""

def load_and_clean(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"文件未找到: {csv_path}")

    print(f"正在读取文件: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    year_col = detect_column(df, YEAR_COL_CANDIDATES)

    keep_cols = [AUTHOR_COL, PAPER_COL]
    if year_col:
        keep_cols.append(year_col)
    df = df[keep_cols].copy()

    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]

    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)

    # 每个 (author,paper) 只保留一条
    if year_col:
        df = df.groupby([AUTHOR_COL, PAPER_COL], as_index=False).agg({year_col: "max"})
    else:
        df = df.drop_duplicates([AUTHOR_COL, PAPER_COL])

    return df, year_col

def build_index_maps_fast(df_slice: pd.DataFrame):
    ai, author_ids = pd.factorize(df_slice[AUTHOR_COL], sort=False)
    pk, paper_ids  = pd.factorize(df_slice[PAPER_COL], sort=False)
    df_idx = df_slice.copy()
    df_idx["ai"] = ai.astype(int)
    df_idx["pk"] = pk.astype(int)
    return np.asarray(author_ids, dtype=object), np.asarray(paper_ids, dtype=object), df_idx

def zero_diag_csr_inplace(U):
    U = U.tolil(copy=False)
    U.setdiag(0)
    U = U.tocsr(copy=False)
    U.eliminate_zeros()
    return U

def edges_from_sparse(S, author_ids):
    S = S.tocoo()
    mask = S.row < S.col
    rows, cols, data = S.row[mask], S.col[mask], S.data[mask]
    return pd.DataFrame({
        "src_author_id": author_ids[rows],
        "dst_author_id": author_ids[cols],
        "weight": data.astype(float)
    })

def build_network_matrix_newman(author_ids, paper_ids, df_idx):
    if not HAVE_SCIPY:
        raise RuntimeError("未安装 SciPy，无法进行矩阵构图（Newman）。")

    nA, nP = len(author_ids), len(paper_ids)
    A = sp.csr_matrix(
        (np.ones(len(df_idx), dtype=np.int8),
         (df_idx["ai"].to_numpy(), df_idx["pk"].to_numpy())),
        shape=(nA, nP)
    )

    n_authors = np.asarray(A.sum(axis=0)).ravel()
    w = np.zeros_like(n_authors, dtype=float)
    mask = n_authors > 1
    w[mask] = 1.0 / (n_authors[mask] - 1.0)

    U = A @ sp.diags(w) @ A.T
    U = zero_diag_csr_inplace(U.tocsr())
    return edges_from_sparse(U, author_ids)

def filter_s2_lcc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()

def export_gephi_graph(G: nx.Graph, out_path: str, fmt: str):
    fmt = fmt.lower()
    if fmt == "gexf":
        nx.write_gexf(G, out_path)
    elif fmt == "graphml":
        nx.write_graphml(G, out_path)
    else:
        raise ValueError("GRAPH_FORMAT must be 'gexf' or 'graphml'")

def compute_author_year_stats(df_s5: pd.DataFrame, year_col: str):
    """
    For node attribute on seed:
    - avg_pub_year, first_pub_year, last_pub_year, active_span
    """
    if not year_col:
        return {}, {}, {}, {}

    g = df_s5[df_s5[year_col] > 0].groupby(AUTHOR_COL, sort=False)[year_col]
    firsty = g.min().to_dict()
    lasty  = g.max().to_dict()
    avgy   = g.mean().to_dict()
    span   = {k: int(lasty.get(k, 0) - firsty.get(k, 0)) for k in set(firsty) | set(lasty)}
    # convert to Python types
    avgy = {k: float(v) for k, v in avgy.items()}
    firsty = {k: int(v) for k, v in firsty.items()}
    lasty  = {k: int(v) for k, v in lasty.items()}
    return avgy, firsty, lasty, span

def compute_edge_metrics_for_edge_set(df_s5: pd.DataFrame, year_col: str, nodes_set: set, edge_set: set):
    """
    Paper-wise:
    - freq: coauthored paper count
    - avg_year / first_year / last_year / duration (year>0)
    Only update for edges in edge_set.
    """
    if not year_col:
        # no year -> only freq can be computed robustly (still need grouping by paper)
        # but we can still compute freq without year.
        pass

    dfv = df_s5[df_s5[AUTHOR_COL].isin(nodes_set)].copy()
    if dfv.empty:
        return {}, {}, {}, {}, {}

    INF = 10**9
    freq = defaultdict(int)
    year_sum = defaultdict(int)
    year_n = defaultdict(int)
    first_year = defaultdict(lambda: INF)
    last_year = defaultdict(lambda: -1)

    gp = dfv.groupby(PAPER_COL, sort=False)

    for pid, sub in tqdm(gp, desc="Computing ego edge metrics (paper-wise)", total=gp.ngroups):
        authors = sub[AUTHOR_COL].astype(str).unique()
        if len(authors) < 2:
            continue
        authors.sort()

        y = int(sub[year_col].max()) if year_col else 0
        y_valid = (y > 0)

        k = len(authors)
        for i in range(k - 1):
            u = authors[i]
            for j in range(i + 1, k):
                v = authors[j]
                key = (u, v)  # u<v due to sort
                if key not in edge_set:
                    continue

                freq[key] += 1
                if y_valid:
                    year_sum[key] += y
                    year_n[key] += 1
                    if y < first_year[key]:
                        first_year[key] = y
                    if y > last_year[key]:
                        last_year[key] = y

    avg_year = {}
    duration = {}
    fy_out = {}
    ly_out = {}

    for key in edge_set:
        n = year_n.get(key, 0)
        if n > 0:
            avg_year[key] = float(year_sum[key] / n)
            fy = int(first_year[key])
            ly = int(last_year[key])
            fy_out[key] = fy
            ly_out[key] = ly
            duration[key] = int(ly - fy)
        else:
            avg_year[key] = 0.0
            fy_out[key] = 0
            ly_out[key] = 0
            duration[key] = 0

    return freq, avg_year, fy_out, ly_out, duration


# =========================
# Main
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)

    df_all, year_col = load_and_clean(PATH_INPUT)
    print(f"清洗后行数(作者-论文唯一): {len(df_all)}")
    print(f"识别 year_col: {year_col or 'None'}")

    # 时间窗过滤
    if year_col and TIME_START is not None and TIME_END is not None:
        df_slice = df_all[(df_all[year_col] >= TIME_START) & (df_all[year_col] <= TIME_END)].copy()
        t_label = f"{TIME_START}-{TIME_END}"
    else:
        df_slice = df_all.copy()
        t_label = "ALL"

    if df_slice.empty:
        print("[ERROR] 时间窗过滤后无数据。")
        return

    # S5：发文量过滤（先过滤数据行，降低矩阵规模）
    paper_counts = df_slice[AUTHOR_COL].value_counts()
    keep_authors = set(paper_counts[paper_counts >= MIN_PAPER_COUNT].index.astype(str))
    df_s5 = df_slice[df_slice[AUTHOR_COL].astype(str).isin(keep_authors)].copy()
    print(f"S5 后作者数={len(keep_authors)}, 行数={len(df_s5)}")

    if df_s5.empty:
        print("[ERROR] S5 后网络为空。")
        return

    # 预计算作者自身的发文时间统计（给 seed 节点用）
    avg_pub_year, first_pub_year, last_pub_year, active_span = compute_author_year_stats(df_s5, year_col)

    # Newman 矩阵构图
    author_ids, paper_ids, df_idx = build_index_maps_fast(df_s5)
    print(f"矩阵构图规模: 作者数={len(author_ids)}, 论文数={len(paper_ids)}")

    edges_all = build_network_matrix_newman(author_ids, paper_ids, df_idx)

    G_base = nx.from_pandas_edgelist(
        edges_all,
        source="src_author_id",
        target="dst_author_id",
        edge_attr="weight",
        create_using=nx.Graph()
    )

    # S2：最大连通分量 -> 活跃作者网络
    G_active = filter_s2_lcc(G_base)
    print(f"S2(LCC) 后活跃作者网络: nodes={G_active.number_of_nodes()}, edges={G_active.number_of_edges()}")

    if G_active.number_of_nodes() == 0:
        print("[ERROR] S2 后网络为空。")
        return

    all_edges_for_plot = []

    for seed in SEED_AUTHORS:
        seed = str(seed)
        if seed not in G_active:
            print(f"[WARN] seed {seed} 不在活跃作者网络（S5-S2）中，跳过。")
            continue

        for order in EGO_ORDERS:
            print(f"\n=== Seed={seed}, ego_order={order} ===")

            # ego nodes/edges from active network
            G_ego_full = nx.ego_graph(G_active, seed, radius=order, center=True, undirected=True)
            nodes_set = set(map(str, G_ego_full.nodes()))
            if G_ego_full.number_of_edges() == 0:
                print("[WARN] ego 子图无边，跳过。")
                continue

            # Decide which edges to export
            if EXPORT_STAR_EDGES_ONLY:
                # Keep only edges incident to seed
                edge_pairs = [(str(u), str(v)) for u, v in G_ego_full.edges() if (str(u) == seed or str(v) == seed)]
                # Build star graph with same node set (or you can restrict nodes to seed+neighbors)
                G_ego = nx.Graph()
                G_ego.add_nodes_from(nodes_set)
                G_ego.add_edges_from(edge_pairs)
            else:
                G_ego = G_ego_full
                edge_pairs = [(str(u), str(v)) for u, v in G_ego.edges()]

            # Edge set key=(min,max) for metric computation
            edge_set = set()
            for u, v in edge_pairs:
                a, b = (u, v) if u < v else (v, u)
                edge_set.add((a, b))

            # Compute freq/avg_year/duration for edges in this ego graph (using df_s5)
            freq, avg_year, fy, ly, dur = compute_edge_metrics_for_edge_set(
                df_s5=df_s5,
                year_col=year_col,
                nodes_set=nodes_set,
                edge_set=edge_set
            )

            # -------------------------
            # 1) Write edge attributes (Gephi edges)
            # -------------------------
            edges_rows = []
            for u, v in edge_pairs:
                u, v = str(u), str(v)
                a, b = (u, v) if u < v else (v, u)
                key = (a, b)

                # Newman weight from active network
                w = float(G_active[u][v]["weight"]) if G_active.has_edge(u, v) else float(G_ego[u][v].get("weight", 0.0))

                f = int(freq.get(key, 0))
                ay = float(avg_year.get(key, 0.0))
                firsty = int(fy.get(key, 0))
                lasty  = int(ly.get(key, 0))
                d = int(dur.get(key, 0))

                # Edge attrs
                G_ego[u][v]["weight"] = w                 # Gephi default
                G_ego[u][v]["newman_weight"] = w          # explicit
                G_ego[u][v]["freq"] = f
                G_ego[u][v]["avg_year"] = float(round(ay, 4))
                G_ego[u][v]["edge_first_year"] = firsty
                G_ego[u][v]["edge_last_year"] = lasty
                G_ego[u][v]["edge_duration"] = d

                edges_rows.append({
                    "seed": seed,
                    "ego_order": order,
                    "src_author_id": a,
                    "dst_author_id": b,
                    "freq": f,
                    "newman_weight": round(w, 6),
                    "avg_year": round(ay, 4),
                    "edge_first_year": firsty,
                    "edge_last_year": lasty,
                    "edge_duration": d
                })

            edges_df = pd.DataFrame(edges_rows)

            # -------------------------
            # 2) Write node attributes (Gephi nodes) — seed-relative
            # -------------------------
            # Defaults
            nx.set_node_attributes(G_ego, {n: 0 for n in G_ego.nodes()}, "freq_with_seed")
            nx.set_node_attributes(G_ego, {n: 0.0 for n in G_ego.nodes()}, "avg_year_with_seed")
            nx.set_node_attributes(G_ego, {n: 0 for n in G_ego.nodes()}, "duration_with_seed")

            # Optional useful identifiers
            nx.set_node_attributes(G_ego, {n: (1 if n == seed else 0) for n in G_ego.nodes()}, "is_seed")
            nx.set_node_attributes(G_ego, {n: order for n in G_ego.nodes()}, "ego_order")
            nx.set_node_attributes(G_ego, {n: t_label for n in G_ego.nodes()}, "time_window")

            # Collaborator nodes: derive from seed-edge metrics
            # (only if seed-node edge exists in this exported graph)
            seed_total_freq = 0
            for n in G_ego.nodes():
                n = str(n)
                if n == seed:
                    continue

                # does this ego graph contain the seed-n edge?
                if not G_ego.has_edge(seed, n):
                    continue

                a, b = (seed, n) if seed < n else (n, seed)
                key = (a, b)

                f = int(freq.get(key, 0))
                ay = float(avg_year.get(key, 0.0))
                d  = int(dur.get(key, 0))

                G_ego.nodes[n]["freq_with_seed"] = f
                G_ego.nodes[n]["avg_year_with_seed"] = float(round(ay, 4))
                G_ego.nodes[n]["duration_with_seed"] = d

                seed_total_freq += f

            # Seed node: special definitions
            seed_avg_pub = float(round(avg_pub_year.get(seed, 0.0), 4)) if year_col else 0.0
            seed_span = int(active_span.get(seed, 0)) if year_col else 0

            G_ego.nodes[seed]["freq_with_seed"] = int(seed_total_freq)     # “自身为总频次”
            G_ego.nodes[seed]["avg_year_with_seed"] = seed_avg_pub         # “自身为平均发文时间”
            G_ego.nodes[seed]["duration_with_seed"] = seed_span            # “自身为活跃时长”

            # -------------------------
            # 3) Export graph + CSVs
            # -------------------------
            suffix_star = "_STAR" if EXPORT_STAR_EDGES_ONLY else ""
            base_prefix = os.path.join(PATH_OUTDIR, f"ego_{seed}_o{order}_{t_label}_S5{MIN_PAPER_COUNT}_S2{suffix_star}")

            graph_path = f"{base_prefix}.{GRAPH_FORMAT}"
            export_gephi_graph(G_ego, graph_path, GRAPH_FORMAT)
            print(f"[OK] Gephi graph exported: {graph_path}")

            edges_csv_path = f"{base_prefix}_edge_metrics.csv"
            edges_df.to_csv(edges_csv_path, index=False, encoding="utf-8-sig")
            print(f"[OK] Edge metrics CSV exported: {edges_csv_path}")

            all_edges_for_plot.append(edges_df)

    # consolidated plotting CSV
    if all_edges_for_plot:
        all_df = pd.concat(all_edges_for_plot, ignore_index=True)
        all_path = os.path.join(PATH_OUTDIR, "ALL_edge_metrics_for_plot.csv")
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Consolidated plotting CSV exported: {all_path}")
    else:
        print("\n[WARN] No ego exported; consolidated CSV not generated.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
