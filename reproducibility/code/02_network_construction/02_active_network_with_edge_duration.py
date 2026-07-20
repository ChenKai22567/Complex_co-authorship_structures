# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_extract_s5s2_metrics_v5_with_edge_duration.py

目标：
- 构建合著网络（矩阵法，支持 newman/full/jaccard）
- 过滤：S5 -> S2（发文量过滤 -> 最大连通分量）
- 计算作者属性：论文数、活跃时长、总被引、篇均被引、h-index、g-index
- ⭐ 新增：边持续时间（两位作者首次和最后一次合作年份之差）
- 输出：图文件（gexf/graphml）+ nodes.csv + edges.csv（含 edge_first_year/edge_last_year/edge_duration）
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    # 如果没有 tqdm，就退化为普通迭代，不影响运行
    def tqdm(x, **kwargs):
        return x

try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# 配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration")

AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"

YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]
CITE_COL_CANDIDATES = ["times_cited", "Times Cited, WoS Core", "Times Cited", "TC"]

METHODS_TO_RUN = ["newman"]  # 你现在用 newman
GRAPH_FORMAT = "gexf"        # "gexf" or "graphml"
EXPORT_GRAPHS = True

# 过滤：S5 -> S2
MIN_PAPER_COUNT = 2

# 时间切片
TIME_START = 2006
TIME_END = 2025
TIME_STEP = 20  # 20 表示 2006-2025 一个窗口；要滚动就改小


# =========================
# 工具函数
# =========================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                return df_col
    return ""

def load_and_clean(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"文件未找到: {csv_path}")

    print(f"正在读取文件: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    year_col = detect_column(df, YEAR_COL_CANDIDATES)
    cite_col = detect_column(df, CITE_COL_CANDIDATES)

    # 保留必要列，减内存
    keep_cols = [AUTHOR_COL, PAPER_COL]
    if year_col: keep_cols.append(year_col)
    if cite_col: keep_cols.append(cite_col)
    df = df[keep_cols].copy()

    # 清洗空值
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]

    # 年份/被引转数值
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
    if cite_col:
        df[cite_col] = pd.to_numeric(df[cite_col], errors="coerce").fillna(0).astype(int)

    # ⭐ 去重/聚合：每个 (author,paper) 只保留一条，避免地址/机构重复导致重复计数
    agg = {}
    if year_col: agg[year_col] = "max"
    if cite_col: agg[cite_col] = "max"
    if agg:
        df = df.groupby([AUTHOR_COL, PAPER_COL], as_index=False).agg(agg)
    else:
        df = df.drop_duplicates([AUTHOR_COL, PAPER_COL])

    return df, year_col, cite_col

def build_index_maps_fast(df_slice: pd.DataFrame):
    """
    用 factorize 替代 unique+dict 映射，更快更省内存
    """
    ai, author_ids = pd.factorize(df_slice[AUTHOR_COL], sort=False)
    pk, paper_ids = pd.factorize(df_slice[PAPER_COL], sort=False)
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

def build_network_matrix(method: str, author_ids, paper_ids, df_idx):
    if not HAVE_SCIPY:
        raise RuntimeError("未安装 SciPy，无法进行矩阵构图。")

    nA, nP = len(author_ids), len(paper_ids)
    A = sp.csr_matrix(
        (np.ones(len(df_idx), dtype=np.int8),
         (df_idx["ai"].to_numpy(), df_idx["pk"].to_numpy())),
        shape=(nA, nP)
    )

    if method == "full":
        U = A @ A.T

    elif method == "newman":
        n_authors = np.asarray(A.sum(axis=0)).ravel()
        w = np.zeros_like(n_authors, dtype=float)
        mask = n_authors > 1
        w[mask] = 1.0 / (n_authors[mask] - 1.0)
        U = A @ sp.diags(w) @ A.T

    elif method == "jaccard":
        C = A @ A.T
        d = np.asarray(A.sum(axis=1)).ravel()
        C = zero_diag_csr_inplace(C.tocsr())
        Ccoo = C.tocoo()
        denom = d[Ccoo.row] + d[Ccoo.col] - Ccoo.data
        valid = denom > 0
        rows, cols = Ccoo.row[valid], Ccoo.col[valid]
        data = (Ccoo.data[valid] / denom[valid]).astype(float)
        U = sp.coo_matrix((data, (rows, cols)), shape=C.shape)

    else:
        raise ValueError(f"未知构图方法: {method}")

    if method != "jaccard":
        U = zero_diag_csr_inplace(U.tocsr())

    return edges_from_sparse(U, author_ids)

def filter_s5_then_s2(G: nx.Graph, paper_counts: dict, min_papers: int):
    # S5：发文量过滤
    keep = [n for n in G.nodes() if paper_counts.get(n, 0) >= min_papers]
    G2 = G.subgraph(keep).copy()
    # S2：最大连通分量
    if G2.number_of_nodes() == 0:
        return G2
    largest = max(nx.connected_components(G2), key=len)
    return G2.subgraph(largest).copy()


# =========================
# 指标：h / g
# =========================
def h_index(cites):
    if len(cites) == 0:
        return 0
    c = np.sort(np.asarray(cites, dtype=int))[::-1]
    h = 0
    for i, val in enumerate(c, start=1):
        if val >= i:
            h = i
        else:
            break
    return h

def g_index(cites):
    if len(cites) == 0:
        return 0
    c = np.sort(np.asarray(cites, dtype=int))[::-1]
    cum = np.cumsum(c)
    g = 0
    for i, s in enumerate(cum, start=1):
        if s >= i * i:
            g = i
        else:
            break
    return g

def compute_author_metrics(df_slice: pd.DataFrame, year_col: str, cite_col: str) -> pd.DataFrame:
    g = df_slice.groupby(AUTHOR_COL, sort=False)

    papers_count = g.size().astype(int)

    if year_col:
        first_year = g[year_col].min().astype(int)
        last_year  = g[year_col].max().astype(int)
        active_span = (last_year - first_year).astype(int)
    else:
        first_year = pd.Series(0, index=papers_count.index)
        last_year = pd.Series(0, index=papers_count.index)
        active_span = pd.Series(0, index=papers_count.index)

    if cite_col:
        total_cites = g[cite_col].sum().astype(int)
        avg_cites = (total_cites / papers_count).replace([np.inf, -np.inf], 0).fillna(0.0)
        cites_list = g[cite_col].apply(lambda s: s.to_numpy())
        h = cites_list.apply(h_index).astype(int)
        gg = cites_list.apply(g_index).astype(int)
    else:
        total_cites = pd.Series(0, index=papers_count.index)
        avg_cites = pd.Series(0.0, index=papers_count.index)
        h = pd.Series(0, index=papers_count.index)
        gg = pd.Series(0, index=papers_count.index)

    out = pd.DataFrame({
        "papers_count": papers_count,
        "first_year": first_year,
        "last_year": last_year,
        "active_span": active_span,
        "total_cites": total_cites,
        "avg_cites": avg_cites.round(4),
        "h_index": h,
        "g_index": gg
    })
    out.index.name = AUTHOR_COL
    return out


# =========================
# ⭐ 新增：计算边持续时间（只对最终 LCC 的边）
# =========================
def compute_edge_duration_on_lcc_edges(
    df_slice: pd.DataFrame,
    year_col: str,
    nodes_set: set,
    edge_set: set
) -> tuple[dict, dict, dict]:
    """
    输入：
    - df_slice：当前切片、已去重的 author-paper 数据
    - year_col：年份列名（必须存在才能算 duration）
    - nodes_set：最终 LCC 的节点集合（只在这些节点上计算边）
    - edge_set：最终 LCC 的边集合（无向，key = (min(u,v), max(u,v))）

    输出：
    - edge_first_year: dict[(u,v)] = first_year
    - edge_last_year : dict[(u,v)] = last_year
    - edge_duration  : dict[(u,v)] = last_year - first_year
    """
    if not year_col:
        # 没有年份就没法算
        return {}, {}, {}

    # 仅保留：年份有效 + 节点在 LCC 内
    dfv = df_slice[df_slice[year_col] > 0].copy()
    dfv = dfv[dfv[AUTHOR_COL].isin(nodes_set)].copy()
    if dfv.empty:
        return {}, {}, {}

    # 用 defaultdict 存最早/最晚
    INF = 10**9
    edge_first = defaultdict(lambda: INF)
    edge_last = defaultdict(lambda: -1)

    # 按论文分组：每篇论文的作者集合 -> 产生作者对
    gp = dfv.groupby(PAPER_COL, sort=False)

    for pid, sub in tqdm(gp, desc="Computing edge duration (paper-wise)", total=gp.ngroups):
        # 该论文年份（如果偶尔不一致，用 max 更稳）
        y = int(sub[year_col].max())

        # 作者列表（该论文上、且在 LCC 内）
        authors = sub[AUTHOR_COL].astype(str).unique()
        if len(authors) < 2:
            continue

        # 排序后组合，保证 (u,v) 唯一化
        authors.sort()

        # 生成 pair：O(k^2)，但 k 通常不大；而且我们只对 LCC 边 edge_set 更新
        k = len(authors)
        for i in range(k - 1):
            u = authors[i]
            for j in range(i + 1, k):
                v = authors[j]
                key = (u, v)  # 因为 authors 已排序，所以 u<v

                # ⭐ 只对最终 LCC 的边更新，避免无效 pair
                if key not in edge_set:
                    continue

                if y < edge_first[key]:
                    edge_first[key] = y
                if y > edge_last[key]:
                    edge_last[key] = y

    # 组装 duration
    edge_first_year = {}
    edge_last_year = {}
    edge_duration = {}
    for key in edge_set:
        fy = edge_first.get(key, INF)
        ly = edge_last.get(key, -1)
        if ly >= 0 and fy < INF:
            edge_first_year[key] = int(fy)
            edge_last_year[key] = int(ly)
            edge_duration[key] = int(ly - fy)  # 你的定义：差值（不 +1）
        else:
            edge_first_year[key] = 0
            edge_last_year[key] = 0
            edge_duration[key] = 0

    return edge_first_year, edge_last_year, edge_duration


# =========================
# 主程序
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)
    df_all, year_col, cite_col = load_and_clean(PATH_INPUT)
    print(f"清洗后行数(作者-论文唯一): {len(df_all)}")
    print(f"识别列: year_col={year_col or 'None'}, cite_col={cite_col or 'None'}")

    # 时间切片
    slices = []
    if year_col and TIME_STEP < 9999:
        curr = TIME_START
        while curr <= TIME_END:
            end = min(curr + TIME_STEP - 1, TIME_END)
            slices.append((curr, end))
            curr += TIME_STEP
    else:
        slices = [("ALL", "ALL")]

    for (t_s, t_e) in slices:
        t_label = f"{t_s}-{t_e}"
        print(f"\n====== 时间切片: {t_label} ======")

        if t_s == "ALL":
            df_slice = df_all
        else:
            df_slice = df_all[(df_all[year_col] >= t_s) & (df_all[year_col] <= t_e)]

        if df_slice.empty:
            print("  [跳过] 当前切片无数据")
            continue

        # 预计算作者发文量（S5）
        slice_author_counts = df_slice[AUTHOR_COL].value_counts().to_dict()

        # 预计算作者指标（一次算完）
        metrics_df = compute_author_metrics(df_slice, year_col, cite_col)

        # 映射索引用于矩阵构图
        author_ids, paper_ids, df_idx = build_index_maps_fast(df_slice)
        print(f"  作者数={len(author_ids)}, 论文数={len(paper_ids)}")

        for method in METHODS_TO_RUN:
            print(f"  >>> 构图方法: {method.upper()}")

            edges = build_network_matrix(method, author_ids, paper_ids, df_idx)

            # 更快建图：from_pandas_edgelist
            G_base = nx.from_pandas_edgelist(
                edges,
                source="src_author_id",
                target="dst_author_id",
                edge_attr="weight",
                create_using=nx.Graph()
            )

            # S5 -> S2
            G_final = filter_s5_then_s2(G_base, slice_author_counts, MIN_PAPER_COUNT)
            print(f"  过滤后: nodes={G_final.number_of_nodes()}, edges={G_final.number_of_edges()}")

            if G_final.number_of_nodes() == 0:
                print("  [结果] 过滤后网络为空")
                continue

            # 写入节点属性
            nodes = list(G_final.nodes())
            sub_metrics = metrics_df.reindex(nodes).fillna(0)

            nx.set_node_attributes(G_final, sub_metrics["papers_count"].to_dict(), "papers_count")
            nx.set_node_attributes(G_final, sub_metrics["active_span"].to_dict(), "active_span")
            nx.set_node_attributes(G_final, sub_metrics["first_year"].to_dict(), "first_year")
            nx.set_node_attributes(G_final, sub_metrics["last_year"].to_dict(), "last_year")
            nx.set_node_attributes(G_final, sub_metrics["total_cites"].to_dict(), "total_cites")
            nx.set_node_attributes(G_final, sub_metrics["avg_cites"].to_dict(), "avg_cites")
            nx.set_node_attributes(G_final, sub_metrics["h_index"].to_dict(), "h_index")
            nx.set_node_attributes(G_final, sub_metrics["g_index"].to_dict(), "g_index")

            # =========================
            # ⭐ 计算边持续时间（只对最终 LCC 边）
            # =========================
            if year_col:
                # 无向边 key 统一化：小的在前
                edge_set = set()
                for u, v in G_final.edges():
                    u = str(u); v = str(v)
                    key = (u, v) if u < v else (v, u)
                    edge_set.add(key)

                nodes_set = set(map(str, G_final.nodes()))

                edge_first_year, edge_last_year, edge_duration = compute_edge_duration_on_lcc_edges(
                    df_slice=df_slice,
                    year_col=year_col,
                    nodes_set=nodes_set,
                    edge_set=edge_set
                )

                # 写入到图的边属性
                for u, v in G_final.edges():
                    uu, vv = (str(u), str(v))
                    key = (uu, vv) if uu < vv else (vv, uu)
                    G_final[u][v]["edge_first_year"] = edge_first_year.get(key, 0)
                    G_final[u][v]["edge_last_year"] = edge_last_year.get(key, 0)
                    G_final[u][v]["edge_duration"] = edge_duration.get(key, 0)
            else:
                print("  [WARN] 未检测到年份列，无法计算 edge_duration，将不输出该属性。")

            # 输出文件
            base_name = f"active_coauthorship_{method}_s5_s2_{str(t_label).replace('-', '_')}"
            if EXPORT_GRAPHS:
                graph_path = os.path.join(PATH_OUTDIR, f"{base_name}.{GRAPH_FORMAT}")
                if GRAPH_FORMAT == "gexf":
                    nx.write_gexf(G_final, graph_path)
                else:
                    nx.write_graphml(G_final, graph_path)
                print(f"  图文件已导出: {graph_path}")

            # 节点表
            nodes_path = os.path.join(PATH_OUTDIR, f"{base_name}_nodes.csv")
            sub_metrics.reset_index().to_csv(nodes_path, index=False, encoding="utf-8-sig")

            # 边表（包含 duration；并给你 dist）
            edges_df = nx.to_pandas_edgelist(G_final)
            edges_df.rename(columns={"source": "src_author_id", "target": "dst_author_id"}, inplace=True)

            # dist：保持你原先定义（用于后续加权最短路）
            edges_df["dist"] = 1.0 / (1.0 + edges_df["weight"].astype(float))

            # 确保三列都存在（即使没年份也不会报错）
            if "edge_first_year" not in edges_df.columns:
                edges_df["edge_first_year"] = 0
            if "edge_last_year" not in edges_df.columns:
                edges_df["edge_last_year"] = 0
            if "edge_duration" not in edges_df.columns:
                edges_df["edge_duration"] = 0

            edges_path = os.path.join(PATH_OUTDIR, f"{base_name}_edges.csv")
            edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")

            print(f"  节点表已导出: {nodes_path}")
            print(f"  边表已导出: {edges_path}")
            print("  [OK] edges.csv 已包含：edge_first_year / edge_last_year / edge_duration")

if __name__ == "__main__":
    main()
