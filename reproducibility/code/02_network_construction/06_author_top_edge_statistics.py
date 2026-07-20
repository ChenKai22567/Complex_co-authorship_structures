# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
author_topn_stats_only.py

功能：
- 构建合著网络（矩阵法，支持 newman/full/jaccard）
- 过滤：S5 -> S2（发文量过滤 -> 最大连通分量）
- 计算边持续时间（两位作者首次和最后一次合作年份之差）
- 仅输出 1 个作者属性表：
    1) 每位作者 top-n 最大权重边及其累计和
    2) 每位作者 top-n 最长持续时间边及其累计和

输出：
- 仅 1 个 CSV 文件
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
# 配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "02_network_construction" / "author_top_edge_statistics")

AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"

YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]

# 构图方法：["newman"] / ["full"] / ["jaccard"]
METHOD = "newman"

# S5 -> S2 过滤
MIN_PAPER_COUNT = 2

# 统计每位作者 top-n 边
TOP_N_EDGES = 3

# 时间过滤（只输出 1 个表，因此这里固定为一个时间窗口）
TIME_START = 2006
TIME_END = 2025

# 输出文件名
OUTPUT_FILENAME = f"author_topn_stats_{METHOD}_S5_{MIN_PAPER_COUNT}_S2_{TIME_START}_{TIME_END}.csv"


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

    keep_cols = [AUTHOR_COL, PAPER_COL]
    if year_col:
        keep_cols.append(year_col)

    df = df[keep_cols].copy()

    # 去掉 author_id / paper_id 缺失
    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[
        (df[AUTHOR_COL].astype(str).str.strip() != "") &
        (df[PAPER_COL].astype(str).str.strip() != "")
    ].copy()

    # year 转数值
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)

    # 每个 (author, paper) 只保留一条，避免地址重复
    if year_col:
        df = df.groupby([AUTHOR_COL, PAPER_COL], as_index=False).agg({year_col: "max"})
    else:
        df = df.drop_duplicates([AUTHOR_COL, PAPER_COL])

    return df, year_col


def build_index_maps_fast(df_slice: pd.DataFrame):
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
        raise RuntimeError("未安装 SciPy，无法进行矩阵构图。请先安装 scipy。")

    nA, nP = len(author_ids), len(paper_ids)

    A = sp.csr_matrix(
        (
            np.ones(len(df_idx), dtype=np.int8),
            (df_idx["ai"].to_numpy(), df_idx["pk"].to_numpy())
        ),
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
    keep = [n for n in G.nodes() if paper_counts.get(n, 0) >= min_papers]
    G2 = G.subgraph(keep).copy()

    if G2.number_of_nodes() == 0:
        return G2

    largest = max(nx.connected_components(G2), key=len)
    return G2.subgraph(largest).copy()


def compute_edge_duration_on_lcc_edges(df_slice: pd.DataFrame, year_col: str, nodes_set: set, edge_set: set):
    """
    只对最终 LCC 中的边计算：
    - edge_first_year
    - edge_last_year
    - edge_duration = last_year - first_year
    """
    if not year_col:
        return {}, {}, {}

    dfv = df_slice[df_slice[year_col] > 0].copy()
    dfv = dfv[dfv[AUTHOR_COL].isin(nodes_set)].copy()

    if dfv.empty:
        return {}, {}, {}

    INF = 10**9
    edge_first = defaultdict(lambda: INF)
    edge_last = defaultdict(lambda: -1)

    gp = dfv.groupby(PAPER_COL, sort=False)

    for _, sub in tqdm(gp, desc="Computing edge duration (paper-wise)", total=gp.ngroups):
        y = int(sub[year_col].max())
        authors = sub[AUTHOR_COL].astype(str).unique()

        if len(authors) < 2:
            continue

        authors.sort()
        k = len(authors)

        for i in range(k - 1):
            u = authors[i]
            for j in range(i + 1, k):
                v = authors[j]
                key = (u, v)
                if key not in edge_set:
                    continue

                if y < edge_first[key]:
                    edge_first[key] = y
                if y > edge_last[key]:
                    edge_last[key] = y

    edge_first_year = {}
    edge_last_year = {}
    edge_duration = {}

    for key in edge_set:
        fy = edge_first.get(key, INF)
        ly = edge_last.get(key, -1)

        if ly >= 0 and fy < INF:
            edge_first_year[key] = int(fy)
            edge_last_year[key] = int(ly)
            edge_duration[key] = int(ly - fy)
        else:
            edge_first_year[key] = 0
            edge_last_year[key] = 0
            edge_duration[key] = 0

    return edge_first_year, edge_last_year, edge_duration


def compute_top_n_edge_stats(G: nx.Graph, top_n: int = 3) -> pd.DataFrame:
    """
    对每个作者统计：
    1) 权重最大的 n 条边：
       - top_weight_1 ... top_weight_n
       - top_weight_sum_1 ... top_weight_sum_n
    2) 持续时间最长的 n 条边：
       - top_duration_1 ... top_duration_n
       - top_duration_sum_1 ... top_duration_sum_n

    规则：
    - 若作者边数少于 n，则缺失部分单条值记 0
    - 累计和按“现有最大前 k 条”计算
    """
    rows = []

    for node in G.nodes():
        weight_list = []
        duration_list = []

        for _, _, data in G.edges(node, data=True):
            w = float(data.get("weight", 0) or 0)
            d = int(data.get("edge_duration", 0) or 0)
            weight_list.append(w)
            duration_list.append(d)

        weight_list.sort(reverse=True)
        duration_list.sort(reverse=True)

        weight_cumsum = np.cumsum(weight_list) if len(weight_list) > 0 else np.array([])
        duration_cumsum = np.cumsum(duration_list) if len(duration_list) > 0 else np.array([])

        row = {
            AUTHOR_COL: node,
            "degree_final": len(weight_list)
        }

        for k in range(1, top_n + 1):
            row[f"top_weight_{k}"] = round(weight_list[k - 1], 6) if len(weight_list) >= k else 0.0
            row[f"top_duration_{k}"] = int(duration_list[k - 1]) if len(duration_list) >= k else 0

            if len(weight_list) > 0:
                idx_w = min(k, len(weight_list)) - 1
                row[f"top_weight_sum_{k}"] = round(float(weight_cumsum[idx_w]), 6)
            else:
                row[f"top_weight_sum_{k}"] = 0.0

            if len(duration_list) > 0:
                idx_d = min(k, len(duration_list)) - 1
                row[f"top_duration_sum_{k}"] = int(duration_cumsum[idx_d])
            else:
                row[f"top_duration_sum_{k}"] = 0

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(by=[AUTHOR_COL]).reset_index(drop=True)
    return out


# =========================
# 主程序
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)

    df_all, year_col = load_and_clean(PATH_INPUT)
    print(f"清洗后行数（author-paper唯一）: {len(df_all)}")
    print(f"识别年份列: {year_col if year_col else 'None'}")

    # 时间过滤：只保留一个时间窗口
    if year_col:
        df_slice = df_all[(df_all[year_col] >= TIME_START) & (df_all[year_col] <= TIME_END)].copy()
    else:
        df_slice = df_all.copy()

    if df_slice.empty:
        raise ValueError("时间过滤后无数据，请检查 TIME_START / TIME_END 或年份列。")

    # S5 用：作者发文量
    slice_author_counts = df_slice[AUTHOR_COL].value_counts().to_dict()

    # 构图索引
    author_ids, paper_ids, df_idx = build_index_maps_fast(df_slice)
    print(f"作者数 = {len(author_ids)}, 论文数 = {len(paper_ids)}")

    # 构图
    edges = build_network_matrix(METHOD, author_ids, paper_ids, df_idx)

    G_base = nx.from_pandas_edgelist(
        edges,
        source="src_author_id",
        target="dst_author_id",
        edge_attr="weight",
        create_using=nx.Graph()
    )

    # S5 -> S2
    G_final = filter_s5_then_s2(G_base, slice_author_counts, MIN_PAPER_COUNT)
    print(f"S5->S2 过滤后：nodes = {G_final.number_of_nodes()}, edges = {G_final.number_of_edges()}")

    if G_final.number_of_nodes() == 0:
        raise ValueError("过滤后网络为空，请检查阈值设置。")

    # 计算边持续时间
    if year_col:
        edge_set = set()
        for u, v in G_final.edges():
            uu = str(u)
            vv = str(v)
            key = (uu, vv) if uu < vv else (vv, uu)
            edge_set.add(key)

        nodes_set = set(map(str, G_final.nodes()))

        edge_first_year, edge_last_year, edge_duration = compute_edge_duration_on_lcc_edges(
            df_slice=df_slice,
            year_col=year_col,
            nodes_set=nodes_set,
            edge_set=edge_set
        )

        for u, v in G_final.edges():
            uu = str(u)
            vv = str(v)
            key = (uu, vv) if uu < vv else (vv, uu)
            G_final[u][v]["edge_first_year"] = edge_first_year.get(key, 0)
            G_final[u][v]["edge_last_year"] = edge_last_year.get(key, 0)
            G_final[u][v]["edge_duration"] = edge_duration.get(key, 0)
    else:
        print("[WARN] 未检测到年份列，edge_duration 将全部按 0 处理。")
        for u, v in G_final.edges():
            G_final[u][v]["edge_first_year"] = 0
            G_final[u][v]["edge_last_year"] = 0
            G_final[u][v]["edge_duration"] = 0

    # 计算每位作者 top-n 边统计
    top_edge_stats_df = compute_top_n_edge_stats(G_final, top_n=TOP_N_EDGES)

    # 输出仅 1 个 CSV
    out_path = os.path.join(PATH_OUTDIR, OUTPUT_FILENAME)
    top_edge_stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] 已输出唯一表格：{out_path}")
    print("表中包含：")
    print(f"- author_id")
    print(f"- degree_final")
    print(f"- top_weight_1 ~ top_weight_{TOP_N_EDGES}")
    print(f"- top_weight_sum_1 ~ top_weight_sum_{TOP_N_EDGES}")
    print(f"- top_duration_1 ~ top_duration_{TOP_N_EDGES}")
    print(f"- top_duration_sum_1 ~ top_duration_sum_{TOP_N_EDGES}")


if __name__ == "__main__":
    main()
