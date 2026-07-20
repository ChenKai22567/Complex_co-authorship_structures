# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_stats_advanced_v6_final.py

==============================================================================
【功能详解与使用说明】
==============================================================================

1. 核心目标：
   基于作者-论文关联数据（CSV），构建合著网络，执行多种过滤算法，并统计网络指标。
   支持时间切片分析（纵向演化）和多方法对比（横向对比）。

2. 数据输入：
   - 格式：CSV 文件。
   - 必须列：author_id (作者ID), paper_id (论文ID)。
   - 可选列：publication_year (年份，用于时间切片), times_cited (被引，用于计算属性)。

3. 网络构建方法 (Counting Methods)：
   - Full Counting (全计数): 两人合著一次，边权重+1。
   - Newman (分数计数): 边权重 = 1 / (该论文作者数 - 1)。用于降低多作者论文的影响。
   - Jaccard (杰卡德相似度): 衡量合作紧密度，权重 = 合作篇数 / (A总篇数 + B总篇数 - 合作篇数)。

4. 过滤算法体系 (Filter Pipeline)：
   程序支持自定义“过滤管道”，按顺序执行以下算法：
   - [S1] 去除孤立节点 (Remove Isolates): 剔除没有连边的节点。
   - [S2] 最大连通分量 (LCC): 仅保留网络中节点数最多的那个连通子图。
   - [S3] 差异过滤器 (Disparity Filter): 基于统计显著性的主干提取算法（公式 3.7）。
          [修正] 采用 AND 逻辑：仅当边两端节点的 Alpha 值均小于阈值时保留该边。
   - [S5] 发文量过滤 (Min Paper Count): 剔除在当前时间窗内发文量低于阈值的作者。

5. 统计指标输出 (新增指标*):
   - 基础指标：节点数、边数、密度、平均度。
   - *加权指标：平均加权度 (Average Weighted Degree)。
   - 结构指标：连通分量数量、子网规模分布。
   - 覆盖指标：覆盖文章数、网络作者平均发文量、覆盖文章篇均被引。
   - 属性统计：节点在当前切片内的总发文量、被引量、活跃年份。
   - 分布统计：网络内作者发文量的分布情况。

6. 输出文件：
   - 图文件 (.gexf/.graphml): 可导入 Gephi/VOSviewer 可视化。
   - 汇总表 (.csv): 包含所有切片、所有方法的统计指标汇总。

==============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx

# 尝试导入 SciPy
try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# 1. 全局配置区 (User Configuration)
# =========================

# 输入文件路径
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
# 输出目录路径
PATH_OUTDIR = str(_OUTPUT_ROOT / "02_network_construction" / "coauthorship_network_pipeline")

# CSV 列名映射配置
AUTHOR_COL = "author_id"
PAPER_COL = "paper_id"

# 自动识别年份列的候选名单
YEAR_COL_CANDIDATES = ["publication_year", "year", "pub_year", "PY", "publication date"]
# 自动识别被引次数列的候选名单
CITE_COL_CANDIDATES = ["times_cited", "Times Cited, WoS Core", "Times Cited", "TC"]

# --- 核心运行参数 ---

# 1. 计数方法选择
# 可选: ["full", "newman", "jaccard"]
METHODS_TO_RUN = ["full", "newman", "jaccard"]

# 2. 过滤方案列表
FILTER_CONFIGS = [
    ["S1"],  # 方案 A: 仅去孤立
    ["S2"],  # 方案 B: 仅最大连通
    ["S3", "S1"],  # 方案 C: 差异过滤 -> 去孤立
    ["S5", "S1"],  # 方案 D: 发文过滤 -> 去孤立
    ["S5", "S2"],  # 方案 E: 发文过滤 -> 最大连通
    ["S5", "S3", "S2"]  # 方案 F: 发文过滤 -> 差异过滤 -> 最大连通
]

# 3. [S3] 差异过滤参数 (Alpha)
S3_ALPHA = 0.1

# 4. [S5] 发文量过滤参数 (Min Count)
MIN_PAPER_COUNT = 2

# 5. [S4] 时间切片配置
TIME_START = 2006
TIME_END = 2025
TIME_STEP = 20

# 6. 导出设置
EXPORT_GRAPHS = True
GRAPH_FORMAT = "gexf"


# =========================
# 工具函数
# =========================

def ensure_outdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                return df_col
    return ""


def load_data(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"文件未找到: {csv_path}")

    print(f"正在读取文件: {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)

    year_col = detect_column(df, YEAR_COL_CANDIDATES)
    cite_col = detect_column(df, CITE_COL_CANDIDATES)

    df = df.dropna(subset=[AUTHOR_COL, PAPER_COL])
    df = df[(df[AUTHOR_COL].str.strip() != "") & (df[PAPER_COL].str.strip() != "")]

    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)
    if cite_col:
        df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0).astype(int)

    return df, year_col, cite_col


def build_index_maps(df_slice):
    author_ids = df_slice[AUTHOR_COL].unique()
    paper_ids = df_slice[PAPER_COL].unique()

    aid2idx = {a: i for i, a in enumerate(author_ids)}
    pid2idx = {p: i for i, p in enumerate(paper_ids)}

    df_idx = df_slice.copy()
    df_idx["ai"] = df_idx[AUTHOR_COL].map(aid2idx).astype(int)
    df_idx["pk"] = df_idx[PAPER_COL].map(pid2idx).astype(int)

    return author_ids, paper_ids, df_idx


# =========================
# 矩阵构图
# =========================

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
        "src_author_id": [author_ids[i] for i in rows],
        "dst_author_id": [author_ids[j] for j in cols],
        "weight": data.astype(float)
    })


def build_network_matrix(method: str, author_ids, paper_ids, df_idx):
    if not HAVE_SCIPY:
        raise RuntimeError("未安装 SciPy 库，无法进行矩阵运算。")

    A = sp.csr_matrix(
        (np.ones(len(df_idx), dtype=np.int8), (df_idx["ai"], df_idx["pk"])),
        shape=(len(author_ids), len(paper_ids))
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
        raise ValueError(f"未知的计数方法: {method}")

    if method != "jaccard":
        U = zero_diag_csr_inplace(U.tocsr())

    return edges_from_sparse(U, author_ids)


# =========================
# 过滤算法
# =========================

def filter_isolates_s1(G):
    """[S1] 去除孤立节点"""
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    return G


def filter_lcc_s2(G):
    """[S2] 仅保留最大连通分量"""
    if G.number_of_nodes() == 0: return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def filter_disparity_s3(G, alpha_threshold=0.1):
    """
    [S3] 差异过滤器 (Disparity Filter)
    公式: alpha_ij = (1 - p_ij)^(k-1)
    [修正] 判定逻辑: 使用 AND 逻辑。
    """
    if G.number_of_edges() == 0: return G

    strength = {n: 0.0 for n in G.nodes()}
    degree = {n: 0 for n in G.nodes()}

    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        strength[u] += w
        strength[v] += w
        degree[u] += 1
        degree[v] += 1

    edges_to_remove = []

    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)

        # 计算 u 端
        k_u = degree[u]
        if k_u > 1:
            p_uv = w / strength[u]
            val = max(0.0, 1.0 - p_uv)
            alpha_u = val ** (k_u - 1)
        else:
            alpha_u = 0.0

        # 计算 v 端
        k_v = degree[v]
        if k_v > 1:
            p_vu = w / strength[v]
            val = max(0.0, 1.0 - p_vu)
            alpha_v = val ** (k_v - 1)
        else:
            alpha_v = 0.0

        # [修正] 必须两端均显著 (AND)
        significant = (alpha_u < alpha_threshold) or (alpha_v < alpha_threshold)

        if not significant:
            edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)
    # 移除因删边产生的孤立点
    G = filter_isolates_s1(G)

    print(f"      [S3] Alpha={alpha_threshold}: Removed {len(edges_to_remove)} edges.")
    return G


def filter_min_papers_s5(G, paper_counts, min_count):
    """[S5] 按作者发文数量过滤"""
    nodes_to_remove = [n for n in G.nodes() if paper_counts.get(n, 0) < min_count]
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
        print(f"      [S5] MinPapers={min_count}: Removed {len(nodes_to_remove)} nodes.")
    return G


def run_filter_pipeline(G, pipeline_list, alpha_val, paper_counts, min_paper_count):
    """管道执行器"""
    G_curr = G.copy()
    for step in pipeline_list:
        if step == "S1":
            G_curr = filter_isolates_s1(G_curr)
        elif step == "S2":
            G_curr = filter_lcc_s2(G_curr)
        elif step == "S3":
            G_curr = filter_disparity_s3(G_curr, alpha_threshold=alpha_val)
        elif step == "S5":
            G_curr = filter_min_papers_s5(G_curr, paper_counts, min_paper_count)
        else:
            print(f"      [Warning] 未知的过滤步骤: {step}，已跳过。")
    return G_curr


# =========================
# 统计指标计算
# =========================

def compute_node_attributes(df_slice, G, year_col, cite_col):
    """计算节点属性写入图文件"""
    grouped = df_slice.groupby(AUTHOR_COL)
    stats_count = grouped.size().to_dict()

    stats_min_yr = {}
    stats_avg_yr = {}
    if year_col:
        valid = df_slice[df_slice[year_col] > 0]
        if not valid.empty:
            g_time = valid.groupby(AUTHOR_COL)[year_col]
            stats_min_yr = g_time.min().to_dict()
            stats_avg_yr = g_time.mean().to_dict()

    stats_cite = {}
    if cite_col:
        stats_cite = grouped[cite_col].sum().to_dict()

    attrs = {}
    for n in G.nodes():
        attrs[n] = {
            "papers_count": int(stats_count.get(n, 0)),
            "total_cites": int(stats_cite.get(n, 0)),
            "min_year": int(stats_min_yr.get(n, 0)),
            "avg_year": float(round(stats_avg_yr.get(n, 0), 1))
        }
    return attrs


def compute_coverage_stats(G, df_slice, cite_col):
    """计算高级覆盖指标"""
    nodes = list(G.nodes())
    if not nodes:
        return 0, 0.0, 0.0

    # 1. 计算网络中作者的平均发文量
    slice_counts = df_slice[AUTHOR_COL].value_counts().to_dict()
    total_papers_of_nodes = sum(slice_counts.get(n, 0) for n in nodes)
    avg_author_papers = total_papers_of_nodes / len(nodes)

    # 2. 计算覆盖文章数 & 篇均被引
    valid_edges = set(frozenset((u, v)) for u, v in G.edges())

    if not valid_edges:
        return 0, avg_author_papers, 0.0

    paper_cites_map = {}
    if cite_col:
        temp = df_slice[[PAPER_COL, cite_col]].drop_duplicates(subset=[PAPER_COL])
        paper_cites_map = temp.set_index(PAPER_COL)[cite_col].to_dict()

    paper_groups = df_slice.groupby(PAPER_COL)[AUTHOR_COL].apply(list)

    covered_papers_count = 0
    covered_papers_total_cites = 0

    for pid, authors in paper_groups.items():
        if len(authors) < 2:
            continue
        relevant_authors = [a for a in authors if a in G]
        if len(relevant_authors) < 2:
            continue

        is_covered = False
        for i in range(len(relevant_authors)):
            for j in range(i + 1, len(relevant_authors)):
                if frozenset((relevant_authors[i], relevant_authors[j])) in valid_edges:
                    is_covered = True
                    break
            if is_covered:
                break

        if is_covered:
            covered_papers_count += 1
            covered_papers_total_cites += paper_cites_map.get(pid, 0)

    avg_paper_cites = 0.0
    if covered_papers_count > 0:
        avg_paper_cites = covered_papers_total_cites / covered_papers_count

    return covered_papers_count, avg_author_papers, avg_paper_cites


def get_component_stats(G):
    if G.number_of_nodes() == 0: return [], {}
    comps = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in comps], reverse=True)
    dist = {}
    for s in sizes:
        dist[s] = dist.get(s, 0) + 1
    return sizes, dist


def get_paper_dist(nodes, global_counts):
    if not nodes: return {}
    counts = [global_counts.get(n, 0) for n in nodes]
    if not counts: return {}
    s = pd.Series(counts)
    vc = s.value_counts().sort_index()
    norm = s.value_counts(normalize=True).sort_index()
    res = {}
    for k in vc.index:
        res[int(k)] = {"count": int(vc[k]), "ratio": round(float(norm[k]), 4)}
    return res


# =========================
# 主程序
# =========================

def main():
    print(">>> 启动网络分析程序...")
    ensure_outdir(PATH_OUTDIR)

    # 1. 加载数据
    df_all, year_col, cite_col = load_data(PATH_INPUT)
    print(f"    数据总行数: {len(df_all)}")

    slices = []
    if year_col and TIME_STEP < 9999:
        curr = TIME_START
        while curr <= TIME_END:
            end = min(curr + TIME_STEP - 1, TIME_END)
            slices.append((curr, end))
            curr += TIME_STEP
    else:
        slices = [("ALL", "ALL")]

    summary_rows = []

    for (t_s, t_e) in slices:
        t_label = f"{t_s}-{t_e}"
        print(f"\n====== 处理时间切片: {t_label} ======")

        if t_s == "ALL":
            df_slice = df_all
        else:
            df_slice = df_all[(df_all[year_col] >= t_s) & (df_all[year_col] <= t_e)]

        if df_slice.empty:
            print("    [跳过] 当前时间段无数据。")
            continue

        auth_ids, paper_ids, df_idx = build_index_maps(df_slice)
        print(f"    包含作者数: {len(auth_ids)}, 论文数: {len(paper_ids)}")

        slice_author_counts = df_slice[AUTHOR_COL].value_counts().to_dict()

        for method in METHODS_TO_RUN:
            print(f"  >>> 构图方法: {method.upper()}")

            edges = build_network_matrix(method, auth_ids, paper_ids, df_idx)
            G_base = nx.Graph()
            for r in edges.itertuples(index=False):
                G_base.add_edge(r.src_author_id, r.dst_author_id, weight=r.weight)

            for pipeline in FILTER_CONFIGS:
                pipeline_name = "-".join(pipeline)
                print(f"    -> 执行过滤组合: [{pipeline_name}]")

                G_final = run_filter_pipeline(
                    G_base,
                    pipeline,
                    alpha_val=S3_ALPHA,
                    paper_counts=slice_author_counts,
                    min_paper_count=MIN_PAPER_COUNT
                )

                nodes_n = G_final.number_of_nodes()
                if nodes_n > 0:
                    # 1. 计算节点属性
                    attrs = compute_node_attributes(df_slice, G_final, year_col, cite_col)
                    nx.set_node_attributes(G_final, attrs)

                    # 2. 基础指标
                    dens = nx.density(G_final)
                    avg_d = (2 * G_final.number_of_edges()) / nodes_n

                    # [新增] 平均加权度计算
                    total_weight = sum(d.get('weight', 0.0) for u, v, d in G_final.edges(data=True))
                    avg_wd = (2 * total_weight) / nodes_n

                    # 3. 计算覆盖文章数、平均发文、篇均被引
                    cov_papers, avg_auth_p, avg_pap_c = compute_coverage_stats(G_final, df_slice, cite_col)

                    # 4. 子网统计
                    comp_sizes, comp_dist = get_component_stats(G_final)

                    # 5. 发文分布
                    paper_dist = get_paper_dist(list(G_final.nodes()), slice_author_counts)

                    if EXPORT_GRAPHS:
                        fname = f"coauthorship_{method}_{pipeline_name.lower().replace('-', '_')}_{str(t_label).replace('-', '_')}.{GRAPH_FORMAT}"
                        out_p = os.path.join(PATH_OUTDIR, fname)
                        if GRAPH_FORMAT == "gexf":
                            nx.write_gexf(G_final, out_p)
                        else:
                            nx.write_graphml(G_final, out_p)

                    summary_rows.append({
                        "slice": t_label,
                        "method": method,
                        "pipeline": pipeline_name,
                        "param_alpha": S3_ALPHA if 'S3' in pipeline else None,
                        "param_min_papers": MIN_PAPER_COUNT if 'S5' in pipeline else None,
                        "nodes": nodes_n,
                        "edges": G_final.number_of_edges(),
                        "density": round(dens, 6),
                        "avg_degree": round(avg_d, 4),
                        # 新增：平均加权度
                        "avg_weighted_degree": round(avg_wd, 4),
                        # 覆盖指标
                        "covered_papers_count": cov_papers,
                        "avg_author_paper_count": round(avg_auth_p, 2),
                        "avg_paper_citations": round(avg_pap_c, 2),
                        # 结构指标
                        "num_components": len(comp_sizes),
                        "component_sizes_list": json.dumps(comp_sizes),
                        "component_size_dist": json.dumps(comp_dist),
                        "paper_count_dist": json.dumps(paper_dist)
                    })
                else:
                    print("      [结果] 过滤后网络为空。")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_f = os.path.join(PATH_OUTDIR, "coauthorship_network_summary.csv")
        df_sum.to_csv(out_f, index=False, encoding="utf-8-sig")
        print(f"\n>>> 全部完成。统计汇总已保存至: {out_f}")
    else:
        print("\n>>> 未生成任何有效网络图。")


if __name__ == "__main__":
    main()
