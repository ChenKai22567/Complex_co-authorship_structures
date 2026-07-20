# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
V4 Final: S5/S2 Dual Pipeline + Advanced Semantic Analysis (TF-IDF & Frequency).

功能详细说明：
1. 数据处理：
   - 读取数据，清洗年份。
   - 提取作者属性：Label (真实姓名 raw_name) 和 Keywords (合并去重)。

2. 双管道网络构建 (Dual Pipeline)：
   - [Pipeline A - Calc]: 核心计算网络。经过 S5(发文>=2) 和 S2(最大连通) 过滤。
     用于运行 Infomap 算法，确保社区划分由核心骨干结构决定，不受边缘噪声干扰。
   - [Pipeline B - View]: 全局展示网络。仅经过 S2 过滤。
     保留了依附于核心结构的边缘作者（发文量=1），用于最终可视化展示。

3. 高级语义分析 (Advanced Semantics)：
   - 针对每个社区，计算三组指标辅助命名：
     a. [TF-IDF]: 最具区分度的关键词（降低通用词权重，突显社区特色）。
     b. [Freq]:   最高频的关键词（显示该社区讨论最多的热门话题）。
     c. [Hubs]:   社区内的核心作者（基于子图度中心性 Top 3）。

4. 映射与导出：
   - 将 Pipeline A 计算出的社区 ID 映射回 Pipeline B 的节点。
   - 核心逻辑：Pipeline B 中存在但 Pipeline A 中被过滤的节点，社区 ID 设为 0。
   - 导出 GEXF 和 CSV，包含 Label, Keywords, Community_L1/L2。

依赖库：
pip install infomap networkx pandas numpy scipy
"""

import os
import math
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
import infomap
from networkx.algorithms.community.quality import modularity

try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# 1. 全局配置
# =========================
PATH_INPUT = str(_OUTPUT_ROOT / "01_data_preparation" / "author_disambiguation" / "author_disambiguation_records.csv")
PATH_OUTDIR = str(_OUTPUT_ROOT / "04_community_analysis" / "infomap_community_semantics")

TARGET_YEAR = 2025

# [Pipeline A] 计算用：严格过滤
PIPELINE_CALC = ["S5", "S2"]

# [Pipeline B] 展示用：宽泛过滤
PIPELINE_VIEW = ["S2"]

LEVELS_TO_SAVE = [1, 2]
NUM_TRIALS = 1000
SEED = 42
INFOMAP_ARGS = f"--markov-time 1.1 -N {NUM_TRIALS} --seed {SEED}"


# =========================
# 2. 基础工具
# =========================
def ensure_outdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(csv_path: str):
    print(f"Reading data: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    year_col = next((c for c in df.columns if c in ["publication_year", "year", "pub_year", "PY"]), None)
    if year_col is None:
        raise ValueError("Cannot find year column.")
    df = df.dropna(subset=["author_id", "paper_id"])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)
    return df, year_col


def extract_author_attributes(df: pd.DataFrame):
    """
    提取作者属性：
    1. id2name: ID -> 真实姓名
    2. id2keywords: ID -> 关键词字符串 (分号分隔)
    """
    print("Extracting author attributes (Names & Keywords)...")
    id2name = df.groupby("author_id")["raw_name"].first().to_dict()

    df_kw = df[["author_id", "keywords"]].dropna()

    def aggregate_keywords(series):
        all_kws = []
        for item in series:
            if not isinstance(item, str): continue
            # 兼容中文分号和英文分号
            tokens = item.replace("；", ";").split(";")
            tokens = [t.strip().lower() for t in tokens if t.strip()]
            all_kws.extend(tokens)
        if not all_kws:
            return ""
        # 去重并排序，保持整洁
        return "; ".join(sorted(list(set(all_kws))))

    id2keywords = df_kw.groupby("author_id")["keywords"].apply(aggregate_keywords).to_dict()
    return id2name, id2keywords


def build_network_from_df(df_slice: pd.DataFrame) -> nx.Graph:
    """构建加权投影网络 (Newman Weighting)"""
    if not HAVE_SCIPY:
        raise RuntimeError("Need SciPy installed.")

    a_ids = df_slice["author_id"].unique()
    p_ids = df_slice["paper_id"].unique()

    aid2idx = {a: i for i, a in enumerate(a_ids)}
    pid2idx = {p: i for i, p in enumerate(p_ids)}

    row_ind = df_slice["author_id"].map(aid2idx).astype(int)
    col_ind = df_slice["paper_id"].map(pid2idx).astype(int)

    # A: Author x Paper
    A = sp.csr_matrix((np.ones(len(df_slice), dtype=np.float32), (row_ind, col_ind)),
                      shape=(len(a_ids), len(p_ids)))

    n = np.asarray(A.sum(axis=0)).ravel()
    w = np.zeros_like(n, dtype=np.float32)
    mask = n > 1
    w[mask] = 1.0 / (n[mask] - 1.0)

    U = A @ sp.diags(w) @ A.T
    U.setdiag(0)
    U.eliminate_zeros()

    G = nx.from_scipy_sparse_array(U)
    mapping = {i: aid for i, aid in enumerate(a_ids)}
    G = nx.relabel_nodes(G, mapping)
    return G


def filter_pipeline(G: nx.Graph, pipeline, global_counts: dict, label: str) -> nx.Graph:
    """通用过滤器"""
    G_curr = G.copy()
    print(f"[{label}] Pipeline: {pipeline}...")
    for step in pipeline:
        if step == "S2":
            if len(G_curr) > 0 and nx.number_connected_components(G_curr) > 1:
                largest_cc = max(nx.connected_components(G_curr), key=len)
                G_curr = G_curr.subgraph(largest_cc).copy()
                print(f"  [{label} - S2] Kept Giant Component: {len(G_curr)} nodes.")
        elif step == "S5":
            rem = [n for n in G_curr.nodes() if global_counts.get(n, 0) < 2]
            G_curr.remove_nodes_from(rem)
            print(f"  [{label} - S5] Removed {len(rem)} nodes (counts < 2).")
    return G_curr


# =========================
# 3. Infomap & 社区构建
# =========================
def run_infomap_get_items(G: nx.Graph):
    print(f"  -> Running Infomap on Strict Graph ({INFOMAP_ARGS})...")
    im = infomap.Infomap(INFOMAP_ARGS)
    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    int_to_node = {i: n for n, i in node_to_int.items()}

    for u, v, d in G.edges(data=True):
        im.add_link(node_to_int[u], node_to_int[v], float(d.get("weight", 1.0)))
    im.run()

    if hasattr(im, "multilevel_modules"):
        items = list(im.multilevel_modules)
    elif hasattr(im, "get_multilevel_modules"):
        d = im.get_multilevel_modules(states=False)
        items = list(d.items())
    else:
        raise RuntimeError("Infomap version incompatible.")

    return im, items, int_to_node


def build_partition(items, int_to_node, level: int):
    """构建社区划分字典 {node: comm_id}"""
    partition = {}
    path_to_id = {}
    next_id = 1
    for node_id, mods in items:
        if mods is None: continue
        target_path = tuple(mods[:min(level, len(mods))])
        if target_path not in path_to_id:
            path_to_id[target_path] = next_id
            next_id += 1
        partition[int_to_node[int(node_id)]] = path_to_id[target_path]
    return partition, len(path_to_id)


# =========================
# 4. 高级语义分析 (Advanced Semantics)
# =========================
def analyze_community_semantics_advanced(G: nx.Graph, partition: dict,
                                         id2keywords: dict, id2name: dict,
                                         top_k_comm=135):
    """
    [混合分析]
    1. TF-IDF: 识别最具区分度的词（Unique Topics）
    2. Frequency: 识别最热门的词（Popular Topics）
    3. Hub Authors: 识别社区核心人物（Authority）
    """
    print(f"\n  >> [Advanced Analysis] Generating Community Profiles (Top {top_k_comm})...")

    # 1. 准备文档集
    comm_docs = {}  # cid -> [all keywords list]
    comm_members = {}  # cid -> [member ids]

    for aid, cid in partition.items():
        comm_members.setdefault(cid, []).append(aid)
        kws_str = id2keywords.get(aid, "")
        if kws_str:
            words = [w.strip().lower() for w in kws_str.split(";") if w.strip()]
            comm_docs.setdefault(cid, []).extend(words)

    # 2. 计算全局 DF (用于 IDF)
    doc_freq = {}
    num_comms = len(comm_docs)
    for cid, words in comm_docs.items():
        unique_words = set(words)
        for w in unique_words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    # 3. 排序并输出
    sorted_comms = sorted(comm_members.items(), key=lambda x: len(x[1]), reverse=True)

    for cid, members in sorted_comms[:top_k_comm]:
        words = comm_docs.get(cid, [])
        if not words:
            print(f"    Comm {cid}: No keywords.")
            continue

        # --- A. 计算统计指标 ---
        tf_counter = Counter(words)
        total_words = len(words)

        # TF-IDF List
        tfidf_scores = []
        for w, count in tf_counter.items():
            tf = count / total_words
            # 平滑 IDF: log(N / (df + 1)) + 1
            idf = math.log(num_comms / (doc_freq.get(w, 0) + 1)) + 1
            tfidf_scores.append((w, tf * idf))

        # Get Top Lists
        top_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:6]
        top_freq = tf_counter.most_common(6)

        tfidf_str = ", ".join([f"{w}" for w, s in top_tfidf])
        freq_str = ", ".join([f"{w}({c})" for w, c in top_freq])

        # --- B. 识别核心作者 (子图度中心性) ---
        comm_subgraph = G.subgraph(members)
        degrees = dict(comm_subgraph.degree(weight='weight'))
        top_authors_ids = sorted(degrees, key=degrees.get, reverse=True)[:4]
        top_authors_names = [str(id2name.get(aid, aid)) for aid in top_authors_ids]
        authors_str = ", ".join(top_authors_names)

        print(f"    --------------------------------------------------")
        print(f"    Community {cid} | Size: {len(members)}")
        print(f"      [Distinctive (TF-IDF)]: {tfidf_str}")
        print(f"      [Popular (Freq)]:       {freq_str}")
        print(f"      [Core Authors]:         {authors_str}")


# =========================
# 5. 映射与导出
# =========================
def export_mapped_gexf(G_view: nx.Graph, partitions_by_level: dict,
                       id2name: dict, id2keywords: dict,
                       out_dir: str, year: int):
    """映射社区ID，导出宽泛网络"""

    # 1. 映射 Community ID (不存在则为0)
    for lvl, part in partitions_by_level.items():
        attr_dict = {n: part.get(n, 0) for n in G_view.nodes()}
        nx.set_node_attributes(G_view, attr_dict, name=f"community_L{lvl}")

    # 2. 映射属性
    kw_attr = {n: id2keywords.get(n, "") for n in G_view.nodes()}
    nx.set_node_attributes(G_view, kw_attr, name="keywords")

    label_attr = {n: str(id2name.get(n, n)) for n in G_view.nodes()}
    nx.set_node_attributes(G_view, label_attr, name="label")

    # 3. 导出 GEXF
    out_gexf = os.path.join(out_dir, f"net_S2_hybrid_mapped_{year}.gexf")
    nx.write_gexf(G_view, out_gexf)

    # 4. 导出 CSV
    rows = []
    lvls_sorted = sorted(partitions_by_level.keys())
    for n in G_view.nodes():
        row = {
            "Id": n,
            "Label": label_attr[n],
            "Keywords": kw_attr[n]
        }
        for lvl in lvls_sorted:
            row[f"community_L{lvl}"] = int(partitions_by_level[lvl].get(n, 0))

        # 标记是否为核心节点
        is_core = int(partitions_by_level[lvls_sorted[0]].get(n, 0)) > 0
        row["is_core_node"] = 1 if is_core else 0

        rows.append(row)

    out_csv = os.path.join(out_dir, f"nodes_S2_hybrid_mapped_{year}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nExport Success:")
    print(f"  GEXF: {out_gexf}")
    print(f"  CSV:  {out_csv}")


# =========================
# 6. 主程序
# =========================
def main():
    ensure_outdir(PATH_OUTDIR)

    # 1. 数据准备
    df, year_col = load_data(PATH_INPUT)
    id2name, id2keywords = extract_author_attributes(df)

    print(f"\nProcessing Year: {TARGET_YEAR}")
    df_cum = df[df[year_col] <= TARGET_YEAR].drop_duplicates(subset=["author_id", "paper_id"])
    paper_counts = df_cum["author_id"].value_counts().to_dict()

    print("Building Base Network...")
    G_base = build_network_from_df(df_cum)
    print(f"Base Graph: |V|={G_base.number_of_nodes()}, |E|={G_base.number_of_edges()}")

    # -------------------------------------------------------
    # [Pipeline A] 核心计算 (S5 + S2)
    # -------------------------------------------------------
    print("\n=== Pipeline A: Core Community Calculation ===")
    G_calc = filter_pipeline(G_base, PIPELINE_CALC, paper_counts, label="CALC")

    # Infomap 社区发现
    im, items, int_to_node = run_infomap_get_items(G_calc)

    partitions_by_level = {}
    for lvl in LEVELS_TO_SAVE:
        print(f"\n>>> Level {lvl} Analysis <<<")
        part, n_comms = build_partition(items, int_to_node, level=lvl)
        partitions_by_level[lvl] = part

        # 混合语义分析：传入 G_calc 以计算核心作者
        analyze_community_semantics_advanced(G_calc, part, id2keywords, id2name, top_k_comm=135)

    # -------------------------------------------------------
    # [Pipeline B] 全局展示 (S2 Only)
    # -------------------------------------------------------
    print("\n=== Pipeline B: View Network Generation ===")
    G_view = filter_pipeline(G_base, PIPELINE_VIEW, paper_counts, label="VIEW")

    # -------------------------------------------------------
    # 映射与导出
    # -------------------------------------------------------
    print("\n=== Mapping & Exporting ===")
    export_mapped_gexf(G_view, partitions_by_level, id2name, id2keywords, PATH_OUTDIR, TARGET_YEAR)


if __name__ == "__main__":
    main()
