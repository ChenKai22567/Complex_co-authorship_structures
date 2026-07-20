# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
prepare_attack_tables_v3.py

目标（为后续鲁棒性移除实验“打地基”）：
1) 读取 S5->S2 过滤后的合著网络（gexf）
2) 计算/汇总后续要用的节点与边重要性指标
3) 导出：
   - node_list.csv / edge_list.csv：稳定索引（后续随机攻击对 index 洗牌即可）
   - nodes_*.csv / edges_*.csv：确定性策略的“静态移除顺序”（重要->不重要）
   - random_seeds_nodes.csv / random_seeds_edges.csv：随机攻击重复 R 次的 seed（可复现）
   - node_metrics_for_attacks.csv / edge_metrics_for_attacks.csv：指标总表（便于检查/复用）
   - timing_report.csv：耗时统计

随机攻击（Random）是否要重复？
- 建议至少 100 次。这里默认导出 100 个种子，后续移除实验中用这些 seed 生成 100 条随机序列并取均值。

作者：升级版（面向后续移除实验）
"""

import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, List

# =========================
# 0) 路径与全局参数（按你机器实际路径改这里）
# =========================
GRAPH_GEXF = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025.gexf")
EDGES_CSV = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025_edges.csv")
OUT_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "attack_sequences")

# 随机攻击重复次数（推荐 100 起步）
N_RANDOM_RUNS_NODES = 100
N_RANDOM_RUNS_EDGES = 100

# 随机攻击基础 seed（保证可复现）
BASE_RANDOM_SEED = 20260207

# 是否强制重新计算（False：若 OUT_DIR 中已有缓存指标表则直接读取，节省时间）
FORCE_RECOMPUTE = False

# 如果图里没有 edge_duration，且 edges.csv 有 edge_duration，是否用 edges.csv 补齐到图中
FILL_EDGE_DURATION_FROM_EDGESCSV_IF_MISSING = True

# 边介数是否计算“加权距离版”（默认 False：只算无权最短路精确版本）
# 若打开 True：需定义 dist=1/(1+weight)，会更慢，但有时更符合“强合作更近”的直觉
COMPUTE_WEIGHTED_EDGE_BETWEENNESS_WITH_DIST = False


# =========================
# 1) 计时工具（控制台 + 导出）
# =========================
class Timer:
    """用于统计各个模块耗时，并可导出 timing_report.csv"""
    def __init__(self):
        self.records = []
        self._t0 = None
        self._task = None

    def start(self, task: str):
        self._task = task
        self._t0 = time.perf_counter()

    def stop(self):
        sec = time.perf_counter() - self._t0
        self.records.append((self._task, sec))
        print(f"[TIME] {self._task}: {sec:.3f}s")

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["task", "seconds"])


# =========================
# 2) 小工具：权重 -> 距离（用于加权最短路）
# =========================
def weight_to_dist(w: float) -> float:
    """把合作强度 weight 映射为最短路距离 dist：weight 越大 dist 越小"""
    return 1.0 / (1.0 + float(w))


# =========================
# 3) 读取网络 + 清洗/补齐
# =========================
def load_graph(timer: Timer) -> nx.Graph:
    """读取 gexf，并确保是无向、weight 为 float"""
    timer.start("load_graph_gexf")
    if not os.path.exists(GRAPH_GEXF):
        raise FileNotFoundError(f"[ERROR] 未找到网络文件：{GRAPH_GEXF}")
    G = nx.read_gexf(GRAPH_GEXF)
    # 合著网络应为无向图，若读出是有向则转为无向
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    timer.stop()

    # 统一 weight
    timer.start("sanitize_edge_weight")
    for u, v, d in G.edges(data=True):
        if "weight" not in d or d["weight"] in (None, ""):
            d["weight"] = 1.0
        d["weight"] = float(d["weight"])
    timer.stop()

    # 补齐 edge_duration（如果需要）
    if FILL_EDGE_DURATION_FROM_EDGESCSV_IF_MISSING:
        timer.start("fill_edge_duration_if_missing")
        has_dur = any(("edge_duration" in d) for _, _, d in G.edges(data=True))
        if (not has_dur) and os.path.exists(EDGES_CSV):
            df_edges = pd.read_csv(EDGES_CSV)
            if "edge_duration" in df_edges.columns:
                # 用无向 key 对齐 (min,max)
                df_edges["u"] = df_edges["src_author_id"].astype(str)
                df_edges["v"] = df_edges["dst_author_id"].astype(str)
                df_edges["a"] = df_edges[["u", "v"]].min(axis=1)
                df_edges["b"] = df_edges[["u", "v"]].max(axis=1)
                dur_map = dict(zip(zip(df_edges["a"], df_edges["b"]), df_edges["edge_duration"]))

                filled = 0
                for u, v, d in G.edges(data=True):
                    a, b = str(u), str(v)
                    if a > b:
                        a, b = b, a
                    if "edge_duration" not in d:
                        d["edge_duration"] = int(dur_map.get((a, b), 0))
                        filled += 1
                print(f"[OK] edge_duration filled for {filled} edges (missing -> filled).")
            else:
                print("[WARN] edges.csv 中没有 edge_duration 列，duration 将为 0。")
        timer.stop()

    print(f"[OK] Loaded graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    return G


# =========================
# 4) 计算节点指标（面向后续攻击）
# =========================
def compute_node_metrics(G: nx.Graph, timer: Timer) -> pd.DataFrame:
    """
    节点指标：
    - degree
    - strength（加权度：sum(weight)）
    - betweenness（精确，无权最短路）
    - effective_size（结构洞：effective_size）
    """
    nodes = list(G.nodes())
    nodes_str = [str(u) for u in nodes]

    # 度
    timer.start("node_degree")
    deg = dict(G.degree())
    timer.stop()

    # 加权度（强度）
    timer.start("node_strength_weighted_degree")
    strength = {}
    for u in nodes:
        s = 0.0
        for v in G.neighbors(u):
            s += float(G[u][v].get("weight", 1.0))
        strength[u] = s
    timer.stop()

    # 介数中心性（精确，无权最短路）
    timer.start("node_betweenness_exact_unweighted")
    bet = nx.betweenness_centrality(G, normalized=True, weight=None)
    timer.stop()

    # 结构洞 effective_size（NetworkX 内置）
    timer.start("node_effective_size_structural_holes")
    eff = nx.algorithms.structuralholes.effective_size(G, nodes=nodes, weight=None)
    timer.stop()

    df = pd.DataFrame({
        "author_id": nodes_str,
        "degree": [deg[u] for u in nodes],
        "strength": [strength[u] for u in nodes],
        "betweenness": [bet[u] for u in nodes],
        "effective_size": [eff[u] for u in nodes],
    })
    return df


# =========================
# 5) 计算边指标（面向后续攻击）
# =========================
def compute_edge_overlap_jaccard(G: nx.Graph) -> Dict[Tuple[str, str], float]:
    r"""
    边重叠度（Overlap）：Jaccard 形式
    overlap(u,v) = |N(u)\cap N(v)| / |N(u)\cup N(v)|
    值域 [0,1]；越小越像“桥边”，常用于优先攻击 low-overlap。
    """
    # 预构建邻居集合（字符串化）
    neigh = {str(u): set(map(str, G.neighbors(u))) for u in G.nodes()}
    overlap = {}

    for u, v in G.edges():
        a, b = str(u), str(v)
        # 不把对方当作共同邻居，避免人为抬高 overlap
        Nu = neigh[a] - {b}
        Nv = neigh[b] - {a}
        # 用“小集合 & 大集合”加速交并（Python set 交集本身已做优化）
        inter = Nu & Nv
        uni = Nu | Nv
        ov = (len(inter) / len(uni)) if len(uni) > 0 else 0.0
        key = (a, b) if a < b else (b, a)
        overlap[key] = float(ov)

    return overlap


def compute_edge_metrics(G: nx.Graph, timer: Timer) -> pd.DataFrame:
    """
    边指标：
    - weight
    - edge_duration（若无则为 0）
    - edge_betweenness（精确，无权最短路）
    - edge_overlap（Jaccard overlap）
    可选：
    - edge_betweenness_weighted_dist（精确，加权最短路 dist）
    """
    # 先提取边表（统一无向 key）
    edges = []
    for u, v, d in G.edges(data=True):
        a, b = str(u), str(v)
        if a > b:
            a, b = b, a
        w = float(d.get("weight", 1.0))
        dur = int(d.get("edge_duration", 0))
        edges.append((a, b, w, dur))

    df = pd.DataFrame(edges, columns=["src_author_id", "dst_author_id", "weight", "edge_duration"])

    # 边介数（精确，无权最短路）
    timer.start("edge_betweenness_exact_unweighted")
    ebc = nx.edge_betweenness_centrality(G, normalized=True, weight=None)
    timer.stop()

    eb_map = {}
    for (u, v), val in ebc.items():
        a, b = str(u), str(v)
        if a > b:
            a, b = b, a
        eb_map[(a, b)] = float(val)

    df["edge_betweenness"] = [
        eb_map.get((a, b), 0.0) for a, b in zip(df["src_author_id"], df["dst_author_id"])
    ]

    # 可选：加权距离版边介数
    if COMPUTE_WEIGHTED_EDGE_BETWEENNESS_WITH_DIST:
        timer.start("edge_betweenness_exact_weighted_dist")
        # 写入 dist 属性
        for u, v, d in G.edges(data=True):
            d["dist"] = weight_to_dist(d.get("weight", 1.0))
        ebc_w = nx.edge_betweenness_centrality(G, normalized=True, weight="dist")
        timer.stop()

        ebw_map = {}
        for (u, v), val in ebc_w.items():
            a, b = str(u), str(v)
            if a > b:
                a, b = b, a
            ebw_map[(a, b)] = float(val)

        df["edge_betweenness_weighted_dist"] = [
            ebw_map.get((a, b), 0.0) for a, b in zip(df["src_author_id"], df["dst_author_id"])
        ]

    # 边重叠度 overlap
    timer.start("edge_overlap_jaccard")
    ov_map = compute_edge_overlap_jaccard(G)
    timer.stop()

    df["edge_overlap"] = [
        ov_map.get((a, b), 0.0) for a, b in zip(df["src_author_id"], df["dst_author_id"])
    ]

    return df


# =========================
# 6) 导出“稳定索引表”（后续随机攻击只洗牌 index）
# =========================
def export_stable_index_lists(df_node: pd.DataFrame, df_edge: pd.DataFrame, timer: Timer):
    """
    输出 node_list.csv / edge_list.csv：
    - node_list: node_id, author_id
    - edge_list: edge_id, src, dst
    后续随机攻击：对 node_id / edge_id 做 permutation 即可获得一条随机移除序列
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    timer.start("export_node_list_csv")
    node_list = df_node[["author_id"]].copy()
    node_list.insert(0, "node_id", np.arange(len(node_list), dtype=int))
    node_list.to_csv(os.path.join(OUT_DIR, "node_list.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("export_edge_list_csv")
    edge_list = df_edge[["src_author_id", "dst_author_id"]].copy()
    edge_list.insert(0, "edge_id", np.arange(len(edge_list), dtype=int))
    edge_list.to_csv(os.path.join(OUT_DIR, "edge_list.csv"), index=False, encoding="utf-8-sig")
    timer.stop()


# =========================
# 7) 导出随机攻击 seeds（推荐做法：避免写超大文件）
# =========================
def export_random_seeds(timer: Timer):
    """
    为随机攻击导出 seeds（可复现）：
    - random_seeds_nodes.csv：run_id, seed
    - random_seeds_edges.csv：run_id, seed
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    timer.start("export_random_seeds_nodes")
    seeds_nodes = [{"run_id": i + 1, "seed": BASE_RANDOM_SEED + 100000 + i} for i in range(N_RANDOM_RUNS_NODES)]
    pd.DataFrame(seeds_nodes).to_csv(os.path.join(OUT_DIR, "random_seeds_nodes.csv"),
                                     index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("export_random_seeds_edges")
    seeds_edges = [{"run_id": i + 1, "seed": BASE_RANDOM_SEED + 200000 + i} for i in range(N_RANDOM_RUNS_EDGES)]
    pd.DataFrame(seeds_edges).to_csv(os.path.join(OUT_DIR, "random_seeds_edges.csv"),
                                     index=False, encoding="utf-8-sig")
    timer.stop()


# =========================
# 8) 导出确定性攻击“静态移除序列”（重要 -> 不重要）
# =========================
def export_node_orders(df_node: pd.DataFrame, timer: Timer):
    """
    节点策略（重要->不重要）：
    - Degree（降序）
    - Strength（降序）
    - Betweenness（降序）
    - Effective Size（降序）
    Random 不在这里导出顺序（随机用 seed 动态生成）
    """
    out = OUT_DIR
    os.makedirs(out, exist_ok=True)

    timer.start("save_nodes_degree")
    tmp = df_node.sort_values("degree", ascending=False)[["author_id", "degree"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "nodes_degree.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_nodes_strength")
    tmp = df_node.sort_values("strength", ascending=False)[["author_id", "strength"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "nodes_strength.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_nodes_betweenness")
    tmp = df_node.sort_values("betweenness", ascending=False)[["author_id", "betweenness"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "nodes_betweenness.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_nodes_effective_size")
    tmp = df_node.sort_values("effective_size", ascending=False)[["author_id", "effective_size"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "nodes_effective_size.csv"), index=False, encoding="utf-8-sig")
    timer.stop()


def export_edge_orders(df_edge: pd.DataFrame, timer: Timer):
    """
    边策略：
    - Weak weight：weight 升序（弱边先移除）
    - Strong weight：weight 降序（强边先移除）
    - Strong weight + duration：weight 降序；同权重 duration 降序（持续更久更重要）
    - Edge betweenness：降序
    - Edge overlap：默认输出 low-overlap-first（桥边优先），也输出 high-overlap-first
    Random 不在这里导出顺序（随机用 seed 动态生成）
    """
    out = OUT_DIR
    os.makedirs(out, exist_ok=True)
    base_cols = ["src_author_id", "dst_author_id"]

    timer.start("save_edges_weak_weight")
    tmp = df_edge.sort_values("weight", ascending=True)[base_cols + ["weight"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_weak_weight.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_edges_strong_weight")
    tmp = df_edge.sort_values("weight", ascending=False)[base_cols + ["weight"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_strong_weight.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_edges_strong_weight_duration")
    tmp = df_edge.sort_values(["weight", "edge_duration"], ascending=[False, False])[
        base_cols + ["weight", "edge_duration"]
    ].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_strong_weight_duration.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_edges_edge_betweenness")
    tmp = df_edge.sort_values("edge_betweenness", ascending=False)[
        base_cols + ["edge_betweenness"]
    ].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_edge_betweenness.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_edges_overlap_low_first")
    tmp = df_edge.sort_values("edge_overlap", ascending=True)[base_cols + ["edge_overlap"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_edge_overlap_low_first.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    timer.start("save_edges_overlap_high_first")
    tmp = df_edge.sort_values("edge_overlap", ascending=False)[base_cols + ["edge_overlap"]].copy()
    tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
    tmp.to_csv(os.path.join(out, "edges_edge_overlap_high_first.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    # 若你开启了加权边介数，这里也顺便导出
    if "edge_betweenness_weighted_dist" in df_edge.columns:
        timer.start("save_edges_edge_betweenness_weighted_dist")
        tmp = df_edge.sort_values("edge_betweenness_weighted_dist", ascending=False)[
            base_cols + ["edge_betweenness_weighted_dist"]
        ].copy()
        tmp.insert(0, "rank", np.arange(1, len(tmp) + 1, dtype=int))
        tmp.to_csv(os.path.join(out, "edges_edge_betweenness_weighted_dist.csv"),
                   index=False, encoding="utf-8-sig")
        timer.stop()


# =========================
# 9) 主程序
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    timer = Timer()

    # 指标缓存文件（避免重复算介数/边介数）
    node_metric_path = os.path.join(OUT_DIR, "node_metrics_for_attacks.csv")
    edge_metric_path = os.path.join(OUT_DIR, "edge_metrics_for_attacks.csv")

    # 1) 读图
    G = load_graph(timer)

    # 2) 节点指标：缓存优先
    if (not FORCE_RECOMPUTE) and os.path.exists(node_metric_path):
        timer.start("load_cached_node_metrics")
        df_node = pd.read_csv(node_metric_path)
        timer.stop()
        print("[OK] Loaded cached node metrics.")
    else:
        df_node = compute_node_metrics(G, timer)
        timer.start("save_node_metrics_table")
        df_node.to_csv(node_metric_path, index=False, encoding="utf-8-sig")
        timer.stop()
        print("[OK] Node metrics saved:", node_metric_path)

    # 3) 边指标：缓存优先
    if (not FORCE_RECOMPUTE) and os.path.exists(edge_metric_path):
        timer.start("load_cached_edge_metrics")
        df_edge = pd.read_csv(edge_metric_path)
        timer.stop()
        print("[OK] Loaded cached edge metrics.")
    else:
        df_edge = compute_edge_metrics(G, timer)
        timer.start("save_edge_metrics_table")
        df_edge.to_csv(edge_metric_path, index=False, encoding="utf-8-sig")
        timer.stop()
        print("[OK] Edge metrics saved:", edge_metric_path)

    # 4) 导出稳定索引列表（后续随机攻击只洗牌 index）
    export_stable_index_lists(df_node, df_edge, timer)

    # 5) 导出随机攻击 seeds（100 次或更多）
    export_random_seeds(timer)

    # 6) 导出确定性攻击顺序（重要->不重要）
    export_node_orders(df_node, timer)
    export_edge_orders(df_edge, timer)

    # 7) 导出耗时报告
    timer.start("save_timing_report")
    timing_df = timer.to_df()
    timing_df.to_csv(os.path.join(OUT_DIR, "timing_report.csv"), index=False, encoding="utf-8-sig")
    timer.stop()

    print("\n[OK] All prepared files saved to:", OUT_DIR)
    print("\nOverall timing summary (desc):")
    print(timing_df.sort_values("seconds", ascending=False).to_string(index=False))

    # 8) 给后续移除实验的“用法提示”
    print("\n[Next-Step Tips]")
    print("1) 随机攻击：读取 node_list.csv / edge_list.csv + random_seeds_*.csv")
    print("   用 numpy RNG: rng = np.random.default_rng(seed); order = rng.permutation(N)")
    print("2) 确定性攻击：直接读取 nodes_degree.csv / edges_edge_betweenness.csv 等")
    print("3) 若你要做“自适应攻击”（每移除一步重算度/介数），那将是另一套模拟脚本（更慢但更严格）。")


if __name__ == "__main__":
    main()
