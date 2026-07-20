# -*- coding: utf-8 -*-
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
robustness_exact_v2.py

目标：对合著网络在不同攻击策略下，计算：
(1) 结构抗毁性：LCC ratio、pairwise connectivity ratio、#components
(2) 功能抗毁性：Global efficiency（精确）、容量指标（精确）

支持：
- 节点移除：Random / Degree / Strength / Betweenness / Effective size
- 边移除：Random / Weak weight / Strong weight / Edge betweenness / Overlap / Strong weight + duration

输出（足够后续绘图/对比/统计）：
- 每个策略（每个 run）输出一条“曲线表”CSV：
  columns:
    strategy, mode, run_id, step, removed_count, removed_frac, removed_percent,
    n_nodes, n_edges, lcc_size, lcc_ratio, num_components,
    pair_connected, pair_ratio,
    efficiency, eff_ratio,
    cap_remaining, cap_remaining_ratio,
    cap_in_lcc, cap_in_lcc_ratio
- Random 额外输出 mean/std 曲线
- timing_report.csv：各模块耗时（含功能指标 start/end 计时累计）

重要说明（你要求“最精确”）：
- EXACT_WEIGHTED 全局效率需要 all-pairs Dijkstra，在 17k 节点上极其消耗资源；
  但本脚本仍给出“精确实现”，你可自行选择模式与减少点数/减少随机重复次数。
"""

import os
import time
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


# =========================
# 0) 你需要修改的配置
# =========================

# 你的 LCC 网络（S5->S2 后的 gexf）
GRAPH_GEXF = str(_OUTPUT_ROOT / "02_network_construction" / "active_network_with_edge_duration" / "active_coauthorship_newman_s5_s2_2006_2025.gexf")

# 你已经导出的攻击顺序表目录（里面应包含 nodes_*.csv / edges_*.csv / node_list.csv / edge_list.csv / random_seeds_*.csv）
ATTACK_TABLE_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "attack_sequences")

# 输出目录
OUT_DIR = str(_OUTPUT_ROOT / "05_node_importance_robustness" / "exact_network_robustness")

# Random 重复次数（你之前提到 100 次，这里默认就设 100；想快就改小）
N_RANDOM_RUNS_NODES = 100
N_RANDOM_RUNS_EDGES = 100

# 曲线点数（包含 step=0）
# 例如 51 表示 0%,2%,4%,...,100% 共 51 个点
N_POINTS_NODES = 51
N_POINTS_EDGES = 51

# 最大移除比例（可只算到 0.5 之类，便于论文展示）
MAX_REMOVAL_FRAC_NODES = 1.0
MAX_REMOVAL_FRAC_EDGES = 1.0

# 功能鲁棒性：效率计算模式（全部“精确”）
# "EXACT_UNWEIGHTED"：无权全局效率（精确，仍很重，但一般比加权快）
# "EXACT_WEIGHTED"  ：用 dist=1/(1+weight) 做加权全局效率（精确，极重）
EFFICIENCY_MODE = "EXACT_UNWEIGHTED"

# 是否在读图时写入 dist（EXACT_WEIGHTED 必须 True）
USE_DIST_ATTR = True

# 作为“功能容量/影响力”的节点属性字段（建议 total_cites / papers_count / g_index）
CAPACITY_ATTR = "total_cites"

# 绘图开关（输出简单曲线图，便于快速检查）
PLOT_QUICK_CURVES = True


# =========================
# 1) 计时器（含 start/end 打印）
# =========================
class Timer:
    def __init__(self):
        self.rows = []  # (task, seconds)
        self._t0 = None
        self._name = None

    def start(self, name: str):
        self._name = name
        self._t0 = time.perf_counter()
        print(f"[START] {name}")

    def stop(self):
        dt = time.perf_counter() - self._t0
        self.rows.append((self._name, dt))
        print(f"[END]   {self._name} | time={dt:.3f}s")

    def add(self, name: str, seconds: float):
        self.rows.append((name, float(seconds)))

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["task", "seconds"])


# =========================
# 2) 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def weight_to_dist(w: float) -> float:
    """把 weight 转成距离 dist：合作越强（weight 越大）-> 距离越短（dist 越小）"""
    return 1.0 / (1.0 + float(w))


def canonical_edge(u: str, v: str) -> Tuple[str, str]:
    """无向边标准化：用排序保证 (u,v) 与 (v,u) 等价"""
    return (u, v) if u <= v else (v, u)


def load_graph(timer: Timer) -> nx.Graph:
    """读取 gexf，确保无向、weight 为 float，并可选写入 dist"""
    timer.start("load_graph_gexf")
    if not os.path.exists(GRAPH_GEXF):
        raise FileNotFoundError(f"[ERROR] 未找到 gexf：{GRAPH_GEXF}")
    G = nx.read_gexf(GRAPH_GEXF)
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    timer.stop()

    timer.start("sanitize_edge_weight_and_dist")
    for u, v, d in G.edges(data=True):
        # 清洗 weight
        w = d.get("weight", 1.0)
        if w is None or w == "":
            w = 1.0
        w = float(w)
        d["weight"] = w

        # 需要加权效率时，写入 dist
        if USE_DIST_ATTR:
            d["dist"] = weight_to_dist(w)
    timer.stop()

    # 你的网络理论上就是 LCC；若不是，自动取 LCC
    if not nx.is_connected(G):
        print("[WARN] 图不是连通的，将自动取最大连通分量 LCC。")
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    print(f"[OK] Loaded graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, connected={nx.is_connected(G)}")
    return G


def load_attack_orders(timer: Timer):
    """
    读取攻击顺序表：
    节点：degree/strength/betweenness/effective_size
    边：weak/strong/edge_betweenness/overlap_low_first/strong_weight_duration
    以及：node_list, edge_list, seeds_nodes, seeds_edges
    """
    timer.start("load_attack_tables")

    node_orders = {}
    node_map = {
        "degree": "nodes_degree.csv",
        "strength": "nodes_strength.csv",
        "betweenness": "nodes_betweenness.csv",
        "effective_size": "nodes_effective_size.csv",
    }
    for k, fn in node_map.items():
        p = os.path.join(ATTACK_TABLE_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] 缺少节点攻击表：{p}")
        df = pd.read_csv(p)
        node_orders[k] = df["author_id"].astype(str).tolist()

    edge_orders = {}
    edge_map = {
        "weak_weight": "edges_weak_weight.csv",
        "strong_weight": "edges_strong_weight.csv",
        "edge_betweenness": "edges_edge_betweenness.csv",
        "overlap_low_first": "edges_edge_overlap_low_first.csv",
        "strong_weight_duration": "edges_strong_weight_duration.csv",
    }
    for k, fn in edge_map.items():
        p = os.path.join(ATTACK_TABLE_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] 缺少边攻击表：{p}")
        df = pd.read_csv(p)
        src = df["src_author_id"].astype(str).tolist()
        dst = df["dst_author_id"].astype(str).tolist()
        edge_orders[k] = [canonical_edge(u, v) for u, v in zip(src, dst)]

    node_list = pd.read_csv(os.path.join(ATTACK_TABLE_DIR, "node_list.csv"))
    edge_list = pd.read_csv(os.path.join(ATTACK_TABLE_DIR, "edge_list.csv"))
    seeds_nodes = pd.read_csv(os.path.join(ATTACK_TABLE_DIR, "random_seeds_nodes.csv"))
    seeds_edges = pd.read_csv(os.path.join(ATTACK_TABLE_DIR, "random_seeds_edges.csv"))

    timer.stop()
    return node_orders, edge_orders, node_list, edge_list, seeds_nodes, seeds_edges


def build_capacity_dict(timer: Timer, G: nx.Graph) -> Dict[str, float]:
    """从图节点属性读取容量字段（如 total_cites），若缺失则退化为 1"""
    timer.start("build_capacity_dict")
    cap = {}
    for u, d in G.nodes(data=True):
        v = d.get(CAPACITY_ATTR, None)
        if v is None or v == "":
            cap[str(u)] = 1.0
        else:
            try:
                cap[str(u)] = float(v)
            except Exception:
                cap[str(u)] = 1.0
    timer.stop()
    return cap


# =========================
# 3) 结构指标（精确）
# =========================
def components_and_sizes(G: nx.Graph) -> Tuple[List[int], List[set]]:
    """返回所有连通分量大小列表 & 分量节点集合列表"""
    if G.number_of_nodes() == 0:
        return [], []
    comps = list(nx.connected_components(G))
    sizes = [len(c) for c in comps]
    return sizes, comps


def pairwise_connectivity(sizes: List[int]) -> int:
    """连通对数：sum s*(s-1)/2"""
    total = 0
    for s in sizes:
        total += s * (s - 1) // 2
    return int(total)


def sum_capacity(nodes: set, cap: Dict[str, float]) -> float:
    """给定节点集合，容量求和"""
    return float(sum(cap.get(str(u), 0.0) for u in nodes))


# =========================
# 4) 功能指标：全局效率（精确）+ 计时
# =========================
def exact_global_efficiency_unweighted(G: nx.Graph) -> float:
    """
    精确无权全局效率：
    E = (1 / (n(n-1))) * sum_{i!=j} 1/d_ij
    不连通的对（d=inf）贡献 0
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    inv_sum = 0.0
    # all_pairs_shortest_path_length 是精确 BFS 结果（无权）
    for src, dist_dict in nx.all_pairs_shortest_path_length(G):
        for tgt, d in dist_dict.items():
            if tgt != src and d > 0:
                inv_sum += 1.0 / d

    return inv_sum / (n * (n - 1))


def exact_global_efficiency_weighted(G: nx.Graph, weight_attr: str = "dist") -> float:
    """
    精确加权全局效率（使用 dist 做边长）：
    E = (1 / (n(n-1))) * sum_{i!=j} 1/d_ij
    注意：dist 必须为正；不连通对贡献 0

    这是 all-pairs Dijkstra，计算代价极高，但“精确”。
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    inv_sum = 0.0
    for src, dist_dict in nx.all_pairs_dijkstra_path_length(G, weight=weight_attr):
        for tgt, d in dist_dict.items():
            if tgt != src and d > 0:
                inv_sum += 1.0 / d

    return inv_sum / (n * (n - 1))


def compute_efficiency_exact_with_timing(timer: Timer, G: nx.Graph) -> float:
    """
    计算效率，并输出功能指标 start/end 计时（你要求的）
    """
    timer.start("functional_efficiency_exact")
    if EFFICIENCY_MODE == "EXACT_UNWEIGHTED":
        eff = exact_global_efficiency_unweighted(G)
    elif EFFICIENCY_MODE == "EXACT_WEIGHTED":
        # 加权效率需要 dist 属性
        eff = exact_global_efficiency_weighted(G, weight_attr="dist")
    else:
        raise ValueError(f"Unsupported EFFICIENCY_MODE: {EFFICIENCY_MODE}")
    timer.stop()
    return float(eff)


def compute_capacity_exact_with_timing(timer: Timer, G: nx.Graph, cap: Dict[str, float]) -> float:
    """
    计算剩余容量，并输出功能指标 start/end 计时（你要求的）
    """
    timer.start("functional_capacity_remaining_exact")
    val = float(sum(cap.get(str(u), 0.0) for u in G.nodes()))
    timer.stop()
    return val


# =========================
# 5) 核心：给定移除顺序，输出“足够绘图”的曲线表（精确）
# =========================
def simulate_curve_exact(
    timer: Timer,
    G0: nx.Graph,
    mode: str,  # "node" or "edge"
    strategy_name: str,
    remove_order,
    n_points: int,
    max_frac: float,
    cap: Dict[str, float],
    run_id: int
) -> pd.DataFrame:
    """
    给定移除顺序，按 n_points 个点逐步移除并计算精确曲线。

    输出字段足够后续绘图：
    - removed_frac / removed_percent
    - lcc_ratio / eff_ratio / cap_in_lcc_ratio 等
    - 结构与功能的“原始值”也保留（lcc_size, pair_connected, efficiency, cap_remaining 等）
    """
    timer.start(f"simulate_curve_exact[{mode}|{strategy_name}|run={run_id}]")

    G = G0.copy()
    N0 = G0.number_of_nodes()
    M0 = G0.number_of_edges()
    total_pairs0 = N0 * (N0 - 1) // 2

    # 原始容量总和（用于归一化）
    cap_total0 = float(sum(cap.get(str(u), 0.0) for u in G0.nodes()))
    if cap_total0 <= 0:
        cap_total0 = float(N0)  # 防止除零

    # 原始效率（精确）
    eff0 = compute_efficiency_exact_with_timing(timer, G0)
    if eff0 <= 0:
        eff0 = 1e-12

    # 处理移除上限
    max_remove = int(len(remove_order) * max_frac)
    steps = max(1, n_points - 1)
    per_step = max(1, math.ceil(max_remove / steps))

    # 若是边移除，需要把 remove_order 过滤成当前图确实存在的边
    if mode == "edge":
        # 建立现有边集合（canonical）
        exist = set(canonical_edge(str(u), str(v)) for u, v in G0.edges())
        remove_order = [e for e in remove_order if e in exist]

    # 若是点移除，需要过滤不存在节点（极少）
    if mode == "node":
        exist_nodes = set(str(u) for u in G0.nodes())
        remove_order = [str(u) for u in remove_order if str(u) in exist_nodes]

    rows = []
    removed_count = 0

    for step in range(n_points):
        # ---------- 结构指标（精确） ----------
        sizes, comps = components_and_sizes(G)
        if len(sizes) == 0:
            lcc_size = 0
            num_comp = 0
            pair_conn = 0
            cap_lcc = 0.0
        else:
            max_idx = int(np.argmax(sizes))
            lcc_nodes = comps[max_idx]
            lcc_size = sizes[max_idx]
            num_comp = len(sizes)
            pair_conn = pairwise_connectivity(sizes)
            cap_lcc = sum_capacity(lcc_nodes, cap)

        lcc_ratio = lcc_size / N0
        pair_ratio = (pair_conn / total_pairs0) if total_pairs0 > 0 else 0.0

        # ---------- 功能指标：容量（精确 + 计时） ----------
        cap_remaining = compute_capacity_exact_with_timing(timer, G, cap)
        cap_remaining_ratio = cap_remaining / cap_total0
        cap_in_lcc_ratio = cap_lcc / cap_total0

        # ---------- 功能指标：效率（精确 + 计时） ----------
        efficiency = compute_efficiency_exact_with_timing(timer, G)
        eff_ratio = efficiency / eff0

        # ---------- 移除比例 ----------
        if mode == "node":
            removed_frac = removed_count / N0
        else:
            removed_frac = (removed_count / M0) if M0 > 0 else 0.0

        rows.append({
            "strategy": strategy_name,
            "mode": mode,
            "run_id": int(run_id),
            "step": int(step),

            "removed_count": int(removed_count),
            "removed_frac": float(removed_frac),
            "removed_percent": float(removed_frac * 100.0),

            "n_nodes": int(G.number_of_nodes()),
            "n_edges": int(G.number_of_edges()),

            "lcc_size": int(lcc_size),
            "lcc_ratio": float(lcc_ratio),
            "num_components": int(num_comp),

            "pair_connected": int(pair_conn),
            "pair_ratio": float(pair_ratio),

            "efficiency": float(efficiency),
            "eff_ratio": float(eff_ratio),

            "cap_remaining": float(cap_remaining),
            "cap_remaining_ratio": float(cap_remaining_ratio),

            "cap_in_lcc": float(cap_lcc),
            "cap_in_lcc_ratio": float(cap_in_lcc_ratio),
        })

        # 最后一个点不再移除
        if step == n_points - 1:
            break

        # ---------- 执行移除 ----------
        start = removed_count
        end = min(removed_count + per_step, max_remove)
        if start >= end:
            continue

        batch = remove_order[start:end]

        if mode == "node":
            G.remove_nodes_from(batch)
        else:
            # batch 已经 canonical，直接 remove_edges_from
            G.remove_edges_from(batch)

        removed_count = end

    timer.stop()
    return pd.DataFrame(rows)


# =========================
# 6) Random 重复：输出每次 run 曲线 + mean/std
# =========================
def random_orders_from_seeds(
    mode: str,
    id_list_df: pd.DataFrame,
    seeds: List[int]
):
    """
    根据 seeds 生成随机攻击顺序（可复现）：
    - node_list.csv: author_id
    - edge_list.csv: src_author_id, dst_author_id
    """
    if mode == "node":
        items = id_list_df["author_id"].astype(str).tolist()
        for sd in seeds:
            rng = np.random.default_rng(sd)
            perm = rng.permutation(len(items))
            yield sd, [items[i] for i in perm]
    else:
        src = id_list_df["src_author_id"].astype(str).tolist()
        dst = id_list_df["dst_author_id"].astype(str).tolist()
        items = [canonical_edge(u, v) for u, v in zip(src, dst)]
        for sd in seeds:
            rng = np.random.default_rng(sd)
            perm = rng.permutation(len(items))
            yield sd, [items[i] for i in perm]


def aggregate_mean_std(curves_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对 random 的多次 run 按 step 聚合 mean/std"""
    group_cols = ["mode", "strategy", "step", "removed_count", "removed_frac", "removed_percent"]
    val_cols = [
        "n_nodes", "n_edges",
        "lcc_size", "lcc_ratio", "num_components",
        "pair_connected", "pair_ratio",
        "efficiency", "eff_ratio",
        "cap_remaining", "cap_remaining_ratio",
        "cap_in_lcc", "cap_in_lcc_ratio"
    ]
    mean_df = curves_df.groupby(group_cols, as_index=False)[val_cols].mean()
    std_df = curves_df.groupby(group_cols, as_index=False)[val_cols].std(ddof=0).fillna(0.0)
    return mean_df, std_df


# =========================
# 7) 快速绘图（可选）
# =========================
def plot_quick(df: pd.DataFrame, x: str, y: str, title: str, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df[x], df[y], linewidth=2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 8) 主流程
# =========================
def main():
    ensure_dir(OUT_DIR)
    timer = Timer()

    # 1) 读图
    G0 = load_graph(timer)

    # 2) 容量字典
    cap = build_capacity_dict(timer, G0)

    # 3) 读攻击表
    node_orders, edge_orders, node_list, edge_list, seeds_nodes, seeds_edges = load_attack_orders(timer)

    # 4) 保存参数信息（方便你写论文/复现实验）
    params_path = os.path.join(OUT_DIR, "run_params.json")
    params = {
        "GRAPH_GEXF": GRAPH_GEXF,
        "ATTACK_TABLE_DIR": ATTACK_TABLE_DIR,
        "N_RANDOM_RUNS_NODES": N_RANDOM_RUNS_NODES,
        "N_RANDOM_RUNS_EDGES": N_RANDOM_RUNS_EDGES,
        "N_POINTS_NODES": N_POINTS_NODES,
        "N_POINTS_EDGES": N_POINTS_EDGES,
        "MAX_REMOVAL_FRAC_NODES": MAX_REMOVAL_FRAC_NODES,
        "MAX_REMOVAL_FRAC_EDGES": MAX_REMOVAL_FRAC_EDGES,
        "EFFICIENCY_MODE": EFFICIENCY_MODE,
        "USE_DIST_ATTR": USE_DIST_ATTR,
        "CAPACITY_ATTR": CAPACITY_ATTR
    }
    pd.Series(params).to_json(params_path, force_ascii=False, indent=2)
    print("[OK] saved params:", params_path)

    # ========== A) 节点攻击（确定性策略） ==========
    node_strategies = ["degree", "strength", "betweenness", "effective_size"]
    all_curves = []

    for strat in node_strategies:
        df_curve = simulate_curve_exact(
            timer=timer,
            G0=G0,
            mode="node",
            strategy_name=f"node_{strat}",
            remove_order=node_orders[strat],
            n_points=N_POINTS_NODES,
            max_frac=MAX_REMOVAL_FRAC_NODES,
            cap=cap,
            run_id=0
        )
        out_csv = os.path.join(OUT_DIR, f"curve_node_{strat}_exact.csv")
        df_curve.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("[OK] saved:", out_csv)
        all_curves.append(df_curve)

        if PLOT_QUICK_CURVES:
            plot_quick(df_curve, "removed_percent", "lcc_ratio",
                       f"Node-{strat} LCC (Exact)", os.path.join(OUT_DIR, f"node_{strat}_lcc.png"))
            plot_quick(df_curve, "removed_percent", "eff_ratio",
                       f"Node-{strat} Efficiency (Exact)", os.path.join(OUT_DIR, f"node_{strat}_eff.png"))
            plot_quick(df_curve, "removed_percent", "cap_in_lcc_ratio",
                       f"Node-{strat} Capacity-in-LCC (Exact)", os.path.join(OUT_DIR, f"node_{strat}_capLCC.png"))

    # ========== B) 节点攻击（Random 多次精确） ==========
    # 注意：每个 step 都要算一次精确效率，因此 random=100 会非常重，但这是“最精确”版本。
    node_seeds = seeds_nodes["seed"].astype(int).tolist()[:N_RANDOM_RUNS_NODES]
    random_node_runs = []

    for run_idx, (sd, order) in enumerate(tqdm(random_orders_from_seeds("node", node_list, node_seeds),
                                              total=len(node_seeds), desc="Random node runs (Exact)")):
        df_run = simulate_curve_exact(
            timer=timer,
            G0=G0,
            mode="node",
            strategy_name="node_random",
            remove_order=order,
            n_points=N_POINTS_NODES,
            max_frac=MAX_REMOVAL_FRAC_NODES,
            cap=cap,
            run_id=run_idx + 1
        )
        random_node_runs.append(df_run)

    if len(random_node_runs) > 0:
        df_random_nodes_all = pd.concat(random_node_runs, ignore_index=True)
        out_all = os.path.join(OUT_DIR, "curve_node_random_allruns_exact.csv")
        df_random_nodes_all.to_csv(out_all, index=False, encoding="utf-8-sig")
        print("[OK] saved:", out_all)

        mean_df, std_df = aggregate_mean_std(df_random_nodes_all)
        mean_df.to_csv(os.path.join(OUT_DIR, "curve_node_random_mean_exact.csv"), index=False, encoding="utf-8-sig")
        std_df.to_csv(os.path.join(OUT_DIR, "curve_node_random_std_exact.csv"), index=False, encoding="utf-8-sig")
        print("[OK] saved random node mean/std")

        all_curves.append(mean_df)

    # ========== C) 边攻击（确定性策略） ==========
    edge_strategies = ["weak_weight", "strong_weight", "edge_betweenness", "overlap_low_first", "strong_weight_duration"]

    for strat in edge_strategies:
        df_curve = simulate_curve_exact(
            timer=timer,
            G0=G0,
            mode="edge",
            strategy_name=f"edge_{strat}",
            remove_order=edge_orders[strat],
            n_points=N_POINTS_EDGES,
            max_frac=MAX_REMOVAL_FRAC_EDGES,
            cap=cap,
            run_id=0
        )
        out_csv = os.path.join(OUT_DIR, f"curve_edge_{strat}_exact.csv")
        df_curve.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("[OK] saved:", out_csv)
        all_curves.append(df_curve)

        if PLOT_QUICK_CURVES:
            plot_quick(df_curve, "removed_percent", "lcc_ratio",
                       f"Edge-{strat} LCC (Exact)", os.path.join(OUT_DIR, f"edge_{strat}_lcc.png"))
            plot_quick(df_curve, "removed_percent", "eff_ratio",
                       f"Edge-{strat} Efficiency (Exact)", os.path.join(OUT_DIR, f"edge_{strat}_eff.png"))
            plot_quick(df_curve, "removed_percent", "cap_in_lcc_ratio",
                       f"Edge-{strat} Capacity-in-LCC (Exact)", os.path.join(OUT_DIR, f"edge_{strat}_capLCC.png"))

    # ========== D) 边攻击（Random 多次精确） ==========
    edge_seeds = seeds_edges["seed"].astype(int).tolist()[:N_RANDOM_RUNS_EDGES]
    random_edge_runs = []

    for run_idx, (sd, order) in enumerate(tqdm(random_orders_from_seeds("edge", edge_list, edge_seeds),
                                              total=len(edge_seeds), desc="Random edge runs (Exact)")):
        df_run = simulate_curve_exact(
            timer=timer,
            G0=G0,
            mode="edge",
            strategy_name="edge_random",
            remove_order=order,
            n_points=N_POINTS_EDGES,
            max_frac=MAX_REMOVAL_FRAC_EDGES,
            cap=cap,
            run_id=run_idx + 1
        )
        random_edge_runs.append(df_run)

    if len(random_edge_runs) > 0:
        df_random_edges_all = pd.concat(random_edge_runs, ignore_index=True)
        out_all = os.path.join(OUT_DIR, "curve_edge_random_allruns_exact.csv")
        df_random_edges_all.to_csv(out_all, index=False, encoding="utf-8-sig")
        print("[OK] saved:", out_all)

        mean_df, std_df = aggregate_mean_std(df_random_edges_all)
        mean_df.to_csv(os.path.join(OUT_DIR, "curve_edge_random_mean_exact.csv"), index=False, encoding="utf-8-sig")
        std_df.to_csv(os.path.join(OUT_DIR, "curve_edge_random_std_exact.csv"), index=False, encoding="utf-8-sig")
        print("[OK] saved random edge mean/std")

        all_curves.append(mean_df)

    # ========== E) 汇总输出（便于你统一绘图） ==========
    timer.start("save_curves_merged")
    if len(all_curves) > 0:
        merged = pd.concat(all_curves, ignore_index=True)
        merged_path = os.path.join(OUT_DIR, "curves_merged_for_plotting_exact.csv")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print("[OK] saved merged curves:", merged_path)
    timer.stop()

    # ========== F) timing report ==========
    timer.start("save_timing_report")
    timing_path = os.path.join(OUT_DIR, "timing_report.csv")
    timer.df().to_csv(timing_path, index=False, encoding="utf-8-sig")
    timer.stop()
    print("[OK] saved timing:", timing_path)

    print("\n[ALL DONE] outputs:", OUT_DIR)
    print("\nTiming summary (desc):")
    print(timer.df().sort_values("seconds", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
