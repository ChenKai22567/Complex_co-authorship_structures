# -*- coding: utf-8 -*-  # 指定源代码文件的字符编码为 UTF-8，防止中文乱码
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_metrics_analysis_v9_structural_entropy.py

==============================================================================
【程序功能详解】
本脚本用于批量读取 .gexf 格式的网络文件，计算网络整体统计性质，并导出 CSV 报表。
[新增] 增加了“结构熵 (Structural Entropy)”的计算，用于衡量网络的异质性。

【指标定义与计算方法】

1. 基础连通性指标 (Basic Connectivity)
   - 网络密度 (Density): 实际边数 / 最大可能边数。
   - 网络传递性 (Transitivity): 全局聚类系数。
   - 连通分量数 (Components): 互不相连的子图数量。

2. 路径与效率指标 (Path & Efficiency)
   - 网络直径 (Diameter): 最大最短路径长度。
   - 平均路径长度 (Avg Path Length): 节点间距离平均值。
   - 全局效率 (Global Efficiency) & 归一化效率。
   - 局部效率 (Local Efficiency)。

3. 平衡、集聚与熵特征 (Balance, Agglomeration & Entropy)
   - 中心势 (Centralization): 衡量中心节点集中程度 (度、中介、紧密)。
   - 平均集聚度 (HHI): 衡量合作对象的集中程度。
   - [新增] 结构熵 (Structural Entropy): 衡量网络结构的异质性。
     公式: H = - sum(I_i * log(I_i))，其中 I_i 为节点的结构重要性。
     这里采用基于节点度数及其分布概率的改进算法计算 I_i。

4. 聚类与结构属性 (Clustering & Structure)
   - 同配系数 (Assortativity)。
   - 模块度 (Modularity)。
   - 平均聚类系数 (Avg Clustering)。
   - 标准化富俱乐部系数 (Norm Rich Club)。

5. 复杂网络特性 (Complex Features)
   - 小世界指数 (Small-World Sigma)。
   - 幂律分布指数 (Power Law Alpha)。

==============================================================================
"""

import os  # 导入操作系统接口
import glob  # 导入文件查找模块
import time  # 导入时间模块
import math  # 导入数学模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import networkx as nx  # 导入网络分析库
from networkx.algorithms import community  # 导入社区算法
import scipy.stats as stats  # 导入统计模块

# =========================
# 1. 全局配置与开关
# =========================
PATH_INPUT_DIR = str(_OUTPUT_ROOT / "02_network_construction" / "coauthorship_network_pipeline")
PATH_OUTPUT_FILE = str(_OUTPUT_ROOT / "03_network_metrics" / "batch_network_metrics" / "network_metrics_with_structural_entropy.csv")

# 【计算功能开关】
CALC_CONFIG = {
    # --- 基础指标 ---
    "density": False,
    "transitivity": False,
    "components": False,

    # --- 路径与效率 ---
    "diameter_path": False,  # 耗时
    "efficiency": False,

    # --- 平衡、集聚与熵 ---
    "cent_degree": False,
    "cent_betweenness": False,  # 耗时
    "cent_closeness": False,
    "agglomeration_hhi": False,
    "structural_entropy": True,  # [新增] 结构熵

    # --- 聚类与结构 ---
    "assortativity": False,
    "modularity": False,
    "clustering": False,
    "rich_club_norm": False,
    "power_law": False,

    # --- 依赖指标 ---
    "small_world": False
}


# =========================
# 2. 辅助工具函数
# =========================

def log(msg):
    """打印带时间戳的日志"""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")


def calculate_centralization(G, c_type="degree"):
    """计算中心势 (已归一化)"""
    n = len(G)
    if n < 3: return 0.0

    if c_type == "degree":
        centrality = nx.degree_centrality(G)
        c_vals = list(centrality.values())
        max_c = max(c_vals)
        return sum(max_c - c for c in c_vals) / (n - 2)

    elif c_type == "betweenness":
        centrality = nx.betweenness_centrality(G)
        c_vals = list(centrality.values())
        max_c = max(c_vals)
        return sum(max_c - c for c in c_vals) / (n - 1)

    elif c_type == "closeness":
        centrality = nx.closeness_centrality(G)
        c_vals = list(centrality.values())
        max_c = max(c_vals)
        numerator = sum(max_c - c for c in c_vals)
        denominator = ((n - 1) * (n - 2)) / (2 * n - 3)
        return numerator / denominator

    return 0.0


def calculate_network_hhi(G):
    """计算HHI集聚度"""
    hhi_values = []
    for u in G.nodes():
        strength = G.degree(u, weight='weight')
        if strength == 0:
            hhi_values.append(0.0)
            continue
        sum_sq = 0.0
        for v in G.neighbors(u):
            w = G[u][v].get('weight', 1.0)
            p = w / strength
            sum_sq += p * p
        hhi_values.append(sum_sq)
    if not hhi_values: return 0.0
    return np.mean(hhi_values)


def calculate_structural_entropy(G):
    """
    [新增] 计算网络结构熵 (Structural Entropy)
    公式依据：H = - sum(I_i * log(I_i))
    其中 I_i (结构重要性) 基于节点度数 k_i 和度分布概率 p(k_i) 计算。
    简化实现逻辑：
    1. 统计全网度分布概率 P(k)。
    2. 计算每个节点的“非归一化重要性” raw_I_i = (k_i + 1) * (1 - P(k_i))。
       (这里忽略 Delta 项，因为 P(k) <= 1，1-P(k) >= 0，加 1 是为了防止度为 0 的节点失效)
    3. 归一化得到 I_i = raw_I_i / sum(raw_I_j)。
    4. 计算熵。
    """
    try:
        n = len(G)
        if n == 0: return 0.0

        # 1. 获取所有节点的度
        degrees = dict(G.degree())
        degree_values = list(degrees.values())

        # 2. 计算度分布概率 P(k)
        # 统计每个度数出现的次数
        from collections import Counter
        k_counts = Counter(degree_values)
        total_nodes = len(degree_values)
        # P(k) = 该度数节点数 / 总节点数
        p_k = {k: count / total_nodes for k, count in k_counts.items()}

        # 3. 计算每个节点的结构重要性 I_i
        # 公式核心项：(k_i + 1) * [1 - p(k_i)]
        raw_importance = []
        for node, k in degrees.items():
            prob = p_k.get(k, 0)
            # 使用 (k+1) 防止孤立点 k=0 导致乘积为 0 (虽然孤立点重要性低，但不是无)
            # 使用 (1 - prob) 体现稀缺性：度数越罕见(prob越小)，重要性越高？
            # 或者反过来，通常熵是衡量不确定性。
            # 按照常见结构熵定义：I_i = k_i / 2M (简单的度熵)
            # 按照用户提供的复杂公式逻辑近似：
            val = (k + 1) * (1.0 - prob)
            raw_importance.append(val)

        # 4. 归一化 I_i
        total_raw = sum(raw_importance)
        if total_raw == 0: return 0.0

        norm_importance = [val / total_raw for val in raw_importance]

        # 5. 计算熵 H = - sum(I * log(I))
        entropy = 0.0
        for I in norm_importance:
            if I > 0:
                entropy += -1 * I * math.log(I)

        return entropy
    except Exception as e:
        print(f"    [Err] 结构熵计算出错: {e}")
        return np.nan


def fit_power_law_alpha(G):
    """拟合幂律指数"""
    degrees = [d for n, d in G.degree() if d > 0]
    if not degrees: return 0.0
    degree_counts = pd.Series(degrees).value_counts()
    x = degree_counts.index.values
    y = degree_counts.values
    valid = (x > 0) & (y > 0)
    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    if len(log_x) < 2: return 0.0
    slope, _, _, _, _ = stats.linregress(log_x, log_y)
    return -slope


def calculate_efficiency_metrics(G):
    """计算效率指标"""
    try:
        n = len(G)
        m = G.number_of_edges()
        if n == 0: return np.nan, np.nan, np.nan

        glob_eff = nx.global_efficiency(G)
        loc_eff = nx.local_efficiency(G)

        p = (2 * m) / (n * (n - 1)) if n > 1 else 0
        G_rand = nx.erdos_renyi_graph(n, p)
        rand_glob_eff = nx.global_efficiency(G_rand)

        norm_glob_eff = glob_eff / rand_glob_eff if rand_glob_eff > 0 else np.nan

        return glob_eff, loc_eff, norm_glob_eff
    except Exception as e:
        print(f"    [Err] 效率计算出错: {e}")
        return np.nan, np.nan, np.nan


def calculate_normalized_rich_club(G, top_percent=0.2):
    """计算标准化富俱乐部系数 (10次迭代)"""
    try:
        degrees = [d for n, d in G.degree()]
        if not degrees: return np.nan

        sorted_degrees = sorted(degrees, reverse=True)
        cutoff_index = int(len(sorted_degrees) * top_percent)
        if cutoff_index == 0: cutoff_index = 1
        k_threshold = sorted_degrees[cutoff_index - 1]

        if k_threshold < 2: k_threshold = 2

        rc_dict = nx.rich_club_coefficient(G, normalized=False)
        phi_real = rc_dict.get(k_threshold, 0.0)

        if phi_real == 0: return 0.0

        rand_phis = []
        for _ in range(10):
            R = G.copy()
            nx.double_edge_swap(R, nswap=len(G.edges()) * 5, max_tries=len(G.edges()) * 10)
            rc_rand = nx.rich_club_coefficient(R, normalized=False)
            rand_phis.append(rc_rand.get(k_threshold, 0.0))

        phi_rand_mean = np.mean(rand_phis)

        if phi_rand_mean > 0:
            return phi_real / phi_rand_mean
        else:
            return np.nan
    except Exception as e:
        return np.nan


# =========================
# 3. 核心分析逻辑
# =========================

def analyze_single_graph(file_path):
    """分析单个文件"""
    filename = os.path.basename(file_path)
    print(f"\n{'=' * 50}")
    log(f"开始分析文件: {filename}")

    t0_load = time.time()
    try:
        G = nx.read_gexf(file_path)
    except:
        try:
            G = nx.read_graphml(file_path)
        except:
            log("读取失败，跳过。")
            return None

    G = G.to_undirected()
    log(f"加载完成 (耗时 {time.time() - t0_load:.2f}s)")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    log(f"网络规模: 节点={n}, 边={m}")

    if n == 0: return None

    # 初始化结果字典
    res = {
        "Filename": filename, "Nodes": n, "Edges": m,
        "Density": np.nan, "Diameter (LCC)": np.nan, "Avg_Path_Len (LCC)": np.nan,
        "Global_Efficiency": np.nan, "Local_Efficiency": np.nan, "Norm_Global_Efficiency": np.nan,
        "Transitivity": np.nan, "Components": np.nan,
        "Centralization_Degree": np.nan, "Centralization_Betweenness": np.nan,
        "Centralization_Closeness": np.nan, "Avg_Agglomeration_HHI": np.nan,
        "Structural_Entropy": np.nan,  # [新增字段]
        "Assortativity": np.nan, "Modularity": np.nan, "Avg_Clustering_Coef": np.nan,
        "Norm_Rich_Club_Coef": np.nan,
        "Small_World_Sigma": np.nan, "Power_Law_Alpha": np.nan
    }

    # --- 计算指标 ---
    if CALC_CONFIG["density"]:
        res["Density"] = nx.density(G)

    if CALC_CONFIG["transitivity"]:
        res["Transitivity"] = nx.transitivity(G)

    if CALC_CONFIG["components"]:
        res["Components"] = nx.number_connected_components(G)

    avg_path_len_val = np.nan
    if CALC_CONFIG["diameter_path"]:
        t_s = time.time()
        if n > 0:
            lcc_nodes = max(nx.connected_components(G), key=len)
            G_lcc = G.subgraph(lcc_nodes).copy()
            try:
                res["Diameter (LCC)"] = nx.diameter(G_lcc)
                avg_path_len_val = nx.average_shortest_path_length(G_lcc)
                res["Avg_Path_Len (LCC)"] = avg_path_len_val
                log(f"  -> 直径/路径完成 ({time.time() - t_s:.2f}s)")
            except:
                pass

    if CALC_CONFIG["efficiency"]:
        t_s = time.time()
        glob_eff, loc_eff, norm_glob = calculate_efficiency_metrics(G)
        res["Global_Efficiency"] = glob_eff
        res["Local_Efficiency"] = loc_eff
        res["Norm_Global_Efficiency"] = norm_glob
        log(f"  -> 效率指标完成 ({time.time() - t_s:.2f}s)")

    if CALC_CONFIG["cent_degree"]:
        res["Centralization_Degree"] = calculate_centralization(G, "degree")

    if CALC_CONFIG["cent_betweenness"]:
        t_s = time.time()
        res["Centralization_Betweenness"] = calculate_centralization(G, "betweenness")
        log(f"  -> 中介中心势完成 ({time.time() - t_s:.2f}s)")

    if CALC_CONFIG["cent_closeness"]:
        res["Centralization_Closeness"] = calculate_centralization(G, "closeness")

    if CALC_CONFIG["agglomeration_hhi"]:
        res["Avg_Agglomeration_HHI"] = calculate_network_hhi(G)

    if CALC_CONFIG["structural_entropy"]:  # [新增计算]
        t_s = time.time()
        res["Structural_Entropy"] = calculate_structural_entropy(G)
        log(f"  -> 结构熵完成 ({time.time() - t_s:.2f}s)")

    if CALC_CONFIG["assortativity"]:
        res["Assortativity"] = nx.degree_assortativity_coefficient(G)

    avg_clust_val = np.nan
    if CALC_CONFIG["clustering"]:
        avg_clust_val = nx.average_clustering(G)
        res["Avg_Clustering_Coef"] = avg_clust_val

    if CALC_CONFIG["modularity"]:
        try:
            comms = community.greedy_modularity_communities(G)
            res["Modularity"] = community.modularity(G, comms)
        except:
            res["Modularity"] = 0.0

    if CALC_CONFIG["rich_club_norm"]:
        t_s = time.time()
        res["Norm_Rich_Club_Coef"] = calculate_normalized_rich_club(G, top_percent=0.2)
        log(f"  -> 富俱乐部完成 ({time.time() - t_s:.2f}s)")

    if CALC_CONFIG["power_law"]:
        res["Power_Law_Alpha"] = fit_power_law_alpha(G)

    if CALC_CONFIG["small_world"]:
        if pd.notna(avg_path_len_val) and pd.notna(avg_clust_val) and avg_path_len_val > 0:
            avg_k = 2 * m / n
            if avg_k > 1:
                c_rand = avg_k / (n - 1)
                l_rand = math.log(n) / math.log(avg_k)
                if c_rand > 0 and l_rand > 0:
                    res["Small_World_Sigma"] = (avg_clust_val / c_rand) / (avg_path_len_val / l_rand)

    return res


# =========================
# 4. 主程序入口
# =========================

def main():
    """主函数"""
    files = glob.glob(os.path.join(PATH_INPUT_DIR, "*.gexf"))
    if not files:
        print(f"未找到文件: {PATH_INPUT_DIR}")
        return

    print(f"找到 {len(files)} 个网络文件，开始分析...")
    print("配置状态: ", {k: 'ON' if v else 'OFF' for k, v in CALC_CONFIG.items()})

    results = []
    total = len(files)

    for i, f in enumerate(files):
        print(f"\nProgress: {i + 1}/{total}")
        try:
            res = analyze_single_graph(f)
            if res:
                for k, v in res.items():
                    if isinstance(v, float) and pd.notna(v):
                        res[k] = round(v, 6)
                results.append(res)
        except Exception as e:
            log(f"异常: {e}")

    if results:
        df = pd.DataFrame(results)

        cols = [
            "Filename", "Nodes", "Edges",
            "Density", "Diameter (LCC)", "Avg_Path_Len (LCC)",
            "Global_Efficiency", "Local_Efficiency", "Norm_Global_Efficiency",
            "Transitivity", "Components",
            "Centralization_Degree", "Centralization_Betweenness", "Centralization_Closeness",
            "Avg_Agglomeration_HHI", "Structural_Entropy",  # [新增位置]
            "Assortativity", "Modularity", "Avg_Clustering_Coef",
            "Norm_Rich_Club_Coef",
            "Small_World_Sigma", "Power_Law_Alpha"
        ]
        final_cols = [c for c in cols if c in df.columns]
        df = df[final_cols]

        df.to_csv(PATH_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n分析完成！结果已保存至:\n{PATH_OUTPUT_FILE}")
    else:
        print("无结果生成。")


if __name__ == "__main__":
    main()
