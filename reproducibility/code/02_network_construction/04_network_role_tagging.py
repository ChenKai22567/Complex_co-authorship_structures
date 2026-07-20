# -*- coding: utf-8 -*-  # 指定源代码文件的字符编码为 UTF-8，防止中文乱码
from pathlib import Path as _ArchivePath
_ALGORITHM_ROOT = _ArchivePath(__file__).resolve().parents[2]
_INPUT_ROOT = _ALGORITHM_ROOT / "data"
_OUTPUT_ROOT = _ALGORITHM_ROOT / "results"

"""
net_hierarchy_single_variable.py

==============================================================================
【程序功能详解】
本脚本旨在将多层级网络（核心、活跃、整体）的身份信息合并为一个单一的分类变量。
生成的网络文件将包含一个新的节点属性：`Author_Role`。

【分类逻辑 (优先级由高到低)】
1. **Core (核心骨干)**:
   - 该节点存在于“核心骨干网络”文件中。
   - 这类作者通常是学科的领军人物，处于网络最中心。

2. **Active (活跃作者)**:
   - 该节点不在核心网络中，但存在于“活跃作者网络”文件中。
   - 这类作者发文量较高，是学科的中坚力量。

3. **Marginal (普通/边缘作者)**:
   - 既不在核心也不在活跃网络中，仅存在于“学科整体网络”中。
   - 这类作者通常发文量较少，处于网络边缘。

【使用方法】
1. 修改配置区的文件名。
2. 运行脚本。
3. 在 Gephi 中，直接使用 `Author_Role` 属性进行 Partition（节点分区）染色。
==============================================================================
"""

import os  # 导入操作系统接口模块，用于路径处理
import networkx as nx  # 导入 NetworkX 库，用于图数据的读取与操作

# =========================
# 1. 全局配置区 (User Configuration)
# =========================

# 工作目录：存放网络文件的文件夹路径
WORK_DIR = str(_OUTPUT_ROOT / "02_network_construction" / "coauthorship_network_pipeline")

# 【输入文件】(请根据实际情况修改文件名)
# 1. 核心骨干网络 (最高优先级)
FILE_CORE = "coauthorship_newman_s5_s3_s2_2006_2025.gexf"

# 2. 活跃作者网络 (中等优先级)
FILE_ACTIVE = "coauthorship_newman_s5_s2_2006_2025.gexf"

# 3. 学科整体网络 (目标网络，将向此文件写入属性)
FILE_OVERALL = "coauthorship_newman_s2_2006_2025.gexf"

# 【输出设置】
# 生成的新属性名称
NEW_ATTR_NAME = "Author_Role"
# 输出文件的后缀 (防止覆盖原文件)
SUFFIX = "_with_roles"


# =========================
# 2. 核心处理逻辑
# =========================

def load_node_set(filename):
    """
    辅助函数：读取网络文件并返回节点 ID 的集合。
    """
    path = os.path.join(WORK_DIR, filename)  # 拼接完整路径
    if not os.path.exists(path):  # 检查文件是否存在
        print(f"[错误] 找不到文件: {path}")  # 报错
        return set()  # 返回空集合

    print(f"正在加载参考网络: {filename} ...")  # 打印进度
    try:
        # 根据后缀名选择读取方式
        if filename.endswith(".gexf"):
            G = nx.read_gexf(path)
        else:
            G = nx.read_graphml(path)

        nodes = set(G.nodes())  # 提取所有节点 ID 并转为集合
        print(f"  -> 成功加载，包含 {len(nodes)} 个节点")  # 打印统计信息
        return nodes  # 返回集合
    except Exception as e:
        print(f"  [异常] 读取失败: {e}")  # 打印异常信息
        return set()  # 返回空集合


def main():
    """主程序入口"""
    print(">>> 开始执行：网络角色合并标记...")  # 程序启动提示

    # --- 第一步：加载参考名单 ---
    core_nodes = load_node_set(FILE_CORE)  # 加载核心作者名单
    active_nodes = load_node_set(FILE_ACTIVE)  # 加载活跃作者名单

    if not core_nodes and not active_nodes:  # 检查是否至少加载了一个参考集
        print("未加载到任何参考节点，无法进行标记，程序退出。")
        return

    # --- 第二步：读取目标网络 ---
    target_path = os.path.join(WORK_DIR, FILE_OVERALL)  # 目标文件路径
    if not os.path.exists(target_path):  # 检查目标文件
        print(f"[错误] 目标整体网络不存在: {target_path}")
        return

    print(f"\n正在读取目标整体网络: {FILE_OVERALL} ...")
    try:
        # 读取整体网络
        if FILE_OVERALL.endswith(".gexf"):
            G = nx.read_gexf(target_path)
        else:
            G = nx.read_graphml(target_path)
    except Exception as e:
        print(f"[致命错误] 无法读取目标网络: {e}")
        return

    print(f"  -> 目标网络原始节点数: {len(G.nodes())}")

    # --- 第三步：计算并注入属性 ---
    # 统计计数器，用于最后打印报告
    count_core = 0
    count_active = 0
    count_marginal = 0

    print(f"正在注入新属性: '{NEW_ATTR_NAME}' ...")

    for node in G.nodes():  # 遍历整体网络中的每一个节点
        role_label = "Marginal"  # 默认角色：边缘/普通

        # 逻辑判断：优先级 Core > Active > Marginal
        if node in core_nodes:  # 如果在核心名单中
            role_label = "Core"  # 标记为核心
            count_core += 1  # 计数
        elif node in active_nodes:  # 如果不在核心，但在活跃名单中
            role_label = "Active"  # 标记为活跃
            count_active += 1  # 计数
        else:  # 都不在
            role_label = "Marginal"  # 保持默认
            count_marginal += 1  # 计数

        # 将计算出的角色字符串写入节点属性
        G.nodes[node][NEW_ATTR_NAME] = role_label

    # --- 第四步：打印统计报告 ---
    print("\n" + "=" * 30)
    print("【分类统计报告】")
    print(f"  1. Core (核心骨干): {count_core} 人")
    print(f"  2. Active (活跃作者): {count_active} 人")
    print(f"  3. Marginal (边缘作者): {count_marginal} 人")
    print(f"  总计检查: {count_core + count_active + count_marginal} / {len(G.nodes())}")
    print("=" * 30 + "\n")

    # --- 第五步：保存文件 ---
    # 构造输出文件名：原名_with_roles.gexf
    base, ext = os.path.splitext(FILE_OVERALL)
    out_name = f"{base}{SUFFIX}{ext}"
    out_path = os.path.join(WORK_DIR, out_name)

    print(f"正在保存文件至: {out_name} ...")
    try:
        if ext == ".gexf":
            nx.write_gexf(G, out_path)  # 保存为 GEXF
        else:
            nx.write_graphml(G, out_path)  # 保存为 GraphML
        print(">>> 全部完成！")
        print("提示：请将生成的文件导入 Gephi，在【节点颜色】中选择 Partition -> Author_Role 进行渲染。")
    except Exception as e:
        print(f"[保存失败] {e}")


if __name__ == "__main__":
    main()  # 执行主函数
