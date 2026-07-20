# 复杂合著结构

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供复杂合著结构研究的复现材料，覆盖网络构建、时序指标、社区分析、鲁棒性、分形结构、层级结构、核心—边缘结构以及加权 ERGM。

![度分布](visualizations/network_features/figures/degree_distribution.jpg)

## 仓库结构

- `reproducibility/code`：按八个编号主题组织的 41 个规范化 Python/R 算法脚本。
- `reproducibility/data`：数据获取说明和 Release 公共数据解压目录。
- `reproducibility/results`：已发布计算结果的解压目录。
- `reproducibility/metadata`：算法、图像改名、Release 和 SHA-256 清单。
- `visualizations`：六个论文部分的绘图脚本、小型绘图数据和 JPG 预览图。
- `environment`：Python 和 R 依赖说明。
- `tools`：仓库及 Release 验证工具。

八个计算主题依次为：数据预处理、网络构建、网络指标、社区分析、节点重要性与鲁棒性、分形分析、层级与核心—边缘分析、ERGM。

## 数据可用性

Release `v0.1.0` 收录论文实际使用的 Web of Science 原始 Excel 导出文件，以及完整的作者消歧记录及其中的姓名、地址和单位字段。为保证发布输入输出与论文计算记录一致，这些数据不做匿名化，也不替换原始字段值。路径、字段和再利用注意事项见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。

Release 同时保留派生网络、作者标签、聚合指标、模型结果、鲁棒性曲线和绘图数据。

## 下载复现数据

从 [Releases 页面](https://github.com/ChenKai22567/Complex_co-authorship_structures/releases/tag/v0.1.0) 下载 `v0.1.0`，也可以使用 GitHub CLI：

```powershell
gh release download v0.1.0 `
  --repo ChenKai22567/Complex_co-authorship_structures `
  --dir release_assets
```

在仓库根目录解压两个数据分卷，它们会填充 `reproducibility/data`、`reproducibility/results` 和 `visualizations/*/data`。独立图像包仅包含 JPG 文件。

下载后执行：

```powershell
python tools/validate_repository.py --release-dir release_assets
```

## 运行环境

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
python -m pip install -r environment/requirements.txt
python tools/validate_repository.py --skip-release
```

nested WDSBM 需要在 Linux 下安装 `graph-tool`。ERGM 所需 R 包见 `environment/R_PACKAGES.md`。

## 复现顺序

按编号顺序执行八个主题。将 Release 数据包解压到仓库根目录后，数据预处理脚本会通过仓库相对路径读取其中的 WOS 文件；下游脚本以相同的路径契约引用上游结果。

1. 清洗并消歧文献作者记录。
2. 构建完整、活跃、骨干和时序合著网络。
3. 计算宏观、时序、度、聚类、路径和结构熵指标。
4. 运行 Infomap 并计算社区层指标。
5. 计算节点重要性、攻击顺序和精确鲁棒性曲线。
6. 运行分形、MST、重整化流和随机化分析。
7. 估计层级和核心—边缘角色。
8. 使用 seed 42 拟合加权 ERGM。

作者消歧、Infomap 1000 trials、精确鲁棒性、`graph-tool` WDSBM 和 ERGM 属于重型计算。Release 提供已有验证结果，持续集成不进行完整重跑。

## 验证

验证流程检查英文 ASCII 路径、缓存文件、本机绝对路径、Python 语法、数据契约、Release 中 WOS 原始表及完整消歧产物是否齐全、Release 哈希以及 JPG 完整性。本机没有 `Rscript` 时，R 解析检查会明确记录为跳过。

## 引用

请使用 [CITATION.cff](CITATION.cff) 中的仓库引用信息。正式论文题目和 DOI 确定后，将通过 `preferred-citation` 补充。

## 许可证

代码采用 [MIT License](LICENSE)。文档、图表和本项目生成的派生数据采用 [CC BY 4.0](LICENSE-DATA)。包括 WOS 导出在内的第三方文献记录仍受原数据库及订阅条款约束，本仓库不对其重新授权。
