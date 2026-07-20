import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

script_dir = os.path.dirname(os.path.abspath(__file__))
community_dir = os.path.dirname(script_dir)
input_dir = os.path.join(community_dir, "data")
output_dir = os.path.join(community_dir, "figures")

CSV_PATH = os.path.join(input_dir, "COLOR_community.csv")
TOPK = 10
TEXT_COLOR = "#444444"   # 你想要的字体颜色
DPI = 600
COMMUNITY_NAME_SLOT_CHARS = 24

COMMUNITY_NAMES = {
    21: "BiblioEval",
    54: "HealthNLP",
    39: "PlatformEcon",
    52: "GeoMobility",
    40: "GovAITrust",
    70: "CareSupport",
    12: "DigiDesign",
    53: "SocialTrust",
    14: "LibSocial",
    24: "EntityMetrics",
}

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]

if "l2" not in df.columns:
    raise KeyError(f"找不到 l2 列。当前列：{df.columns.tolist()}")

color_col = None
for c in ["Color", "color", "COLOR", "Colour", "colour", "颜色"]:
    if c in df.columns:
        color_col = c
        break
if color_col is None:
    raise KeyError(f"找不到颜色列（Color）。当前列：{df.columns.tolist()}")

def to_int_safe(x):
    try:
        return int(float(x))
    except Exception:
        return x

df["l2_clean"] = df["l2"].apply(to_int_safe)

# 去掉 l2=0
df2 = df[df["l2_clean"] != 0].copy()

# TopK（按社区规模）
sizes = df2.groupby("l2_clean").size().sort_values(ascending=False)
top_l2 = sizes.head(TOPK).index.tolist()

# 代表颜色：众数
def mode_color(s):
    s = s.dropna().astype(str).str.strip()
    if s.empty:
        return None
    return s.value_counts().idxmax()

rep_colors = df2.groupby("l2_clean")[color_col].apply(mode_color).to_dict()

def legend_label(community_id):
    community_name = str(COMMUNITY_NAMES.get(community_id, "")).strip()
    if community_name:
        return f"#{community_id}  {community_name}"
    return f"#{community_id}  {' ' * COMMUNITY_NAME_SLOT_CHARS}"

labels = [legend_label(v) for v in top_l2]
community_names = [str(COMMUNITY_NAMES.get(v, "")).strip() for v in top_l2]
colors_hex = [rep_colors.get(v, "#cccccc") for v in top_l2]

# 保存映射表（可选）
out_dir = output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

map_csv = os.path.join(out_dir, f"top{TOPK}_l2_color_map_no0.csv")
pd.DataFrame({
    "l2": top_l2,
    "community_name": community_names,
    "legend_name": labels,
    "color_hex": colors_hex
}).to_csv(
    map_csv, index=False, encoding="utf-8-sig"
)
print("已保存映射表：", map_csv)

# 字体设置
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

handles = [Patch(facecolor=col, edgecolor="none", label=lab) for lab, col in zip(labels, colors_hex)]

# 单列竖排
ncol = 1
nrow = len(handles)

# 高度按条目数自适应；dpi 提高到 600
fig_h = max(2.0, 0.32 * nrow + 0.8)
fig_w = 5.2

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
ax.axis("off")

leg = ax.legend(
    handles=handles,
    loc="center",
    ncol=ncol,
    frameon=False,
    # 如果你也想去掉标题，把下一行删掉即可
)

# 关键：设置图例文字颜色（包含标题）
# 标题颜色
if leg.get_title() is not None:
    leg.get_title().set_color(TEXT_COLOR)

# 各条目文字颜色
for t in leg.get_texts():
    t.set_color(TEXT_COLOR)

fig.tight_layout()

legend_png = os.path.join(out_dir, f"legend_top{TOPK}_l2_no0_singlecol_600dpi.png")
legend_svg = os.path.join(out_dir, f"legend_top{TOPK}_l2_no0_singlecol.svg")  # svg 不吃 dpi，但保留矢量
fig.savefig(legend_png, bbox_inches="tight", dpi=DPI)
fig.savefig(legend_svg, bbox_inches="tight")
plt.close(fig)

print("已输出图例：")
print("-", legend_png)
print("-", legend_svg)
