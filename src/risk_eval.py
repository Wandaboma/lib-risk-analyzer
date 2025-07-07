import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import os
import json
import re
from tabulate import tabulate

# -----------------------------
# Config paths
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
path = base_dir + '/../data/'
# path = "C:/Users/Admin/Downloads/db-dump/data/"
X_path = path + "package_substitutability.csv"
Y_path = path + "maintenance_activity.csv"
Z_path = path + "structure_criticality.csv"
directory_path = path + "/advisory-db-osv/crates/"

# -----------------------------
# Load and merge data
# -----------------------------
df_X = pd.read_csv(X_path).rename(columns={"package_name": "crate_name", "avg_top20_similarity": "X"})
df_Y = pd.read_csv(Y_path).rename(columns={"predicted_score": "Y"})
df_Z = pd.read_csv(Z_path).rename(columns={"PR": "Z"})

df = df_Y.merge(df_X, on="crate_name", how="left").merge(df_Z[["crate_name", "Z"]], on="crate_name", how="left")
df = df.dropna(subset=["X", "Y", "Z"])

# # Min-Max normalization
# scaler = MinMaxScaler()
# df[["X", "Y", "Z"]] = scaler.fit_transform(df[["X", "Y", "Z"]])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["X", "Y", "Z"]] = scaler.fit_transform(df[["X", "Y", "Z"]])

# from sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()
# df[["X", "Y", "Z"]] = scaler.fit_transform(df[["X", "Y", "Z"]])

# for col in ["X", "Y", "Z"]:
#     df[col] = df[col].rank(method="average", pct=True)

# Final score calculation
alpha = 0.5
beta = 0.4
gama = 0.1
df["final_score"] = alpha * df["Z"] + beta * (1 - df["X"]) + gama * (1 - df["Y"])

# Rename and sort
df = df.rename(columns={'X': "可替代性", 'Y': "维护质量", 'Z': "结构关键性"})
sorted_df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
sorted_df["rank"] = sorted_df.index + 1

# 提取三项指标列
indicators_df = sorted_df[["可替代性", "维护质量", "结构关键性"]]

# 计算皮尔逊相关性矩阵
corr_matrix = indicators_df.corr(method="pearson")

# 打印结果
print("三项指标之间的皮尔逊相关性矩阵：")
print(corr_matrix)

print(tabulate(sorted_df.head(20), headers='keys', tablefmt='psql', showindex=False))

# -----------------------------
# CVSS Parsing Utilities
# -----------------------------
def parse_cvss_vector(vector_str):
    try:
        match = re.match(r"CVSS:\d+\.\d+\/.*", vector_str)
        if match:
            from cvss import CVSS3
            return CVSS3(vector_str).base_score
    except:
        return None
    return None

def categorize_severity(score):
    if score is None:
        return "Unknown"
    if score < 4.0:
        return "Low"
    elif score < 7.0:
        return "Medium"
    elif score < 9.0:
        return "High"
    else:
        return "Critical"

# -----------------------------
# Extract CVSS Scores
# -----------------------------
cvss_scores = []
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        try:
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("affected", []):
                    name = item.get("package", {}).get("name")
                    if not name:
                        continue
                    max_score = None
                    for s in data.get("severity", []):
                        if s.get("type", "").startswith("CVSS"):
                            score = parse_cvss_vector(s.get("score", ""))
                            if score is not None:
                                max_score = max(score, max_score) if max_score else score
                    if max_score is not None:
                        cvss_scores.append({"crate_name": name, "cvss_score": max_score})
        except:
            continue

cvss_df = pd.DataFrame(cvss_scores)
cvss_df = cvss_df.groupby("crate_name", as_index=False)["cvss_score"].max()
cvss_df["severity"] = cvss_df["cvss_score"].apply(categorize_severity)

# -----------------------------
# Merge with rank info
# -----------------------------
cvss_rank_df = pd.merge(cvss_df, sorted_df[["crate_name", "rank", 'final_score']], on="crate_name")
cvss_rank_df["severity"] = cvss_rank_df["cvss_score"].apply(categorize_severity)

# 分组求每个 severity 等级的平均 rank 和数量
avg_rank_by_severity = (
    cvss_rank_df
    .groupby("severity")
    .agg(
        avg=("rank", "mean"),
        count=("rank", "count")
    )
    .reset_index()
    .sort_values(by="avg")  # 可选：按平均排名升序排序
)

# 输出结果
print("各 CVSS 严重性等级的平均排名及样本数量：")
print(avg_rank_by_severity)
# -----------------------------
# Kruskal-Wallis Test
# -----------------------------
severity_groups = cvss_rank_df.groupby("severity")
rank_data_by_group = [group["rank"].values for _, group in severity_groups if len(group) >= 3]
group_labels = [severity for severity, group in severity_groups if len(group) >= 3]

stat, p_value = kruskal(*rank_data_by_group)

from scipy.stats import mannwhitneyu

def run_one_sided_test(group1_label, group2_label, df, direction="less"):
    """
    对两个 severity 等级进行单边 Mann–Whitney U 检验
    direction: "less" 表示 group1 排名更靠前（更小）
    """
    group1 = df[df["severity"] == group1_label]["rank"].values
    group2 = df[df["severity"] == group2_label]["rank"].values

    if len(group1) < 3 or len(group2) < 3:
        print(f"⚠️ 样本不足：{group1_label} vs {group2_label}")
        return

    stat, p_value = mannwhitneyu(group1, group2, alternative=direction)
    print(f"\n📊 Mann–Whitney U 单边检验：{group1_label} 的 rank 是否更小 than {group2_label}")
    print(f"U统计量 = {stat:.4f}")
    print(f"p 值 = {p_value:.4f}")
    if p_value < 0.05:
        print("✅ 差异具有统计显著性（p < 0.05）")
    else:
        print("❌ 差异不显著")


# 执行两两比较
run_one_sided_test("Critical", "High", cvss_rank_df, direction="less")
run_one_sided_test("High", "Medium", cvss_rank_df, direction="less")

severity_groups = cvss_rank_df.groupby("severity")
score_data_by_group = [
    group["final_score"].values
    for severity, group in severity_groups
    if len(group) >= 3
]
group_labels = [
    severity
    for severity, group in severity_groups
    if len(group) >= 3
]


# 定义 Mann–Whitney 单边检验函数
def run_one_sided_test(group1_label, group2_label, df, field="final_score", direction="greater"):
    """
    单边 Mann–Whitney 检验
    direction: "greater" 表示 group1 的得分更高（更危险）
    """
    group1 = df[df["severity"] == group1_label][field].values
    group2 = df[df["severity"] == group2_label][field].values

    if len(group1) < 3 or len(group2) < 3:
        print(f"⚠️ 样本不足：{group1_label} vs {group2_label}")
        return

    stat, p_value = mannwhitneyu(group1, group2, alternative=direction)
    print(f"\n📊 Mann–Whitney U 单边检验：{group1_label} 的 {field} 是否显著大于 {group2_label}")
    print(f"U 统计量 = {stat:.4f}")
    print(f"p 值 = {p_value:.4f}")
    if p_value < 0.05:
        print("✅ 差异具有统计显著性（p < 0.05）")
    else:
        print("❌ 差异不显著")


# 两两比较（Critical vs High，High vs Medium）
run_one_sided_test("Critical", "High", cvss_rank_df, field="final_score", direction="greater")
run_one_sided_test("High", "Medium", cvss_rank_df, field="final_score", direction="greater")

# -----------------------------
# Correlation by Severity Group
# -----------------------------
correlation_by_group = []
for severity, group_df in severity_groups:
    if len(group_df) < 3:
        continue
    pearson = group_df["cvss_score"].corr(group_df["rank"], method="pearson")
    spearman = group_df["cvss_score"].corr(group_df["rank"], method="spearman")
    avg_rank = group_df["rank"].mean()
    correlation_by_group.append({
        "severity": severity,
        "count": len(group_df),
        "avg_rank": avg_rank,
        "pearson_corr": pearson,
        "spearman_corr": spearman
    })
correlation_df = pd.DataFrame(correlation_by_group)

# -----------------------------
# Ranking Normalization Summary
# -----------------------------
cvss_df["cvss_rank_score"] = cvss_df["cvss_score"].rank(method="average", pct=True)
rank_summary = cvss_df.groupby("severity").agg(
    avg_rank_score=("cvss_rank_score", "mean"),
    count=("cvss_rank_score", "count")
).reset_index()

# -----------------------------
# Plot: CVSS Score vs. Rank
# -----------------------------
# plt.figure(figsize=(8, 5))
# plt.scatter(cvss_rank_df["cvss_score"], cvss_rank_df["rank"], alpha=0.6)
# plt.xlabel("CVSS Score")
# plt.ylabel("Computed Rank")
# plt.title("CVSS Score vs. Computed Risk Rank")
# plt.grid(True)
# plt.show()
