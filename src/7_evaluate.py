"""
src/7_evaluate.py
步骤7: 统一评估与可视化
  - 读取所有 results/*.csv
  - 打印对比表
  - 绘制 Accuracy/Precision/Recall/F1 柱状图
运行: python src/7_evaluate.py
输出: figures/metrics_compare.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES    = os.path.join(BASE, "results")
FIG    = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- 读取所有结果 ----------
csvs = sorted([f for f in os.listdir(RES) if f.endswith("_results.csv")])
if not csvs:
    raise RuntimeError("没有找到任何 results/*.csv，请先运行 4_*、5、6 脚本")

df = pd.concat([pd.read_csv(os.path.join(RES, c), encoding="utf-8")
                for c in csvs], ignore_index=True)
print("\n" + "=" * 70)
print("全模型效果对比表")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

df.to_csv(os.path.join(RES, "all_results.csv"), index=False, encoding="utf-8")
print(f"\n汇总保存 → {os.path.join(RES, 'all_results.csv')}")

# ---------- 画图 ----------
metrics = ["acc", "prec", "rec", "f1", "auc"]
labels  = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC (ovr macro)"]
label_map = dict(zip(metrics, labels))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
colors = sns.color_palette("husl", len(df))

for idx, (ax, metric) in enumerate(zip(axes, metrics)):
    bars = ax.bar(df["model"] + "\n(" + df["vectorizer"] + ")",
                  df[metric], color=colors, edgecolor="black")
    ax.set_title(label_map[metric], fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

axes[-1].axis("off")   # 第6个子图留空

fig.suptitle("AG News 4类分类 — 不同算法 × 向量化效果对比", fontsize=15, y=1.02)
plt.tight_layout()
out = os.path.join(FIG, "metrics_compare.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"图表保存 → {out}")

# ---------- 打印最佳 ----------
print("\n各指标最佳模型:")
for m in metrics:
    best = df.loc[df[m].idxmax()]
    print(f"  {label_map[m]:35s} → {best['model']} ({best['vectorizer']}): {best[m]:.4f}")
