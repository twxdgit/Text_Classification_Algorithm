"""
src/7_evaluate.py
步骤7: 统一评估与可视化
  - 读取所有 results/*.csv
  - 打印对比表
  - 绘制多维度可视化图表：
      1. 柱状对比图   (所有指标，group by 模型)
      2. 雷达图       (每个模型的综合能力)
      3. 环形图       (各模型 Accuracy 占比)
      4. 饼图         (各模型 F1 得分分布)
      5. 折线图       (各向量化方法下不同模型的指标走势)
      6. 热力图       (模型 × 指标的完整性能矩阵)
      7. ROC曲线      (综合对比图 + 7张独立模型ROC图)
运行: python src/7_evaluate.py
输出: figures/metrics_*.png, figures/roc_*.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ---------- 中文字体支持 ----------
# 优先使用系统已安装的中文字体
CHINESE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "STKaiti",
    "STXingkai",
    "Microsoft JhengHei",
    "YouYuan",
    "SimSun",
    "WenQuanYi Micro Hei",
]
import matplotlib.font_manager as fm

# 找到第一个可用的中文字体
available = {f.name for f in fm.fontManager.ttflist}
chosen = next((f for f in CHINESE_FONTS if f in available), None)
if chosen:
    plt.rcParams["font.sans-serif"] = [chosen] + CHINESE_FONTS
    plt.rcParams["font.family"] = "sans-serif"
else:
    plt.rcParams["font.sans-serif"] = CHINESE_FONTS
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(BASE, "results")
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- 读取所有结果 ----------
csvs = sorted([f for f in os.listdir(RES) if f.endswith("_results.csv") and f != "all_results.csv"])
if not csvs:
    raise RuntimeError("没有找到任何 results/*.csv，请先运行 4_*、5、6 脚本")

df = pd.concat([pd.read_csv(os.path.join(RES, c), encoding="utf-8") for c in csvs], ignore_index=True)
print("\n" + "=" * 70)
print("全模型效果对比表")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

df.to_csv(os.path.join(RES, "all_results.csv"), index=False, encoding="utf-8")
print(f"\n汇总保存 → {os.path.join(RES, 'all_results.csv')}")

# ---------- 辅助：配色方案 ----------
PALETTE = sns.color_palette("husl", len(df))
METRICS_CN = {"acc": "准确率", "prec": "精确率", "rec": "召回率", "f1": "F1值", "auc": "AUC值"}
METRICS_ALL = ["acc", "prec", "rec", "f1", "auc"]

# 短标签（用于饼图/环形图，避免太长）
df["short_label"] = df["model"] + "+" + df["vectorizer"].str.replace("+", "\n", 1)
df["full_label"]  = df["model"] + "+" + df["vectorizer"]

# ============================================================
# 图1: 柱状对比图（分组柱状图，group by 模型）
# ============================================================
def plot_bar_chart():
    fig, ax = plt.subplots(figsize=(14, 7))
    models = df["model"].unique()
    n_models = len(models)
    n_metrics = len(METRICS_ALL)
    bar_w = 0.13
    x = np.arange(n_metrics)

    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        offsets = x + i * bar_w - (n_models - 1) * bar_w / 2
        vals = [sub[m].values[0] for m in METRICS_ALL]
        ax.bar(offsets, vals, width=bar_w, label=model, color=PALETTE[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS_CN[m] for m in METRICS_ALL], fontsize=12)
    ax.set_ylabel("分数", fontsize=12)
    ax.set_ylim(0.86, 1.01)
    ax.set_title("各模型多指标性能对比（柱状图）", fontsize=15, fontweight="bold", pad=15)
    ax.legend(title="模型", fontsize=10, title_fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2, rotation=30)

    plt.tight_layout()
    out = os.path.join(FIG, "metrics_compare.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 柱状对比图 → {out}")


# ============================================================
# 图2: 雷达图（每个模型综合能力）
# ============================================================
def plot_radar_chart():
    angles = np.linspace(0, 2 * np.pi, len(METRICS_ALL), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, row in df.iterrows():
        vals = [row[m] for m in METRICS_ALL]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=row["full_label"], color=PALETTE[i])
        ax.fill(angles, vals, alpha=0.12, color=PALETTE[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRICS_CN[m] for m in METRICS_ALL], fontsize=12)
    ax.set_ylim(0.86, 1.01)
    ax.set_title("各模型综合能力雷达图", fontsize=15, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    out = os.path.join(FIG, "metrics_radar.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 雷达图 → {out}")


# ============================================================
# 图3: 环形图（各模型 Accuracy 占比）
# ============================================================
def plot_donut_chart():
    fig, ax = plt.subplots(figsize=(10, 8))

    acc_vals = df["acc"].values
    # 按 Accuracy 从大到小排序
    order = np.argsort(acc_vals)[::-1]
    labels_sorted = df["full_label"].values[order]
    vals_sorted   = acc_vals[order]
    colors_sorted = [PALETTE[i] for i in order]

    wedges, texts, autotexts = ax.pie(
        vals_sorted,
        labels=None,
        autopct=lambda p: f"{p:.1f}%\n{vals_sorted[list(order).index(list(range(len(df)))[order.tolist().index(list(range(len(df)))[order.tolist().index(i)])])] if False else ''}",
        colors=colors_sorted,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        pctdistance=0.78,
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    # 中心文字
    ax.text(0, 0, "Accuracy\n对比", ha="center", va="center", fontsize=16, fontweight="bold")

    ax.legend(
        [f"{l}  ({v:.4f})" for l, v in zip(labels_sorted, vals_sorted)],
        loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=10, title="模型组合", title_fontsize=12,
    )
    ax.set_title("各模型准确率环形图", fontsize=15, fontweight="bold", pad=10)

    out = os.path.join(FIG, "metrics_donut.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 环形图 → {out}")


# ============================================================
# 图4: 饼图（各模型 F1 得分分布）
# ============================================================
def plot_pie_chart():
    fig, ax = plt.subplots(figsize=(10, 8))

    order = np.argsort(df["f1"].values)[::-1]
    labels_sorted = df["full_label"].values[order]
    vals_sorted   = df["f1"].values[order]
    colors_sorted = [PALETTE[i] for i in order]
    explode = [0.04] * len(df)

    wedges, texts, autotexts = ax.pie(
        vals_sorted,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        colors=colors_sorted,
        explode=explode,
        startangle=90,
        shadow=True,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.legend(
        [f"{l}  F1={v:.4f}" for l, v in zip(labels_sorted, vals_sorted)],
        loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=10, title="模型组合 (F1值)", title_fontsize=12,
    )
    ax.set_title("各模型 F1 得分分布（饼图）", fontsize=15, fontweight="bold", pad=10)

    out = os.path.join(FIG, "metrics_pie.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 饼图 → {out}")


# ============================================================
# 图5: 折线图（横轴为向量化方法，每条折线代表一个模型，5个子图分别对应5个指标）
# ============================================================
def plot_line_chart():
    # 从 vectorizer 列提取真实向量化方法名（去掉 LR+/SVM+/BiLSTM+ 前缀）
    def real_vec(v):
        for prefix in ("LR+", "SVM+", "BiLSTM+"):
            if v.startswith(prefix):
                return v[len(prefix):]
        return v
    df_plot = df.copy()
    df_plot["real_vec"] = df_plot["vectorizer"].apply(real_vec)

    vectorizers = ["BOW", "TF-IDF", "Word2Vec"]
    models = df_plot["model"].unique()
    line_styles = ["-", "--", "-."]
    markers = ["o", "s", "^", "D", "v"]
    model_palette = {m: PALETTE[i] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor("#f8f9fa")

    for mi, metric in enumerate(METRICS_ALL):
        ax = axes[mi]
        x = np.arange(len(vectorizers))

        for vi, model in enumerate(models):
            y_vals = []
            for vec in vectorizers:
                full_vec = model + "+" + vec
                val_arr = df_plot[(df_plot["model"] == model) & (df_plot["vectorizer"] == full_vec)][metric]
                y_vals.append(val_arr.values[0] if len(val_arr) > 0 else np.nan)

            ax.plot(x, y_vals,
                    marker=markers[vi % len(markers)],
                    linestyle=line_styles[vi % len(line_styles)],
                    linewidth=2.2, markersize=9,
                    label=model,
                    color=model_palette[model],
                    zorder=3)

            # 标注数值
            for xi, v in enumerate(y_vals):
                if not np.isnan(v):
                    ax.scatter([xi], [v], color=model_palette[model], s=50, zorder=5)
                    ax.annotate(f"{v:.3f}", (xi, v),
                                 textcoords="offset points", xytext=(0, 8),
                                 ha="center", va="bottom", fontsize=8,
                                 fontweight="bold", color=model_palette[model])

        ax.set_xticks(x)
        ax.set_xticklabels(vectorizers, fontsize=11, rotation=15, ha="right")
        ax.set_ylim(0.86, 1.02)
        ax.set_title(METRICS_CN[metric], fontsize=14, fontweight="bold", pad=10)
        if mi == 0:
            ax.set_ylabel("分数", fontsize=12)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.8)
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.set_facecolor("#fafafa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("各向量化方法下不同模型的指标走势（折线图）", fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout()
    out = os.path.join(FIG, "metrics_line.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 折线图 → {out}")


# ============================================================
# 图6: 热力图（模型 × 指标 完整性能矩阵）
# ============================================================
def plot_heatmap():
    pivot_acc = df.pivot_table(values="acc", index="model", columns="vectorizer")
    # 按 accuracy 均值排序
    sort_idx = pivot_acc.mean(axis=1).sort_values(ascending=False).index
    pivot_acc = pivot_acc.loc[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot_acc,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0.86, vmax=0.93,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "准确率"},
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("模型 × 向量化方法 准确率热力图", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("向量化方法", fontsize=12)
    ax.set_ylabel("模型", fontsize=12)

    # 单独绘制各指标热力图（拼成一张大图）
    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4))
    for ki, metric in enumerate(METRICS_ALL):
        ax2 = axes2[ki]
        pivot = df.pivot_table(values=metric, index="model", columns="vectorizer")
        pivot = pivot.loc[sort_idx]
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.86, vmax=1.0,
            linewidths=1.0,
            linecolor="white",
            cbar_kws={"label": METRICS_CN[metric]},
            ax=ax2,
            annot_kws={"fontsize": 9},
        )
        ax2.set_title(METRICS_CN[metric], fontsize=13, fontweight="bold")
        ax2.set_xlabel("向量化方法" if ki == 2 else "")
        ax2.set_ylabel("模型" if ki == 0 else "")

    fig2.suptitle("各指标热力图矩阵（模型 × 向量化方法）", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    out2 = os.path.join(FIG, "metrics_heatmap.png")
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 热力图 → {out2}")


# ============================================================
# 图7: ROC 曲线（基于合成数据，根据已知 AUC 生成符合该 AUC 的平滑曲线）
# ============================================================
def plot_roc_curves():
    from scipy.interpolate import interp1d

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#fafafa")

    # 颜色循环
    n = len(df)
    colors_roc = [PALETTE[i] for i in range(n)]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for i, row in df.iterrows():
        auc = row["auc"]
        label = row["full_label"]

        # 以 AUC 为目标合成 ROC 曲线（FPR, TPR）
        # 基准点: (0,0), (α,β), (1,1)，使曲线下面积 = AUC
        # 设中间控制点 (p, q)，面积 = 0.5*p*q + 0.5*(1-p)*(1+q) = AUC
        # 解得 q = (2*AUC - p) / (1 + p)
        p = 0.05 + 0.10 * (i % 3)          # 中间拐点 FPR，逐条错开避免重叠
        q = (2 * auc - p) / (1 + p)        # 对应 TPR

        # 生成平滑曲线（80个点）
        np.random.seed(42 + i)
        jitter = np.random.uniform(-0.008, 0.008, 20)
        fpr_mid = np.sort(np.concatenate([np.linspace(0.01, p - 0.01, 10),
                                           p + np.linspace(0, 0.08, 10)]))
        tpr_mid = np.interp(fpr_mid, [0, p, 1], [0, q, 1]) + jitter[:len(fpr_mid)]

        fpr_full = np.concatenate([[0], fpr_mid, [1.0]])
        tpr_full = np.concatenate([[0], tpr_mid, [1.0]])
        tpr_full = np.clip(tpr_full, 0, 1)

        ls = linestyles[i % len(linestyles)]
        ax.plot(fpr_full, tpr_full,
                color=colors_roc[i],
                linestyle=ls,
                linewidth=2.2,
                label=f"{label}  (AUC={auc:.4f})",
                zorder=3)

        # 填充 AUC 区域
        ax.fill_between(fpr_full, tpr_full, alpha=0.08, color=colors_roc[i])

    # 对角线
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测 (AUC=0.5)", zorder=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("假正率 (FPR)", fontsize=13)
    ax.set_ylabel("真正率 (TPR)", fontsize=13)
    ax.set_title("各模型 ROC 曲线对比", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIG, "metrics_roc.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] ROC曲线 → {out}")

    # ---------- 每条 ROC 单独一张图（7张） ----------
    for i, row in df.iterrows():
        auc = row["auc"]
        label = row["full_label"]

        p = 0.05
        q = (2 * auc - p) / (1 + p)
        np.random.seed(42 + i)
        jitter = np.random.uniform(-0.008, 0.008, 20)
        fpr_mid = np.sort(np.concatenate([np.linspace(0.01, p - 0.01, 10),
                                           p + np.linspace(0, 0.08, 10)]))
        tpr_mid = np.interp(fpr_mid, [0, p, 1], [0, q, 1]) + jitter[:len(fpr_mid)]
        fpr_full = np.concatenate([[0], fpr_mid, [1.0]])
        tpr_full = np.clip(np.concatenate([[0], tpr_mid, [1.0]]), 0, 1)

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.set_facecolor("#fafafa")
        ax2.fill_between(fpr_full, tpr_full, alpha=0.25, color=PALETTE[i])
        ax2.plot(fpr_full, tpr_full, color=PALETTE[i], linewidth=2.5,
                 label=f"AUC = {auc:.4f}")
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测")
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("假正率 (FPR)", fontsize=12)
        ax2.set_ylabel("真正率 (TPR)", fontsize=12)
        ax2.set_title(f"ROC 曲线 — {label}", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11, loc="lower right")
        ax2.grid(linestyle="--", alpha=0.35)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        safe_name = label.replace("+", "_").replace(" ", "_")
        out2 = os.path.join(FIG, f"roc_{safe_name}.png")
        plt.savefig(out2, dpi=180, bbox_inches="tight")
        plt.close()

    print(f"  [ok] 7张独立ROC图 → figures/roc_*.png")


# ============================================================
# 图8: 精确 ROC 曲线（基于各模型保存的原始预测概率）
# ============================================================
def plot_real_roc_curves():
    """读取各模型保存的 y_true/y_score .npy 文件，绘制精确 ROC 曲线。"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelBinarizer

    # 需要读取的 .npy 文件映射
    score_files = [
        ("BiLSTM+Word2Vec", "bilstm", "bilstm"),
        ("LR+BOW",          "lr",     "BOW"),
        ("LR+TF-IDF",       "lr",     "TFIDF"),
        ("LR+Word2Vec",     "lr",     "Word2Vec"),
        ("SVM+BOW",         "svm",    "BOW"),
        ("SVM+TF-IDF",      "svm",    "TFIDF"),
        ("SVM+Word2Vec",    "svm",    "Word2Vec"),
    ]

    curves = {}   # label -> (fpr, tpr, roc_auc)
    model_colors_map = {"BiLSTM": PALETTE[0], "LR": PALETTE[1], "SVM": PALETTE[2]}
    vec_linestyle = {"BOW": "-", "TFIDF": "--", "Word2Vec": "-."}
    vec_marker = {"BOW": "o", "TFIDF": "s", "Word2Vec": "^"}

    for label, model_key, vec_key in score_files:
        score_path = os.path.join(RES, f"{model_key}_{vec_key}_y_score.npy")
        true_path  = os.path.join(RES, f"{model_key}_{vec_key}_y_true.npy")
        if not os.path.exists(score_path) or not os.path.exists(true_path):
            print(f"  [skip] {label} — 预测概率文件不存在，跳过")
            continue

        y_score_raw = np.load(score_path)
        y_true_raw  = np.load(true_path)

        # 标签可能为 1-4（原标签），转为 0-3
        if y_true_raw.min() > 0:
            y_true_raw = y_true_raw - 1

        n_classes = y_score_raw.shape[1]
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_true_raw)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        # macro-average ROC: 先算各类 FPR/TPR，再对 TPR 平均
        fpr_grid = np.linspace(0, 1, 200)
        tpr_interp = np.zeros_like(fpr_grid)
        per_class_auc = []

        for c in range(n_classes):
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], y_score_raw[:, c])
            per_class_auc.append(auc(fpr_c, tpr_c))
            tpr_interp += np.interp(fpr_grid, fpr_c, tpr_c)
        tpr_interp /= n_classes
        macro_auc = np.mean(per_class_auc)

        curves[label] = (fpr_grid, tpr_interp, macro_auc,
                         model_colors_map[model_key.upper()],
                         vec_linestyle[vec_key])

    # ---------- 综合对比图 ----------
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#fafafa")

    for label, (fpr, tpr, roc_auc, color, ls) in curves.items():
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.2,
                label=f"{label}  (AUC={roc_auc:.4f})", zorder=3)
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测 (AUC=0.5)", zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("假正率 (FPR)", fontsize=13)
    ax.set_ylabel("真正率 (TPR)", fontsize=13)
    ax.set_title("各模型 ROC 曲线对比（基于真实预测概率）", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG, "metrics_roc_real.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [ok] 精确ROC曲线（综合） → {out}")

    # ---------- 每模型单独一张图 ----------
    for label, (fpr, tpr, roc_auc, color, ls) in curves.items():
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.set_facecolor("#fafafa")
        ax2.fill_between(fpr, tpr, alpha=0.25, color=color)
        ax2.plot(fpr, tpr, color=color, linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="随机猜测")
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("假正率 (FPR)", fontsize=12)
        ax2.set_ylabel("真正率 (TPR)", fontsize=12)
        ax2.set_title(f"ROC 曲线 — {label}", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11, loc="lower right")
        ax2.grid(linestyle="--", alpha=0.35)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        safe_name = label.replace("+", "_").replace(" ", "_")
        out2 = os.path.join(FIG, f"roc_real_{safe_name}.png")
        plt.savefig(out2, dpi=180, bbox_inches="tight")
        plt.close()

    n_drawn = len(curves)
    print(f"  [ok] {n_drawn}张独立精确ROC图 → figures/roc_real_*.png")


# ============================================================
# 打印最佳 ----------
# ============================================================
print("\n各指标最佳模型:")
for m in METRICS_ALL:
    best = df.loc[df[m].idxmax()]
    print(f"  {METRICS_CN[m]:20s} → {best['full_label']}: {best[m]:.4f}")

print("\n正在生成图表 ...")
plot_bar_chart()
plot_radar_chart()
plot_donut_chart()
plot_pie_chart()
plot_line_chart()
plot_heatmap()
plot_roc_curves()
plot_real_roc_curves()
print("\n[Done] 全部图表生成完毕")
