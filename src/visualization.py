import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from src.config import COLORS, COLOR_LIST, TARGET_COL

class ClassDistributionPlot:
    def plot(self, df: pd.DataFrame) -> go.Figure:
        counts = df[TARGET_COL].value_counts()
        labels = ["Benign (B)", "Malignant (M)"]
        values = [counts.get("B", 0), counts.get("M", 0)]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=["Percentage", "Sample Count"]
        )

        fig.add_trace(
            go.Pie(
                labels=labels, values=values,
                pull=[0, 0.1],
                marker=dict(colors=[COLORS['benign'], COLORS['malignant']]),
                textinfo="label+percent",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=labels, y=values,
                marker_color=[COLORS['benign'], COLORS['malignant']],
                text=values, textposition="auto",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title_text="Label Distribution - Breast Cancer Wisconsin",
            plot_bgcolor=COLORS['dark_gray'],
            paper_bgcolor=COLORS['black'],
            font_color=COLORS['white'],
            showlegend=False,
        )

        return fig

class FeatureBoxPlot:
    def plot(self, df: pd.DataFrame, group: str = "mean") -> plt.Figure:
        feat_cols = [c for c in df.columns if c.endswith(f"_{group}")]
        if not feat_cols:
            feat_cols = [c for c in df.columns if c != TARGET_COL][:10]
        
        n = len(feat_cols)
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten()

        palette = {"B": COLORS['benign'], "M": COLORS['malignant']}
        for i, col in enumerate(feat_cols):
            sns.boxplot(
                data=df, x=TARGET_COL, y=col,
                hue=TARGET_COL, palette=palette,
                ax=axes[i], order=["B", "M"],
                legend=False
            )
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        group_label = {"mean": "Mean", "se": "Standard Error", "worst": "Worst"}.get(group, group)
        fig.suptitle(f"Boxplot of '{group_label}' Features - Malignant vs Benign", fontsize=14, y=1.01)
        plt.tight_layout()

        return fig


class FeatureRangePlot:
    def plot(self, df: pd.DataFrame) -> plt.Figure:
        num_cols = df.select_dtypes(include=np.number).columns
        stats_df = df[num_cols].describe().T

        fig, ax = plt.subplots(figsize=(18,6))
        ax.errorbar(
            x=stats_df.index,
            y=stats_df["mean"],
            yerr=[
                stats_df["mean"] - stats_df["min"],
                stats_df["max"] - stats_df["mean"]
            ],
            fmt="o", color='royalblue', ecolor="lightgray",
            elinewidth=3, capsize=4, label="Mean & Range (Min-Max)"
        )
        ax.set_title("Value Range (Min-Mean-Max) of Features", fontsize=14)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        return fig
    
class ViolinPlot:
    def plot(self, df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        num_cols = df.select_dtypes(include=np.number).columns

        def cohens_d(col: str) -> float:
            g1 = df[df[TARGET_COL] == "M"][col].dropna()
            g2 = df[df[TARGET_COL] == "B"][col].dropna()
            pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            return abs(g1.mean() - g2.mean()) / (pooled_std + 1e-9)
        
        scores = {c: cohens_d(c) for c in num_cols}
        top_features = sorted(scores, key=scores.get, reverse=True)[:top_n]


        ncols = 5
        nrows = (top_n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten()

        palette = {"B": COLORS['benign'], "M": COLORS['malignant']}

        for i, col in enumerate(top_features):
            sns.violinplot(
                data=df, x=TARGET_COL, y=col,
                hue=TARGET_COL, palette=palette,
                ax=axes[i], order=["B", "M"],
                inner="box", cut=0,
                legend=False,
            )
            axes[i].set_title(f"{col}\nd={scores[col]:.2f}", fontsize=9)
            axes[i].set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Top {top_n} Most Separable Features (Cohen's d)", fontsize=14)
        plt.tight_layout()

        return fig

class FeatureDensityPlot:
    def plot(self, df: pd.DataFrame) -> plt.Figure:
        with plt.style.context({
            "axes.facecolor"   : COLORS['dark_gray'],
            "axes.linewidth"   : 2,
            "xtick.color"      : COLORS['white'],
            "ytick.color"      : COLORS['white'],
            "axes.labelcolor"  : COLORS['white'],
            "text.color"       : COLORS['white'],
            "figure.autolayout": True,
        }):
            num_cols = df.select_dtypes(include=np.number).columns
            n = len(num_cols)
            ncols = 6
            nrows = (n + ncols - 1) // ncols

            fig = plt.figure(figsize=(ncols * 4, nrows * 3.5))
            fig.patch.set_facecolor(COLORS['black'])
            fig.suptitle("Feature Density - Malignant vs Benign", fontsize=16)

            mal = df[df[TARGET_COL] == "M"]
            ben = df[df[TARGET_COL] == "B"]

            for i, col in enumerate(num_cols):
                ax = plt.subplot(nrows, ncols, i + 1)
                sns.kdeplot(
                    x=mal[col], ax=ax, linewidth=2,
                    color=COLORS['malignant'], label="Malignant"
                )
                sns.kdeplot(
                    x=ben[col], ax=ax, linewidth=2,
                    color=COLORS['benign'], label="Benign"
                )
                ax.set_title(
                    col.replace("_mean", "\n_mean").replace("_worst", "\n_worst").replace("_se", "\n_se"),
                    fontsize=7
                )
                ax.legend(fontsize=6)
            
            plt.tight_layout()

        return fig
    
class CorrelationHeatmapPlot:
    def plot(self, df: pd.DataFrame) -> plt.Figure:
        num_df = df.select_dtypes(include=np.number)

        mean_cols = [c for c in num_df.columns if c.endswith("_mean")]
        se_cols = [c for c in num_df.columns if c.endswith("_se")]
        worst_cols = [c for c in num_df.columns if c.endswith("_worst")]

        groups = [
            (mean_cols,  "_mean Group"),
            (se_cols,    "_se Group"),
            (worst_cols, "_worst Group"),
        ]
        groups = [(cols, lbl) for cols, lbl in groups if cols]

        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
        if n == 1:
            axes = [axes]

        for ax, (cols, lbl) in zip(axes, groups):
            corr = num_df[cols].corr()
            short = {
                c: c.replace("_mean", "").replace("_se", "").replace("_worst", "")
                for c in cols
            }
            corr.rename(index=short, columns=short, inplace=True)

            sns.heatmap(
                corr, cmap="coolwarm_r", annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
            )
            ax.set_title(f"Correlation Matrix - {lbl}", fontsize=12)
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Correlation Matrix by Feature Group", fontsize=15, y=1.02)
        plt.tight_layout()

        return fig

class OutlierDistributionPlot:
    def plot(self, df: pd.DataFrame, z_threshold: float = 3.0) -> go.Figure:
        num_df = df.select_dtypes(include=np.number)
        z_scores = pd.DataFrame(
            np.abs(stats.zscore(num_df)),
            columns=num_df.columns,
            index=num_df.index
        )

        outlier_mask = z_scores > z_threshold

        ol_df = outlier_mask.groupby(df[TARGET_COL]).sum().T
        for cls in ("B", "M"):
            if cls not in ol_df.columns:
                ol_df[cls] = 0
        
        count_b = (df[TARGET_COL] == "B").sum()
        count_m = (df[TARGET_COL] == "M").sum()
        ol_df["Ratio_B"] = ol_df["B"] / count_b
        ol_df["Ratio_M"] = ol_df["M"] / count_m
        total = ol_df["Ratio_B"] + ol_df["Ratio_M"]
        ol_df["Benign (%)"] = (ol_df["Ratio_B"] / total.replace(0, np.nan)) * 100
        ol_df["Malignant (%)"] = (ol_df["Ratio_M"] / total.replace(0, np.nan)) * 100
        ol_df = ol_df.dropna().sort_values("Malignant (%)", ascending=True)

        fig = px.bar(
            ol_df,
            y=ol_df.index,
            x=["Benign (%)", "Malignant (%)"],
            orientation="h",
            title=f"Proportion of Outliers (|z| > {z_threshold}) - Malignant vs Benign",
            color_discrete_map={
                "Benign (%)": COLORS['benign'],
                "Malignant (%)": COLORS['malignant'],
            },
            labels={"value": "Percentage (%)", "index": "Feature"},
            height=900
        )
        fig.update_layout(barmode="stack", title_x=0.5, template="plotly_dark")

        return fig

class PCAVariancePlot:
    def plot(self, explained_variance_ratio: np.ndarray) -> plt.Figure:
        cumulative = np.cumsum(explained_variance_ratio) * 100
        individual = explained_variance_ratio * 100
        n = len(individual)
        x = range(1, n + 1)
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        ax1.bar(x, individual, color=COLORS['benign'], alpha=0.7, label="Explained Variance (%)")
        ax2.plot(x, cumulative, color=COLORS['malignant'], marker="o", linewidth=2, label="Cumulative Variance (%)")
        ax2.axhline(95, color='gray', linestyle='--', linewidth=1, label="95% threshold")

        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance (%)", color=COLORS["benign"])
        ax2.set_ylabel("Cumulative Variance (%)", color=COLORS["malignant"])
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"PC{i}" for i in x], rotation=45, fontsize=8)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        ax1.set_title("PCA - Explained Variance Ratio", fontsize=14)
        plt.tight_layout()
        return fig

class PCAScatterPlot:
    def plot(self, pca_df: pd.DataFrame, mode: str = "2d") -> go.Figure:
        color_map = {"M": COLORS['malignant'], "B": COLORS['benign']}

        if mode == "3d" and "PC3" in pca_df.columns:
            fig = px.scatter_3d(
                pca_df, x="PC1", y="PC2", z="PC3",
                color=TARGET_COL,
                color_discrete_map=color_map,
                title="PCA 3D - PC1 vs PC2 vs PC3",
                opacity=0.7,
                symbol=TARGET_COL
            )
        else:
            fig = px.scatter(
                pca_df, x="PC1", y="PC2",
                color=TARGET_COL,
                color_discrete_map=color_map,
                title="PCA 2D — PC1 vs PC2",
                opacity=0.7,
                symbol=TARGET_COL,
                hover_data=[TARGET_COL]
            )

        fig.update_layout(
            paper_bgcolor=COLORS['black'],
            plot_bgcolor=COLORS['dark_gray'],
            font_color=COLORS['white'],
        )
        
        return fig
        


#===========================================
# Convenience functions
#===========================================

def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    return ClassDistributionPlot().plot(df)

def plot_feature_range(df: pd.DataFrame) -> plt.Figure:
    return FeatureRangePlot().plot(df)

def plot_feature_boxplots(df: pd.DataFrame, group: str = "mean") -> plt.Figure:
    return FeatureBoxPlot().plot(df, group)

def plot_violin(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    return ViolinPlot().plot(df, top_n)

def plot_feature_density(df: pd.DataFrame) -> plt.Figure:
    return FeatureDensityPlot().plot(df)

def plot_correlation_heatmaps(df: pd.DataFrame) -> plt.Figure:
    return CorrelationHeatmapPlot().plot(df)

def plot_outlier_distribution(df: pd.DataFrame, z: float = 3.0) -> go.Figure:
    return OutlierDistributionPlot().plot(df, z)

def plot_pca_explained_variance(evr: np.ndarray) -> plt.Figure:
    return PCAVariancePlot().plot(evr)

def plot_pca_scatter(pca_df: pd.DataFrame, mode: str = "2d") -> go.Figure:
    return PCAScatterPlot().plot(pca_df, mode)

#===========================================
# Confusion Matrix
#===========================================

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] = ['Benign', 'Malignant'],
        title: str = "Confusion Matrix",
        ax: plt.Axes | None = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
        
    sns.heatmap(
        cm, cmap = 'Blues', annot= True, fmt="d",
        linewidths=1, ax=ax, linecolor='gray',
        xticklabels=labels, yticklabels=labels
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.text(
        0.74, 0.6,
        f"TN={tn}; FP={fp} \nFN={fn}; TP={tp}",
        transform=ax.transAxes, fontsize=10,
        va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    )

    plt.tight_layout()

    return fig

#===========================================
# ROC Curve
#===========================================

def plot_roc_curve(
        models: list[dict],
        y_true: np.ndarray,
        threshold: float = 0.5,
        title: str = "ROC Curve Comparision"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, m in enumerate(models):
        color = m.get('color', COLOR_LIST[i % len(COLOR_LIST)])
        ls = m.get('linestyle', '-')
        fpr, tpr, _ = roc_curve(y_true, m['y_proba'])
        auc = roc_auc_score(y_true, m['y_proba'])
        ax.plot(
            fpr, tpr,
            color=color, linewidth=2.2, linestyle=ls,
            label=f"{m['name']}  (AUC = {auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

    if threshold is not None:
        for m in models:
            fpr_t, tpr_t, thresholds_t = roc_curve(y_true, m['y_proba'])
            idx = np.argmin(np.abs(thresholds_t - threshold))
            ax.scatter(
                fpr_t[idx], tpr_t[idx], color=m['color_threshold'], s=100, zorder=5,
                label=f"Threshold = {threshold}\n(FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})"
            )
        

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig

#===========================================
# LR Coefficinents
#===========================================

def plot_coefficients(
        coefficients: np.ndarray,
        feature_names: list[str],
        title: str = "Logistic Regression Coefficients",
        top_n: int | None = None
) -> plt.Figure:
    coef_series = pd.Series(coefficients, index=feature_names)

    if top_n is not None:
        coef_series = coef_series.reindex(
            coef_series.abs().sort_values(ascending=False).index
        ).head(top_n)

    coef_series = coef_series.sort_values()
    bar_colors = [COLORS['malignant'] if v > 0 else COLORS['benign'] for v in coef_series.values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(coef_series) * 0.28)))
    ax.barh(coef_series.index, coef_series.values, color = bar_colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Coefficient value (θ)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    legend_handles = [
        Patch(color=COLORS['malignant'], label="Feature increase raises P(Malignant)"),
        Patch(color=COLORS['benign'], label="Feature increase lowers P(Malignant)")
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    plt.tight_layout()
    return fig