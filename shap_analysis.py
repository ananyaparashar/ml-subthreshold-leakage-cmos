"""
shap_analysis.py
"""
import argparse         
import os               
import numpy as np      
import pandas as pd     
import matplotlib       
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import shap             

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["L_nm", "W_um", "T_K", "Vgs_V", "Vds_V"]

# readable names for plot axis labels
FEATURE_LABELS = {
    "L_nm":  "L (nm)",
    "W_um":  "W (µm)",
    "T_K":   "T (K)",
    "Vgs_V": "V$_{GS}$ (V)",
    "Vds_V": "V$_{DS}$ (V)",
}

TARGET = "log10_Id_A"


DATA_FILES = {
    45: "data/45nm_nmos_bsim4.csv",
    65: "data/65nm_nmos_bsim4.csv",
    90: "data/90nm_nmos_bsim4.csv",
}


SHAP_SAMPLE_SIZE = 2000

# model training
def train_histgbr(df):
    
    X = df[FEATURES]   
    y = df[TARGET]      # target — a series of log10(Id) values

   
    
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=350,
        random_state=42,
    )

    
    model.fit(X_train, y_train)

    return model, X_test   


def compute_shap(model, X_test):
    
    n = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(n, random_state=42)

    print(f"  Computing SHAP values for {n} samples...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_sample)

    return shap_values, X_sample


# plots
def plot_beeswarm(shap_values, node_nm):
    
    fig, ax = plt.subplots(figsize=(9, 5))

    shap.plots.beeswarm(
        shap_values,
        max_display=5,
        show=False,
        color_bar_label="Feature value",
    )

    ax = plt.gca()
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    new_labels = [FEATURE_LABELS.get(t, t) for t in ytick_labels]
    ax.set_yticklabels(new_labels, fontsize=11)

    ax.set_xlabel("SHAP value (impact on log$_{10}$(I$_d$) [decades])", fontsize=11)
    ax.set_title(
        f"SHAP Feature Importance —> HistGBR ({node_nm}nm NMOS)\n"
        f"Each dot = one test sample. Colour = feature value (red=high, blue=low).",
        fontsize=11,
    )

    plt.tight_layout()
    out = f"figures/SHAP_analysis/shap_beeswarm_{node_nm}nm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_mean_shap_bar(shap_values, node_nm):
   
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    
    importance = sorted(
        zip(FEATURES, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    features_sorted = [FEATURE_LABELS[f] for f, _ in importance]
    values_sorted   = [v for _, v in importance]

    
    print(f"\n  Mean |SHAP| values [{node_nm}nm]:")
    for f, v in importance:
        print(f"    {FEATURE_LABELS[f]:20s}  {v:.4f} decades")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        features_sorted[::-1],   # reverse so most important is at top
        values_sorted[::-1],
        color=["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(values_sorted)-1, -1, -1)],
        
    )

    ax.set_xlabel("Mean |SHAP value| (log$_{10}$ decades)", fontsize=11)
    ax.set_title(
        f"Feature Importance —> Mean |SHAP| ({node_nm}nm NMOS)\n"
        f"Average impact on predicted log$_{{10}}$(I$_d$) across test set",
        fontsize=11,
    )

    
    for bar, val in zip(bars, values_sorted[::-1]):
        ax.text(
            bar.get_width() + 0.005,          
            bar.get_y() + bar.get_height()/2,  
            f"{val:.3f}",
            va="center", fontsize=10,
        )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = f"figures/SHAP_analysis/shap_bar_{node_nm}nm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_dependence(shap_values, X_sample, node_nm):
    
    # find the two features with highest mean |SHAP| 
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top2_idx = np.argsort(mean_abs)[::-1][:2]   # indices of top 2
    top2_features = [FEATURES[i] for i in top2_idx]

    for feat in top2_features:
        feat_idx = FEATURES.index(feat)   # which column in shap_values

        fig, ax = plt.subplots(figsize=(7, 5))

       
        x_vals = X_sample[feat].values

        
        y_vals = shap_values.values[:, feat_idx]

        
        other_feat = [f for f in top2_features if f != feat][0]
        other_idx  = FEATURES.index(other_feat)
        color_vals = X_sample[other_feat].values

        sc = ax.scatter(
            x_vals, y_vals,
            c=color_vals,             
            cmap="plasma",            
            s=8, alpha=0.4,
            rasterized=True,          
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(FEATURE_LABELS[other_feat], fontsize=10)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        

        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=12)
        ax.set_ylabel(
            f"SHAP value for {FEATURE_LABELS[feat]}\n"
            f"(impact on log$_{{10}}$(I$_d$) [decades])",
            fontsize=11,
        )
        ax.set_title(
            f"SHAP Dependence: {FEATURE_LABELS[feat]} ({node_nm}nm)\n"
            f"Colour = {FEATURE_LABELS[other_feat]}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        
        feat_clean = feat.replace("_", "").replace(" ", "")
        out = f"figures/SHAP_analysis/shap_dependence_{feat_clean}_{node_nm}nm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def run_shap(node_nm):
    
    path = DATA_FILES[node_nm]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run generate_data.py --node {node_nm} first"
        )

    print(f"\n[{node_nm}nm] Loading data...")
    df = pd.read_csv(path).dropna()
    print(f"  {len(df):,} rows loaded")

    print(f"[{node_nm}nm] Training HistGBR...")
    model, X_test = train_histgbr(df)

    print(f"[{node_nm}nm] Computing SHAP values...")
    shap_values, X_sample = compute_shap(model, X_test)

    os.makedirs("figures/SHAP_analysis", exist_ok=True)

    print(f"[{node_nm}nm] Saving plots...")
    plot_beeswarm(shap_values, node_nm)
    plot_mean_shap_bar(shap_values, node_nm)
    plot_dependence(shap_values, X_sample, node_nm)

    print(f"[{node_nm}nm] Done.")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for leakage surrogates")
    parser.add_argument(
        "--node", type=int, choices=[45, 65, 90],
        help="Single node to analyse (default: all three)"
    )
    args = parser.parse_args()

    nodes = [args.node] if args.node else [45, 65, 90]
    for n in nodes:
        run_shap(n)


if __name__ == "__main__":
    main()
