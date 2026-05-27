"""
train_leakage.py
trains ML surrogate models for subthreshold leakage current using PTM file. Produces parity plot and accuracy-speed pareto figure.

each node is trained indepedantly 

"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

FEATURES = ["L_nm", "W_um", "T_K", "Vgs_V", "Vds_V"]
TARGET   = "log10_Id_A"

DATA_FILES = {
    45: "data/45nm_nmos_bsim4.csv",
    65: "data/65nm_nmos_bsim4.csv",
    90: "data/90nm_nmos_bsim4.csv",
}

COLORS = {
    "Ridge":         "#e74c3c",
    "HistGBR":       "#2ecc71",
    "RandomForest":  "#3498db",
    "MLP":           "#9b59b6",
}

SPICE_US = 50_137  # measured inference time for single ngspice .op call


def build_models():
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "HistGBR": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=350, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, max_depth=18, n_jobs=-1, random_state=42
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(128, 128, 64, 32),
                activation="relu",
                max_iter=1000,
                random_state=42,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ]),
    }


def train_node(node_nm):
    path = DATA_FILES[node_nm]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run generate_data.py --node {node_nm} first"
        )

    df = pd.read_csv(path).dropna()
    print(f"\n[{node_nm}nm] Loaded {len(df):,} rows from {path}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models   = build_models()
    results  = {}
    preds    = {}
    infer_us = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Inference timing — median over 5 batches of 1000 samples
        batch = X_test.iloc[:1000]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict(batch)
            times.append((time.perf_counter() - t0) * 1e6 / len(batch))
        infer_us[name] = float(np.median(times))

        yhat = model.predict(X_test)
        results[name] = {
            "MAE":  mean_absolute_error(y_test, yhat),
            "R2":   r2_score(y_test, yhat),
        }
        preds[name] = (y_test.to_numpy(), yhat)
        print(f"  {name:<14s}  MAE={results[name]['MAE']:.4f} dec  "
              f"R²={results[name]['R2']:.4f}  "
              f"infer={infer_us[name]:.3f} µs/sample")

    os.makedirs("figures", exist_ok=True)
    _plot_parity(node_nm, preds, results)
    _plot_pareto(node_nm, results, infer_us)

    return results, preds, infer_us


# plotting

def _plot_parity(node_nm, preds, results):
    best = min(results, key=lambda k: results[k]["MAE"])
    y_true, yhat = preds[best]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, yhat, s=4, alpha=0.15,
               color=COLORS[best], rasterized=True)
    lims = [y_true.min(), y_true.max()]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Ideal")
    ax.set_xlabel(r"True $\log_{10}(I_d)$ [A]",   fontsize=13)
    ax.set_ylabel(r"Predicted $\log_{10}(I_d)$ [A]", fontsize=13)
    ax.set_title(f"Parity Plot — {best} ({node_nm}nm)", fontsize=13)
    ax.text(
        0.05, 0.92,
        f"MAE = {results[best]['MAE']:.4f} dec\nR² = {results[best]['R2']:.4f}",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = f"figures/parity_plot_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def _plot_pareto(node_nm, results, infer_us):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, res in results.items():
        ax.scatter(infer_us[name], res["MAE"], s=140,
                   color=COLORS[name], zorder=5, label=name)
        ax.annotate(name, (infer_us[name], res["MAE"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=11)

    ax.axvline(SPICE_US, color="gray", linestyle=":", linewidth=1.5)
    ax.text(SPICE_US * 1.05, max(r["MAE"] for r in results.values()) * 0.92,
            f"SPICE\n({SPICE_US:,} µs)", fontsize=9, color="gray")

    ax.set_xlabel("Inference time (µs / sample)", fontsize=12)
    ax.set_ylabel(r"MAE ($\log_{10}$ decades)", fontsize=12)
    ax.set_title(f"Accuracy–Speed Pareto: ML vs SPICE ({node_nm}nm)",
                 fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = f"figures/pareto_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")




def main():
    parser = argparse.ArgumentParser(description="Train leakage surrogate models")
    parser.add_argument(
        "--node", type=int, choices=[45, 65, 90],
        help="Train on a single node only (default: all three)"
    )
    args = parser.parse_args()

    nodes = [args.node] if args.node else [45, 65, 90]
    for n in nodes:
        train_node(n)


if __name__ == "__main__":
    main()
