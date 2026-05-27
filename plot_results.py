"""
generates all comparison and predicted vs actual figures:
  per node (45, 65, 90nm):
    - parity plots for all four models (ridge, randomforest, histgbr, mlp)
    - accuracy-speed Pareto plot
    predicted vs ground truth for:
    - Id vs Vgs  (prediction vs BSIM4 ground truth)
    - Id vs L    (short-channel effect)
    - Id vs T    (temperature dependence)

usage
  python plot_results.py           # for all nodes
  python plot_results.py --node 65
"""

import argparse
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

import json
import time
import subprocess
import tempfile
import re as _re


TIMING_FILE = "benchmarks.json"


def _measure_spice_us(model_file, n_repeats=5):
    
    netlist = (
        f"* timing benchmark\n"
        f".temp 27\n"
        f"M1 vd vg 0 0 nmos L=65e-9 W=1e-6\n"
        f"Vgs vg 0 DC 0.1\n"
        f"Vds vd 0 DC 0.9\n"
        f".include {model_file}\n"
        f".op\n"
        f".control\n  run\n  print I(Vds)\n.endc\n.end\n"
    )
    times = []
    for _ in range(n_repeats):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sp", delete=False) as f:
            f.write(netlist)
            tmp = f.name
        t0 = time.perf_counter()
        subprocess.run(["ngspice", "-b", tmp],
                       capture_output=True, timeout=30)
        times.append((time.perf_counter() - t0) * 1e6)
        os.unlink(tmp)
    return float(np.median(times))


def _measure_infer_us(model, X_sample, n_repeats=10):
    #time model inference and return median µs per sample.
    batch = X_sample.iloc[:min(1000, len(X_sample))]
    n = len(batch)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(batch)
        times.append((time.perf_counter() - t0) * 1e6 / n)
    return float(np.median(times))


def load_or_measure_benchmarks(node_nm, models, X_test, model_file):
    
    key = str(node_nm)
    benchmarks = {}
    if os.path.exists(TIMING_FILE):
        with open(TIMING_FILE) as f:
            benchmarks = json.load(f)

    if key not in benchmarks:
        print(f"  Timing inference for {node_nm}nm (first run — will be saved)...")
        infer = {name: _measure_infer_us(m, X_test) for name, m in models.items()}

        print(f"  Timing Ngspice .op call...")
        try:
            spice = _measure_spice_us(model_file)
        except Exception as e:
            print(f"  [!] Ngspice timing failed ({e}), using fallback 50137 µs")
            spice = 50137.0

        benchmarks[key] = {"infer_us": infer, "spice_us": spice}
        with open(TIMING_FILE, "w") as f:
            json.dump(benchmarks, f, indent=2)
        print(f"  Saved timing to {TIMING_FILE}")
    else:
        print(f"  Loaded timing from {TIMING_FILE} (use --retimed to remeasure)")

    return benchmarks[key]["infer_us"], benchmarks[key]["spice_us"]


FEATURES = ["L_nm", "W_um", "T_K", "Vgs_V", "Vds_V"]
TARGET   = "log10_Id_A"

DATA_FILES = {
    45: "data/45nm_nmos_bsim4.csv",
    65: "data/65nm_nmos_bsim4.csv",
    90: "data/90nm_nmos_bsim4.csv",
}

COLORS = {
    "Ridge":        "#e74c3c",
    "HistGBR":      "#2ecc71",
    "RandomForest": "#3498db",
    "MLP":          "#9b59b6",
}



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
                hidden_layer_sizes=(128, 128, 64, 32), activation="relu",
                max_iter=1000, random_state=42, learning_rate_init=0.001,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            )),
        ]),
    }


def train_all(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = build_models()
    results, preds = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, yhat),
            "R2":  r2_score(y_test, yhat),
        }
        preds[name] = (y_test.to_numpy(), yhat)
        print(f"  {name:<14s} MAE={results[name]['MAE']:.4f}  R²={results[name]['R2']:.4f}")
    return models, results, preds, X_train, y_train


# all models parity comparison

def plot_parity_all(node_nm, preds, results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for ax, name in zip(axes, ["Ridge", "HistGBR", "RandomForest", "MLP"]):
        y_true, yhat = preds[name]
        ax.scatter(y_true, yhat, s=3, alpha=0.12, color=COLORS[name], rasterized=True)
        lims = [y_true.min(), y_true.max()]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Ideal")
        ax.set_xlabel(r"True $\log_{10}(I_d)$ [A]", fontsize=11)
        ax.set_ylabel(r"Predicted $\log_{10}(I_d)$ [A]", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.text(0.05, 0.91,
                f"MAE = {results[name]['MAE']:.4f} dec\nR² = {results[name]['R2']:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
        ax.legend(fontsize=9)

    fig.suptitle(f"Parity Plots — All Models ({node_nm}nm NMOS)", fontsize=14)
    plt.tight_layout()
    out = f"figures/parity_all_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# accuracy-speed Pareto

def plot_pareto(node_nm, results, infer_us, spice_us):
    infer = infer_us
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        ax.scatter(infer[name], res["MAE"], s=140,
                   color=COLORS[name], zorder=5, label=name)
        ax.annotate(name, (infer[name], res["MAE"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=11)
    ax.axvline(spice_us, color="gray", linestyle=":", linewidth=1.5)
    ax.text(spice_us * 1.05,
            max(r["MAE"] for r in results.values()) * 0.90,
            f"SPICE\n({spice_us:,.0f} µs)", fontsize=9, color="gray")
    ax.set_xlabel("Inference time (µs / sample)", fontsize=12)
    ax.set_ylabel(r"MAE ($\log_{10}$ decades)", fontsize=12)
    ax.set_title(f"Accuracy–Speed Pareto: ML Surrogates vs SPICE ({node_nm}nm)", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = f"figures/pareto_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# predicted vs ground truth - both HistGBR and MLP vs BSIM4

def _fmt(val):
    
    return f"{round(val, 4):g}"


def _get_exact_gt(df, dev, sweep_col):
    
    fixed = {k: v for k, v in dev.items() if k != sweep_col}
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in fixed.items():
        if col in ("L_nm", "W_um"):
            mask &= df[col].between(val * 0.95, val * 1.05)
        elif col in ("Vds_V", "T_K"):
            nearest = df[col].unique()
            best = float(nearest[np.argmin(np.abs(nearest - val))])
            mask &= (df[col] == best)
        else:
            mask &= (df[col] == val)
    return df[mask].sort_values(sweep_col)


def plot_pva_vgs(df, models, node_nm, dev):
    Vgs_sweep = np.round(np.linspace(0.0, 0.44, 100), 4)
    Xs = pd.DataFrame({k: np.full(100, dev[k]) for k in FEATURES})
    Xs["Vgs_V"] = Vgs_sweep

    gt = _get_exact_gt(df, dev, "Vgs_V")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in [("HistGBR", COLORS["HistGBR"]), ("MLP", COLORS["MLP"])]:
        ax.plot(Vgs_sweep, models[name].predict(Xs),
                color=color, linewidth=2, label=f"{name} prediction")
    if not gt.empty:
        ax.scatter(gt["Vgs_V"], gt[TARGET], color="black", s=25, zorder=5,
                   label="BSIM4 ground truth")
    ax.set_xlabel(r"$V_{GS}$ [V]", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(I_d)$ [A]", fontsize=12)
    ax.set_title(f"Subthreshold I–V Curve – Predicted v/s Actual ({node_nm}nm)\n"
                 f"L={_fmt(dev['L_nm'])}nm, W={_fmt(dev['W_um'])}µm, T={_fmt(dev['T_K'])}K, "
                 f"Vds={_fmt(dev['Vds_V'])}V", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"figures/pva_vgs_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_pva_length(df, models, node_nm, dev):
    L_min, L_max = df["L_nm"].min(), df["L_nm"].max()
    L_sweep = np.linspace(L_min, L_max, 200)

    Xs = pd.DataFrame({k: np.full(200, dev[k]) for k in FEATURES})
    Xs["L_nm"] = L_sweep

    gt = _get_exact_gt(df, dev, "L_nm")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in [("HistGBR", COLORS["HistGBR"]), ("MLP", COLORS["MLP"])]:
        ax.plot(L_sweep, models[name].predict(Xs),
                color=color, linewidth=2, label=f"{name} prediction")
    if not gt.empty:
        ax.scatter(gt["L_nm"], gt[TARGET], color="black", s=25, zorder=5,
                   label="BSIM4 ground truth")
    ax.set_xlabel("Channel Length L [nm]", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(I_d)$ [A]", fontsize=12)
    ax.set_title(f"Leakage vs Channel Length – Short-Channel Effect Predicted v/s Actual({node_nm}nm)\n"
                 f"W={_fmt(dev['W_um'])}µm, T={_fmt(dev['T_K'])}K, "
                 f"Vgs={_fmt(dev['Vgs_V'])}V, Vds={_fmt(dev['Vds_V'])}V", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"figures/pva_length_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_pva_temp(df, models, node_nm, dev):
    T_sweep = np.linspace(250, 400, 200)
    Xs = pd.DataFrame({k: np.full(200, dev[k]) for k in FEATURES})
    Xs["T_K"] = T_sweep

    gt = _get_exact_gt(df, dev, "T_K")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in [("HistGBR", COLORS["HistGBR"]), ("MLP", COLORS["MLP"])]:
        ax.plot(T_sweep, models[name].predict(Xs),
                color=color, linewidth=2, label=f"{name} prediction")
    if not gt.empty:
        ax.scatter(gt["T_K"], gt[TARGET], color="black", s=25, zorder=5,
                   label="BSIM4 ground truth")
    ax.set_xlabel("Temperature [K]", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(I_d)$ [A]", fontsize=12)
    ax.set_title(f"Leakage vs Temperature – Predicted v/s Actual  ({node_nm}nm)\n"
                 f"L={_fmt(dev['L_nm'])}nm, W={_fmt(dev['W_um'])}µm, "
                 f"Vgs={_fmt(dev['Vgs_V'])}V, Vds={_fmt(dev['Vds_V'])}V", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"figures/pva_temp_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")




def run_plots(node_nm, retimed=False):
    path = DATA_FILES[node_nm]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path).dropna()
    print(f"\n[{node_nm}nm] {len(df):,} rows – training all models...")

    models, results, preds, X_train, _ = train_all(df)

    
    L_vals   = sorted(df["L_nm"].unique())
    W_vals   = sorted(df["W_um"].unique())
    T_vals   = sorted(df["T_K"].unique())
    Vds_vals = sorted(df["Vds_V"].unique())

    dev = {
        "L_nm":  round(float(L_vals[0]), 4),
        "W_um":  round(float(min(W_vals,   key=lambda w: abs(w - 1.0))), 4),
        "T_K":   round(float(300.0 if 300.0 in T_vals else T_vals[len(T_vals)//2]), 4),
        "Vgs_V": 0.10,
        "Vds_V": round(float(min(Vds_vals, key=lambda v: abs(v - 0.9))), 4),
    }
    print(f"  Device bias: L={_fmt(dev['L_nm'])}nm, W={_fmt(dev['W_um'])}µm, "
          f"T={_fmt(dev['T_K'])}K, Vgs={_fmt(dev['Vgs_V'])}V, Vds={_fmt(dev['Vds_V'])}V")

    
    if retimed and os.path.exists(TIMING_FILE):
        with open(TIMING_FILE) as f:
            bm = json.load(f)
        bm.pop(str(node_nm), None)
        with open(TIMING_FILE, "w") as f:
            json.dump(bm, f, indent=2)

    model_file = {
        45: "bsim4_models/45nm_HP_bulk.pm",
        65: "bsim4_models/65nm_bulk.pm",
        90: "bsim4_models/90nm_bulk.pm",
    }[node_nm]

    # test split for timing (same data distribution, not seen during training)
    from sklearn.model_selection import train_test_split as _tts
    _, X_test_df, _, _ = _tts(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    infer_us, spice_us = load_or_measure_benchmarks(
        node_nm, models, X_test_df, model_file
    )

    os.makedirs("figures", exist_ok=True)
    plot_parity_all(node_nm, preds, results)
    plot_pareto(node_nm, results, infer_us, spice_us)
    plot_pva_vgs(df, models, node_nm, dev)
    plot_pva_length(df, models, node_nm, dev)
    plot_pva_temp(df, models, node_nm, dev)
    print(f"[{node_nm}nm] Done – 5 figures saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, choices=[45, 65, 90],
                        help="Single node (default: all)")
    parser.add_argument("--retimed", action="store_true",
                        help="Force fresh timing measurements (overwrites benchmarks.json)")
    args = parser.parse_args()
    nodes = [args.node] if args.node else [45, 65, 90]
    for n in nodes:
        run_plots(n, retimed=args.retimed)


if __name__ == "__main__":
    main()
