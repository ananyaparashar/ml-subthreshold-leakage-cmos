"""
pvt_experiment.py

PVT (Process–Voltage–Temperature) corner analysis for the trained ML models

process corners are modelled as threshold voltage shifts applied to the BSIM4 ground-truth labels:
  FF (fast) : delta Vth = -0.014 V  - more leakage
  TT (ideal)      : delta Vth =  0.000 V  (nominal)
  SS (slow) : delta Vth = +0.014 V  - less leakage

voltage corners scale Vds by +-10% around the nominal supply
temperature corners filter the dataset to T = 250 / 300 / 400 K

the heatmap shows worst-case MAE across voltage corners for each (process * temperature) combination

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["L_nm", "W_um", "T_K", "Vgs_V", "Vds_V"]
TARGET   = "log10_Id_A"

DATA_FILES = {
    45: "data/45nm_nmos_bsim4.csv",
    65: "data/65nm_nmos_bsim4.csv",
    90: "data/90nm_nmos_bsim4.csv",
}

# n*Vt at 300 K  (n = 1.3 (approx.) subthreshold swing factor, Vt=0.02585 V)
N_VT_300 = 0.0338

PROCESS_CORNERS = {
    "FF (fast)":     +0.014,
    "TT (typical)":   0.000,
    "SS (slow)":     -0.014,
}
VOLTAGE_SCALES = {
    "Vdd+10%": 1.10,
    "Vdd nom": 1.00,
    "Vdd-10%": 0.90,
}
TEMP_CORNERS = {
    "Hot (400K)":  400,
    "Nom (300K)":  300,
    "Cold (250K)": 250,
}


def run_pvt(node_nm):
    path = DATA_FILES[node_nm]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run generate_data.py --node {node_nm} first"
        )

    df = pd.read_csv(path).dropna()
    print(f"\n[{node_nm}nm] Loaded {len(df):,} rows")

    X_train, _, y_train, _ = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

    hgbr = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.08, max_iter=350, random_state=42
    )
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 128, 64, 32),
            activation="relu", max_iter=1000, random_state=42,
            learning_rate_init=0.001,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        )),
    ])

    print("  Training HistGBR and MLP on nominal data...")
    hgbr.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    results = []
    print("  Evaluating PVT corners...")

    for pname, dVth in PROCESS_CORNERS.items():
        # delta Vth - delta log10(Id):  delta Id/Id = exp(- delta Vth / n*Vt)
        delta_log10 = -dVth / (N_VT_300 * np.log(10))

        for tname, T_val in TEMP_CORNERS.items():
            df_T = df[df["T_K"] == T_val].copy()
            if df_T.empty:
                continue

            # process corner shift to ground-truth labels
            df_T[TARGET] = df_T[TARGET] + delta_log10

            for vname, vscale in VOLTAGE_SCALES.items():
                df_corner = df_T.copy()
                df_corner["Vds_V"] = (df_corner["Vds_V"] * vscale).clip(upper=1.5)

                X_c = df_corner[FEATURES]
                y_c = df_corner[TARGET]

                mae_hgbr = mean_absolute_error(y_c, hgbr.predict(X_c))
                mae_mlp  = mean_absolute_error(y_c, mlp.predict(X_c))

                results.append({
                    "Process": pname, "Temp": tname, "Voltage": vname,
                    "T_K": T_val, "dVth": dVth, "vscale": vscale,
                    "MAE_HGBR": mae_hgbr, "MAE_MLP": mae_mlp,
                })
                print(f"    {pname:15s} {tname:12s} {vname:10s}  "
                      f"HistGBR={mae_hgbr:.3f}  MLP={mae_mlp:.3f}")

    df_res = pd.DataFrame(results)
    _plot_heatmap(df_res, node_nm)
    _print_summary(df_res, node_nm)


def _plot_heatmap(df_res, node_nm):
    row_order = ["FF (fast)", "TT (typical)", "SS (slow)"]
    col_order = ["Hot (400K)", "Nom (300K)", "Cold (250K)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model_label, col in [
        (axes[0], "HistGBR", "MAE_HGBR"),
        (axes[1], "MLP",     "MAE_MLP"),
    ]:
        pivot = (
            df_res
            .pivot_table(index="Process", columns="Temp", values=col, aggfunc="max")
            .reindex(index=row_order, columns=col_order)
        )

        im = ax.imshow(
            pivot.values.astype(float),
            cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5,
        )
        ax.set_xticks(range(len(col_order)))
        ax.set_yticks(range(len(row_order)))
        ax.set_xticklabels(col_order, fontsize=10)
        ax.set_yticklabels(row_order, fontsize=10)
        ax.set_title(
            f"{model_label} — Worst-case MAE across PVT\n(log₁₀ decades)",
            fontsize=11,
        )
        for i in range(len(row_order)):
            for j in range(len(col_order)):
                val = float(pivot.values[i, j])
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > 0.25 else "black")
        plt.colorbar(im, ax=ax)

    plt.suptitle(
        f"PVT Corner Analysis — {node_nm}nm NMOS Subthreshold Leakage",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs("figures/pvt", exist_ok=True)
    out = f"figures/pvt/pvt_heatmap_{node_nm}nm.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved {out}")


def _print_summary(df_res, node_nm):
    print(f"\n PVT Summary [{node_nm}nm] ")
    print(f"  HistGBR  worst-case MAE: {df_res['MAE_HGBR'].max():.3f} decades")
    print(f"  HistGBR  best-case  MAE: {df_res['MAE_HGBR'].min():.3f} decades")
    print(f"  MLP      worst-case MAE: {df_res['MAE_MLP'].max():.3f} decades")
    print(f"  MLP      best-case  MAE: {df_res['MAE_MLP'].min():.3f} decades")


def main():
    parser = argparse.ArgumentParser(description="PVT corner analysis")
    parser.add_argument(
        "--node", type=int, choices=[45, 65, 90], default=45,
        help="Technology node to analyse (default: 45)"
    )
    args = parser.parse_args()
    run_pvt(args.node)


if __name__ == "__main__":
    main()
