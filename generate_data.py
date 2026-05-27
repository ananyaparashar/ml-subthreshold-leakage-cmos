"""
generate_data.py
nodes
  45nm  - PTM High Performance (Metal Gate / High-K / Strained-Si), Vdd = 1.0V
  65nm  — PTM Standard Bulk,                                         Vdd = 1.0V
  90nm  — PTM Standard Bulk,                                         Vdd = 1.2V

sweep grid
  L     - node-specific values
  W     - 0.2, 0.5, 1.0, 2.0, 5.0, 10.0  µm
  T     - 250, 275, 300, 325, 350, 375, 400  K
  Vgs   - 0.00 to 0.44 V  in 10 mV steps  
  Vds   - 0.05, 0.1, 0.3, 0.5, 0.9, Vdd  V

"""

import argparse
import subprocess
import re
import tempfile
import os
import numpy as np
import pandas as pd

NGSPICE_BIN = "ngspice"

W_vals   = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]) * 1e-6
T_vals   = np.arange(250, 401, 25)
Vgs_vals = np.round(np.arange(0.0, 0.45, 0.01), 4)

NODE_CONFIG = {
    45: {
        "model_file": "bsim4_models/45nm_HP_bulk.pm",
        "output_csv": "data/45nm_nmos_bsim4.csv",
        "vdd":        1.0,
        # L values in nm - stay within PTM-valid range for 45nm HP
        "L_vals_nm":  np.array([45, 55, 65, 80, 100, 130, 180, 250]),
    },
    65: {
        "model_file": "bsim4_models/65nm_bulk.pm",
        "output_csv": "data/65nm_nmos_bsim4.csv",
        "vdd":        1.0,
        "L_vals_nm":  np.array([65, 90, 130, 180, 250, 350]),
    },
    90: {
        "model_file": "bsim4_models/90nm_bulk.pm",
        "output_csv": "data/90nm_nmos_bsim4.csv",
        "vdd":        1.2,
        "L_vals_nm":  np.array([90, 130, 180, 250, 350, 500]),
    },
}


def build_netlist(L_m, W_m, T_kelvin, Vgs, Vds, model_file):
    T_celsius = T_kelvin - 273.15
    return (
        f"* BSIM4 DC operating point\n"
        f".temp {T_celsius:.2f}\n"
        f"M1 vd vg 0 0 nmos L={L_m:.4e} W={W_m:.4e}\n"
        f"Vgs vg 0 DC {Vgs:.4f}\n"
        f"Vds vd 0 DC {Vds:.4f}\n"
        f".include {model_file}\n"
        f".op\n"
        f".control\n"
        f"  run\n"
        f"  print I(Vds)\n"
        f".endc\n"
        f".end\n"
    )


def run_ngspice(netlist_str):
    
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sp", delete=False
    ) as f:
        f.write(netlist_str)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [NGSPICE_BIN, "-b", tmp_path],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)


def parse_current(output):
    
    match = re.search(
        r"i\(vds\)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        output, re.IGNORECASE,
    )
    if match:
        return abs(float(match.group(1)))
    return None


def simulate_node(node_nm):
    cfg = NODE_CONFIG[node_nm]
    L_vals  = cfg["L_vals_nm"] * 1e-9
    vdd     = cfg["vdd"]
    Vds_vals = np.array([0.05, 0.1, 0.3, 0.5, 0.9, vdd])
    Vds_vals = np.unique(np.round(Vds_vals, 4))

    total = len(L_vals) * len(W_vals) * len(T_vals) * len(Vgs_vals) * len(Vds_vals)
    print(f"\n[{node_nm}nm] Starting {total:,} simulations...")

    rows  = []
    count = 0
    fails = 0

    for L in L_vals:
        for W in W_vals:
            for T in T_vals:
                for Vgs in Vgs_vals:
                    for Vds in Vds_vals:
                        netlist = build_netlist(
                            L, W, T, Vgs, Vds, cfg["model_file"]
                        )
                        output = run_ngspice(netlist)
                        Id     = parse_current(output)
                        count += 1

                        if count % 500 == 0:
                            pct = 100 * count / total
                            print(f"  {count:>7,}/{total:,}  ({pct:5.1f}%)  "
                                  f"rows={len(rows)}  fails={fails}")

                        if Id is not None and Id > 1e-20:
                            rows.append({
                                "L_nm":       round(L * 1e9, 4),
                                "W_um":       round(W * 1e6, 4),
                                "T_K":        float(T),
                                "Vgs_V":      float(Vgs),
                                "Vds_V":      float(Vds),
                                "log10_Id_A": float(np.log10(Id)),
                            })
                        else:
                            fails += 1

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(cfg["output_csv"], index=False)
    print(f"[{node_nm}nm] Done — {len(df):,} valid rows saved to {cfg['output_csv']}")
    print(f"[{node_nm}nm] {fails} simulations returned no current (clipped/failed)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate BSIM4 datasets via Ngspice")
    parser.add_argument(
        "--node", type=int, choices=[45, 65, 90],
        help="Run a single node only (default: all three)"
    )
    args = parser.parse_args()

    nodes = [args.node] if args.node else [45, 65, 90]
    for n in nodes:
        simulate_node(n)


if __name__ == "__main__":
    main()
