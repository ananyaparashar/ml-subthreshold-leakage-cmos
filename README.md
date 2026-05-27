# ML Surrogate Models for Subthreshold Leakage in CMOS

Machine learning surrogate models for BSIM4-simulated subthreshold drain current across three PTM technology nodes: 45nm (High Performance), 65nm, and 90nm NMOS.

---

## Problem Statement

Subthreshold leakage - drain current when Vgs < Vth — is the dominant static power source in nanoscale CMOS. Estimating leakage across a design space (geometry, temperature, bias, process corner) requires many SPICE simulations. A single Ngspice `.op` call costs approximately 50 ms; sweeping thousands of operating points for early-stage power analysis is impractical.

This project trains ML surrogate models directly on BSIM4 simulation data, enabling leakage prediction at microsecond inference time with sub-0.02 decade MAE - **a roughly 100,000× speedup** over SPICE.

Each technology node is treated as an independent problem. Cross-node generalization is intentionally out of scope: CMOS nodes do not share a BSIM4 parameterisation, and a model trained on 45nm data carries no valid prior for 65nm behaviour.

---

## Technology Nodes

| Node | Model File | Corner | Vdd |
|------|-----------|--------|-----|
| 45nm | `bsim4_models/45nm_HP_bulk.pm` | High Performance (Metal Gate / High-K / Strained-Si) | 1.0 V |
| 65nm | `bsim4_models/65nm_bulk.pm` | Standard Bulk | 1.0 V |
| 90nm | `bsim4_models/90nm_bulk.pm` | Standard Bulk | 1.2 V |

PTM model files sourced from the Predictive Technology Model project (ptm.asu.edu), BSIM4 level 54. The 45nm node uses the High Performance corner, which features metal gate, high-κ dielectric, and strained silicon
    consistent with physical 45nm process characteristics.

---

## Dataset Generation

`generate_data.py` calls Ngspice in batch mode for each bias point, writing and parsing a temporary `.op` netlist. 

**Sweep grid:**

| Parameter | Values | Notes |
|-----------|--------|-------|
| L | node-specific | within PTM-valid range |
| W | 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 µm | |
| T | 250 - 400 K in 25 K steps | |
| Vgs | 0.00 - 0.44 V in 10 mV steps | subthreshold region |
| Vds | 0.05, 0.1, 0.3, 0.5, 0.9, Vdd | |

| Node | Valid rows | File |
|------|-----------|------|
| 45nm | 90,720 | `data/45nm_nmos_bsim4.csv` |
| 65nm | 68,040 | `data/65nm_nmos_bsim4.csv` |
| 90nm | 68,040 | `data/90nm_nmos_bsim4.csv` |

Target variable: `log10_Id_A` - base-10 logarithm of drain current in amperes. Log-space regression is appropriate because subthreshold current spans 7–8 orders of magnitude.

---

## Models

Four regressors trained per node. Features: `L_nm`, `W_um`, `T_K`, `Vgs_V`, `Vds_V`.

| Model | Config |
|-------|--------|
| Ridge | Linear baseline, standard scaling |
| HistGBR | Histogram gradient boosting, depth 6, lr 0.08, 350 iterations |
| RandomForest | 250 estimators, depth 18 |
| MLP | 128-128-64-32, ReLU, early stopping |

---

## Results

### Per-node accuracy (test set, 20% holdout)

**45nm** (90,720 rows)

| Model | MAE (decades) | R² | Inference (µs/sample) |
|-------|--------------|----|-----------------------|
| Ridge | 0.3584 | 0.9131 | 0.27 |
| HistGBR | 0.0218 | 0.9997 | 25.3 |
| RandomForest | 0.0250 | 0.9995 | 40.9 |
| **MLP** | **0.0107** | **0.9999** | **0.75** |

**65nm** (68,040 rows)

| Model | MAE (decades) | R² | Inference (µs/sample) |
|-------|--------------|----|-----------------------|
| Ridge | 0.3434 | 0.9100 | 0.28 |
| HistGBR | 0.0194 | 0.9997 | 22.5 |
| RandomForest | 0.0219 | 0.9995 | 40.6 |
| **MLP** | **0.0152** | **0.9998** | **0.74** |

**90nm** (68,040 rows)

| Model | MAE (decades) | R² | Inference (µs/sample) |
|-------|--------------|----|-----------------------|
| Ridge | 0.2810 | 0.9476 | 0.30 |
| **HistGBR** | **0.0099** | **0.9999** | 21.8 |
| RandomForest | 0.0225 | 0.9996 | 27.9 |
| MLP | 0.0196 | 0.9997 | **0.76** |

MLP is the best accuracy–speed trade-off on 45nm and 65nm (sub-0.02 decades at under 1 µs). HistGBR achieves the lowest absolute MAE on 90nm. Ridge is the baseline and confirms the problem is non-linear and has a factor of roughly 15× worse than tree/MLP models.

### PVT corner robustness (45nm)

Process corners modelled as +-14 mV Vth shifts (FF/SS), voltage corners as +-10% Vds scaling.

| Corner | HistGBR MAE | MLP MAE |
|--------|-------------|---------|
| TT, all temps | 0.022–0.028 | 0.011–0.032 |
| FF/SS, all temps | 0.172–0.189 | 0.155–0.206 |

TT-corner accuracy is near-nominal. FF/SS degradation (~0.18 decades) is expected: Vth shifts push the operating point into a distribution tail the model was not explicitly trained on. For TT-corner early-stage power estimation this surrogate is directly usable.

---

## Physical Checks

`plot_results.py` validates that the surrogate reproduces correct physical trends:

- **Id vs Vgs**: exponential subthreshold slope, correctly tracking BSIM4 ground truth
- **Id vs L**: leakage increases at shorter L - short-channel effect (DIBL and Vth roll-off)
- **Id vs T**: positive temperature coefficient in subthreshold regime, as expected from the thermal voltage term in the Boltzmann factor

---

## Repository Structure

```
bsim4_models/        PTM BSIM4 model files and Ngspice library
data/                Per-node simulation CSVs
figures/             All output plots
generate_data.py     Ngspice sweep → CSV (replace this to re-simulate)
train_leakage.py     Model training, evaluation, parity + Pareto figures
pvt_experiment.py    PVT corner analysis and heatmap
plot_results.py      Physical sanity-check plots (all models vs BSIM4)
requirements.txt     Python dependencies
```

---

## Usage

Requires Ngspice on PATH:
```bash
brew install ngspice        # macOS
sudo apt install ngspice    # Ubuntu
```

```bash
pip install -r requirements.txt

# Simulate (skip if using provided data/ CSVs)
python generate_data.py --node 45

# Train and evaluate
python train_leakage.py --node 45

# PVT analysis
python pvt_experiment.py --node 45

# Sanity and comparison plots
python plot_results.py --node 45
```

Run without `--node` to process all three nodes.
