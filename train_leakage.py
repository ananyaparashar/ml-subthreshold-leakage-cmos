import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def generate_subthreshold_data(n=60000, seed=7):
    
    rng = np.random.default_rng(seed)

    # Physical constants
    k = 1.380649e-23
    q = 1.602176634e-19

    # Sample ranges
    # Units:
    # L, W: meters; tox: meters; V: volts; T: kelvin
    L = rng.uniform(45e-9, 300e-9, n)      # 45nm to 300nm
    W = rng.uniform(0.2e-6, 10e-6, n)      # 0.2um to 10um
    tox = rng.uniform(0.8e-9, 3.0e-9, n)   # 0.8nm to 3nm
    T = rng.uniform(250, 400, n)           # 250K to 400K

    Vgs = rng.uniform(0.0, 0.4, n)         # subthreshold region
    Vds = rng.uniform(0.05, 1.0, n)

    # Threshold voltage distribution 
    # Vth shifts slightly with L and tox 
    Vth0 = rng.normal(0.35, 0.04, n)  # mean 0.35V
    # short-channel lowers Vth a bit
    Vth_L = -0.08 * np.exp(-(L - 45e-9) / 60e-9)
    # thinner tox lowers Vth a bit
    Vth_tox = -0.03 * (3e-9 - tox) / (3e-9 - 0.8e-9)
    Vth = Vth0 + Vth_L + Vth_tox

    # Subthreshold swing factor n (1.1 to 1.7)
    n_factor = rng.uniform(1.1, 1.7, n)

    Vt = (k * T) / q  # thermal voltage

    # DIBL coefficient
    dibl = 0.04 + 0.12 * np.exp(-(L - 45e-9) / 50e-9) + 0.02 * (3e-9 - tox) / (3e-9 - 0.8e-9)
    dibl = np.clip(dibl, 0.03, 0.22)

    # I0 scaling: proportional to W/L and exp(-tox)
    I0_base = 1e-7  # A (sets scale)
    I0 = I0_base * (W / L) * np.exp(-(tox - 0.8e-9) / 1.2e-9)
    I0 *= rng.lognormal(mean=0.0, sigma=0.25, size=n)

    # Effective Vth with random variation 
    Vth_eff = Vth + rng.normal(0.0, 0.012, n)

    # Subthreshold current model
    # Core exponential
    expo = (Vgs - Vth_eff) / (n_factor * Vt)
    # Add DIBL as Vth reduction via Vds: Vth_eff -= dibl*Vds
    expo_dibl = (Vgs - (Vth_eff - dibl * Vds)) / (n_factor * Vt)

    # Drain term 
    drain_term = 1.0 - np.exp(-Vds / np.maximum(Vt, 1e-3))

    Id = I0 * np.exp(expo_dibl) * drain_term

    # Clip to avoid log underflow/overflow
    Id = np.clip(Id, 1e-18, 1e-2)

    df = pd.DataFrame({
        "L_nm": L * 1e9,
        "W_um": W * 1e6,
        "tox_nm": tox * 1e9,
        "T_K": T,
        "Vgs_V": Vgs,
        "Vds_V": Vds,
        "Vth_V": Vth,
        "n": n_factor,
        "dibl": dibl,
        "log10_Id_A": np.log10(Id)
    })
    return df


def train_and_evaluate(df):
    # Features and target
    X = df[["L_nm", "W_um", "tox_nm", "T_K", "Vgs_V", "Vds_V", "Vth_V", "n", "dibl"]]
    y = df["log10_Id_A"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge (scaled)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "HistGBR": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=350, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, max_depth=18, n_jobs=-1, random_state=42
        )
    }

    results = {}
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        mae = mean_absolute_error(y_test, yhat)
        r2 = r2_score(y_test, yhat)
        results[name] = {"MAE_log10A": mae, "R2": r2}
        preds[name] = (y_test.to_numpy(), yhat)

    return results, preds


def plot_parity(y_true, y_pred, title):
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.25)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True log10(Id) [A]")
    plt.ylabel("Pred log10(Id) [A]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    df = generate_subthreshold_data(n=60000, seed=7)
    results, preds = train_and_evaluate(df)

    print("\n=== Results (test set) ===")
    for k, v in results.items():
        print(f"{k:16s} | MAE = {v['MAE_log10A']:.4f} decades | R2 = {v['R2']:.4f}")

    # Plot parity for best model by MAE
    best = min(results.items(), key=lambda kv: kv[1]["MAE_log10A"])[0]
    y_true, y_hat = preds[best]
    plot_parity(y_true, y_hat, f"Parity Plot — {best}")

    # temperature sweep at fixed geometry & bias
    # pick representative device
    L_nm, W_um, tox_nm = 60.0, 1.0, 1.2
    Vgs, Vds, Vth, nfac, dibl = 0.1, 0.9, 0.33, 1.35, 0.12
    T_sweep = np.linspace(250, 400, 90)

    Xs = pd.DataFrame({
        "L_nm": np.full_like(T_sweep, L_nm),
        "W_um": np.full_like(T_sweep, W_um),
        "tox_nm": np.full_like(T_sweep, tox_nm),
        "T_K": T_sweep,
        "Vgs_V": np.full_like(T_sweep, Vgs),
        "Vds_V": np.full_like(T_sweep, Vds),
        "Vth_V": np.full_like(T_sweep, Vth),
        "n": np.full_like(T_sweep, nfac),
        "dibl": np.full_like(T_sweep, dibl),
    })

    # Retrain best model on full data for plotting trend
    X = df[["L_nm", "W_um", "tox_nm", "T_K", "Vgs_V", "Vds_V", "Vth_V", "n", "dibl"]]
    y = df["log10_Id_A"]

    if "Ridge" in best:
        best_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif "HistGBR" in best:
        best_model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=350, random_state=42)
    else:
        best_model = RandomForestRegressor(n_estimators=250, max_depth=18, n_jobs=-1, random_state=42)

    best_model.fit(X, y)
    yT = best_model.predict(Xs)

    plt.figure()
    plt.plot(T_sweep, yT)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pred log10(Id) [A]")
    plt.title("Predicted subthreshold leakage vs Temperature (fixed device/bias)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
