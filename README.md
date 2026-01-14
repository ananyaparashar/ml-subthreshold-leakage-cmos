# ml-subthreshold-leakage-cmos
ML-based modeling of subthreshold leakage current in CMOS devices
# ML-Based Subthreshold Leakage Modeling in CMOS Devices

This project presents a physics-inspired, data-driven approach to modeling subthreshold leakage current in CMOS devices using machine learning.

## Overview
Subthreshold leakage current exhibits strong exponential dependence on device geometry, bias conditions, and temperature. Traditional compact models rely on analytical expressions with technology-specific parameters. In this work, we explore whether supervised machine learning models can accurately learn this nonlinear mapping directly from data.

## Methodology
- A synthetic dataset is generated using a physics-inspired subthreshold current model incorporating:
  - Channel length, width, and oxide thickness
  - Gate and drain bias
  - Threshold voltage variation
  - Subthreshold swing factor and DIBL
  - Temperature dependence via thermal voltage
- The target variable is the logarithm of drain current (log₁₀(Id)).
- Multiple regression models are trained and evaluated:
  - Ridge Regression (baseline)
  - Random Forest Regressor
  - Histogram Gradient Boosting Regressor (best-performing)

## Results
- The best-performing model achieves low mean absolute error (MAE) in log-current, indicating accurate prediction over multiple decades.
- A parity plot demonstrates close agreement between predicted and true leakage current.
- A temperature sweep at fixed device geometry confirms that the trained model preserves physical monotonicity with temperature.

## Key Plots
- Predicted vs True log₁₀(Id) parity plot
- Subthreshold leakage variation with temperature

## Tools and Libraries
- Python (NumPy, Pandas, Matplotlib)
- Scikit-learn

## Motivation
This work serves as a proof-of-concept for replacing or augmenting traditional compact leakage models with machine learning, with potential applications in low-power VLSI design and technology exploration.

