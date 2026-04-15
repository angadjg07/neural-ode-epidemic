# Neural ODE Epidemic Simulator

A purely autonomous scientific machine learning project forecasting seasonal epidemic dynamics using Continuous Depth Physics-Informed ML.

## Overview
This platform explores a **Universal Differential Equation** (UDE) approach for predicting epidemic propagation. Classical math models strictly depend on static transition constants (transmission and recovery). By marrying strict epidemiological bounds explicitly mapped inside `torchdiffeq` ODE wrappers, we allow neural networks safely parameterized by seasonal Fourier time features to compute dynamic transmission variations iteratively against structural physical equations.

## Results

| Model | RMSE | MAE | Peak Timing Error (Weeks) |
|---|---|---|---|
| Classical SIR | 0.001411 | 0.001293 | 0 |
| Latent Neural ODE | 0.009061 | 0.009043 | 0 |
| Hybrid UDE | 0.005570 | 0.002149 | 0 |

*Values extrapolated from testing trajectory tracking empirical JHU Covid reporting data.*

## SciML Concepts

* **Neural ODEs** ([Chen et al. 2018](https://arxiv.org/abs/1806.07366)): Iterative parameter models optimizing network derivative parameters continuously through solvers leveraging adjoint sensitivity backward passes.
* **Physics-Informed Bounds**: Strict explicit restrictions preventing mass-creation operations over S, I, R sets globally.
* **Universal Differential Equations** ([Rackauckas et al. 2020](https://arxiv.org/abs/2001.04385)): Retaining purely defined system graphs while enabling dynamically neural components to compensate and learn parameterized scalar differentials ($\beta, \gamma$) automatically from data.

## Quickstart

```bash
# Setup Environment
git clone https://github.com/angadjg07/neural-ode-epidemic.git
cd neural-ode-epidemic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run Inference Evaluation and Build Demo Page
python src/evaluate.py
python results/build.py

# Optionally retrain entire stack (Takes 5-15 mins without GPU)
python src/train.py --model hybrid --epochs 500 --run_name final_demo
```
