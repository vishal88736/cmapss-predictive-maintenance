# C-MAPSS Predictive Maintenance & AeroGuard AI

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

An end-to-end predictive maintenance system and operational intelligence suite for commercial turbofan aircraft engines, built on the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset.

The system pairs deep reinforcement learning and temporal representation modeling with **AeroGuard AI**, an interactive real-time telemetry and fleet health monitoring operations dashboard.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │ NASA C-MAPSS Sensor Telemetry (FD001-4)│
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Preprocessing & Sensor Corruption   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  TCAE (Temporal Convolutional Autoenc) │
                      └───────────────────┬────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌─────────────────────┐             ┌─────────────────────┐
             │ TD3 RL Health Agent │             │  LQR Safety Gate    │
             └──────────┬──────────┘             └──────────┬──────────┘
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │     RUL & Maintenance Policy Engine    │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ AeroGuard AI Web Operations Dashboard  │
                      └────────────────────────────────────────┘
```

---

## Key Features

### 1. Machine Learning & Safety Control Pipeline (`devesh/training/`)
- **TCAE (Temporal Convolutional Autoencoder)**: Extracts compressed latent spatial-temporal representations from multi-channel sensor time-series.
- **TD3 (Twin Delayed DDPG)**: Learns optimal maintenance intervention policies and health degradation forecasting with actor-critic reinforcement learning.
- **LQR (Linear Quadratic Regulator) Safety Gate**: Enforces control boundaries and stability constraints to prevent erratic policy exploration and unsafe actions.
- **Evaluation & Benchmarking**: Evaluates engine degradation using Root Mean Squared Error (RMSE) and asymmetric NASA scoring functions.

### 2. AeroGuard AI Web Dashboard (`website/`)
- **Real-Time Fleet Health Monitoring**: Visualizes fleet status, degradation trajectories, and critical alert indicators.
- **Engine Telemetry & Sensor Analytics**: Deep-dive sensor inspectability across high-pressure compressor, turbine temperatures, fan speeds, and fuel flow.
- **Remaining Useful Life (RUL) Predictions**: Dynamic RUL estimation and scheduled maintenance recommendations.
- **Modern Interface**: Glassmorphism UI built with React 18, Vite, Lucide Icons, and Tailwind CSS.

---

## Repository Structure

```
.
├── data/                               # NASA C-MAPSS raw benchmark datasets
│   ├── train_FD001.txt ... FD004.txt   # Engine run-to-failure trajectories
│   ├── test_FD001.txt ... FD004.txt    # Incomplete test trajectories
│   ├── RUL_FD001.txt ... FD004.txt     # True remaining useful life labels
│   └── Damage Propagation Modeling.pdf # C-MAPSS reference documentation
│
├── devesh/training/                    # Model development & training
│   ├── src/                            # Modular Python package
│   │   ├── models/                     # Autoencoder and predictor architectures
│   │   ├── td3/                        # Actor, Critic, Replay Buffer, Reward models
│   │   ├── preprocessing.py            # Normalization, rolling windows, slicing
│   │   ├── corruption.py               # Noise injection and sensor faults
│   │   ├── lqr.py                      # LQR safety controller
│   │   ├── safety_gate.py              # Safety gating logic
│   │   ├── metrics.py                  # RMSE, NASA scoring function
│   │   └── evaluation.py               # Full evaluation routines
│   ├── cmapss_dm_tcae_td3_lqr.py       # Full training script
│   └── cmapss_ag.ipynb                 # Interactive training & analysis notebook
│
├── website/                            # AeroGuard AI web operations frontend
│   ├── src/                            # React application source code
│   │   ├── components/                 # UI components and widgets
│   │   ├── pages/                      # Dashboard pages
│   │   ├── services/                   # Telemetry & prediction services
│   │   └── data/                       # Mock engine and sensor datasets
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── .gitignore                          # Excludes venvs, node_modules, build artifacts
└── README.md
```

---

## Getting Started

### 1. AeroGuard AI Web Dashboard

To run the interactive web application locally:

```bash
cd website

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

Open `http://localhost:5173` in your browser.

### 2. Model Training & Evaluation

To run the training pipeline with Python:

```bash
cd devesh/training

# Install required Python packages
pip install torch numpy pandas scikit-learn optuna matplotlib

# Run the training script
python cmapss_dm_tcae_td3_lqr.py
```

Alternatively, open `devesh/training/cmapss_ag.ipynb` in Jupyter Notebook or Google Colab.

---

## Dataset Reference

NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS):
- **FD001**: 1 Operating Condition, 1 Fault Mode (HPC Degradation)
- **FD002**: 6 Operating Conditions, 1 Fault Mode (HPC Degradation)
- **FD003**: 1 Operating Condition, 2 Fault Modes (HPC & Fan Degradation)
- **FD004**: 6 Operating Conditions, 2 Fault Modes (HPC & Fan Degradation)

---

## License

This project is open-source under the MIT License.
