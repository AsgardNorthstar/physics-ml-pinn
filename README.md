# Aagjuuk Labs: Real-Time Multi-Material Edge Physics Solver

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aagjuuk.streamlit.app/)

**Production Engine Demo:** [https://aagjuuk.streamlit.app/](https://aagjuuk.streamlit.app/)

---

Aagjuuk is an industry-first, high-performance Scientific Machine Learning (SciML) platform designed to solve coupled 3D thermo-mechanical stress and deformation fields on-the-fly (< 5ms) for advanced semiconductor packaging and microelectronics.

By bypassing traditional grid-discretization methods (mesh-bound Finite Element Analysis like ANSYS or COMSOL), Aagjuuk evaluates the continuous physical state directly over continuous space-time coordinates $(x, y, z, t)$ while enabling **Active Closed-Loop Hardware Control**, **Inverse Defect Localization**, and **Real-Time Online Model Adaptation**.

---

## 🚀 Key Technological Breakthroughs

| Feature | Legacy Solutions (ANSYS/COMSOL) | Aagjuuk SciML Engine | Business Impact |
| :--- | :--- | :--- | :--- |
| **Inference Speed** | Minutes to Hours (Mesh-Bound) | **< 4.0 Milliseconds** | Enables real-time, on-tool defect prevention during manufacturing. |
| **Inverse Diagnostics** | Requires physical cross-sectioning (Destructive) | **On-the-Fly 3D Localization** | Non-destructive, sub-surface crack/void mapping in real-time. |
| **Control Integration** | Open-loop simulation only | **Active PID Laser Control** | Dynamically dials back laser/thermal power to keep stress within safe parameters. |
| **Operational Lifespan** | Static models drift and fail over time | **Online Edge Self-Calibration** | Runs mini-gradient descent steps in-browser to adapt to structural wear. |

---

## 📁 Repository Architecture

* `streamlit_app.py`: Real-time interactive 3D control deck displaying thermal gradients, stress fields, and localized defect coordinates.
* `aagjuuk_inverse_solver.py`: The mathematical inverse engine analyzing sensor anomalies to back-calculate 3D defect points.
* `aagjuuk_control_loop.py`: Proportional-Integral-Derivative (PID) controller translating live stress estimations into physical laser power adjustments.
* `aagjuuk_online_adapter.py`: Real-time SGD backpropagation routine optimizing neural weights directly in the user's browser.
* `models/aagjuuk_anisotropic_3d.py`: Continuous 3D solver leveraging fourth-rank Silicon stiffness tensors ($C_{11}$, $C_{12}$, $C_{44}$).
* `models/aagjuuk_self_adaptive_pinn.py`: PINN custom training loop featuring dynamic loss weighting to defeat gradient pathologies.

---

## 🛠️ Developer Quick Start

### Install Package Dependencies
```bash
pip install -r requirements.txt
