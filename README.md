# Aagjuuk: Continuous 3D Thermo-Mechanical Edge Solver for Anisotropic Silicon Interfaces

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aagjuuk.streamlit.app/)

Aagjuuk is a coordinate-based, real-time (sub-5ms) numerical solver designed to compute continuous 3D coupled thermo-mechanical stress fields in microelectronic substrates. 

By replacing traditional grid-discretization methods (mesh-bound Finite Element Analysis) with multi-scale coordinate projections, the engine evaluates stress, thermal, and displacement fields over continuous domains. This architecture enables online, closed-loop actuator calibration and non-destructive defect localization directly on edge compute nodes.

---

## 1. Mathematical Formulation

Aagjuuk models the direct physical coupling between the **3D Fourier Heat Conduction Equation** and the **Anisotropic Navier-Cauchy Equations of Elasticity** for cubic crystalline lattices (e.g., single-crystal silicon).

### 1.1 Thermal Domain
The temperature field $T(\mathbf{x}, t)$ across the continuous domain is governed by the transient heat equation:

$$\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa \nabla T) + Q$$

where:
* $\rho$ represents material density ($\text{kg/m}^3$).
* $C_p$ is the specific heat capacity ($\text{J/kg}\cdot\text{K}$).
* $\kappa$ represents the second-rank symmetric thermal conductivity tensor ($\text{W/m}\cdot\text{K}$).
* $Q$ is the volumetric heat source term representing localized laser absorption ($\text{W/m}^3$).

### 1.2 Structural Mechanics Domain
The resulting displacement field $\mathbf{u} = [u, v, w]^T$ is resolved using the Navier-Cauchy equations of motion under static equilibrium conditions:

$$\frac{\partial \sigma_{ij}}{\partial x_j} + F_i = 0$$

The second-rank stress tensor $\sigma_{ij}$ is coupled to the thermal strain field through the generalized Duhamel-Neumann relation:

$$\sigma_{ij} = C_{ijkl} \left( \varepsilon_{kl} - \alpha_{kl} \Delta T \right)$$

where:
* $C_{ijkl}$ is the fourth-rank anisotropic stiffness tensor (configured with cubic symmetry parameters $C_{11}$, $C_{12}$, and $C_{44}$).
* $\varepsilon_{kl} = \frac{1}{2} \left( \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right)$ is the infinitesimal strain tensor.
* $\alpha_{kl}$ represents the thermal expansion tensor ($\text{K}^{-1}$).
* $\Delta T = T - T_{\text{reference}}$ is the temperature offset.

---

## 2. Core Architectural Components

The repository is partitioned into isolated, modular subsystems optimized for low-latency execution:
