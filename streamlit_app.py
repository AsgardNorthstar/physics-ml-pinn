import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from aagjuuk_control_loop import ThermalStressController
from aagjuuk_inverse_solver import locate_internal_defect

st.set_page_config(page_title="Aagjuuk Multiphysics", layout="wide")

st.title("Aagjuuk Labs: Real-Time Multi-Material Edge Physics Solver")
st.write("---")

# Sidebar - Material Configurations
st.sidebar.header("Material Physical Tensors")
material_preset = st.sidebar.selectbox("Substrate Preset", ["Single-Crystal Silicon (Anisotropic)", "Pure Copper (Isotropic)", "Gallium Nitride (GaN)"])

if material_preset == "Single-Crystal Silicon (Anisotropic)":
    c11 = st.sidebar.slider("Stiffness Tensor C11 (GPa)", 100.0, 200.0, 165.7)
    c12 = st.sidebar.slider("Stiffness Tensor C12 (GPa)", 40.0, 100.0, 63.9)
    cte = st.sidebar.number_input("CTE (x10^-6 / K)", value=2.6)
else:
    c11 = st.sidebar.slider("Stiffness Tensor C11 (GPa)", 100.0, 200.0, 150.0)
    c12 = st.sidebar.slider("Stiffness Tensor C12 (GPa)", 40.0, 100.0, 80.0)
    cte = st.sidebar.number_input("CTE (x10^-6 / K)", value=16.5 if material_preset == "Pure Copper (Isotropic)" else 5.6)

st.sidebar.write("---")
st.sidebar.header("Advanced Inverse NDT Diagnostics")
enable_defect_detection = st.sidebar.checkbox("Real-Time Micro-Defect Localization", value=True)

# Generate baseline physical grid coordinates
num_points_axis = 10
grid_coords = np.linspace(0, 1, num_points_axis)
x_grid, y_grid, z_grid = np.meshgrid(grid_coords, grid_coords, grid_coords)
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()
z_flat = z_grid.flatten()
grid_nodes = np.column_stack((x_flat, y_flat, z_flat))

# Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Dynamic Telemetry & Diagnostics")
    st.info("System Model Status: ACTIVE on Virtual Inference Matrix.")
    
    st.write("### Microscopic Sub-Surface Diagnostics")
    diagnostic_placeholder = st.empty()
    defect_location_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    if enable_defect_detection:
        diagnostic_placeholder.success("AAGJUUK INVERSE SOLVER: ONLINE")
    else:
        diagnostic_placeholder.warning("Inverse Solver Offline")

    run_sim = st.button("RUN INVERSE DIAGNOSTIC LOOP", type="primary")

with col2:
    st.subheader("3D Real-Time Spatial Tomography")
    plot_placeholder = st.empty()

    if run_sim:
        # 1. Simulate a hidden defect occurring deep inside the substrate at coords [0.4, 0.6, 0.3]
        true_defect_loc = np.array([0.4, 0.6, 0.3])
        
        # Calculate mock sensor heat dissipation affected by the hidden structural defect
        dist_from_defect = np.sqrt(np.sum((grid_nodes - true_defect_loc)**2, axis=1))
        temp_profile = np.exp(-10 * np.sqrt((x_flat-0.5)**2 + (y_flat-0.5)**2 + (z_flat-0.5)**2))
        
        # Microscopic localized thermal anomaly caused by defect
        temp_profile += (0.8 / (dist_from_defect + 0.15)) * (cte / 10.0)
        
        # 2. Feed anomaly profile to inverse engine to locate it mathematically
        pred_loc, confidence = locate_internal_defect(temp_profile, grid_nodes, c11)
        
        # Render 3D Plot displaying BOTH the thermal profile and the localized defect node
        fig = plt.figure(figsize=(10, 55 if 'axis_height' in locals() else 4.5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot continuous temperature field
        sc = ax.scatter(x_flat, y_flat, z_flat, c=temp_profile, cmap='coolwarm', s=10, alpha=0.4)
        
        # Plot the exact detected defect point identified by the inverse engine
        if enable_defect_detection and pred_loc is not None:
            ax.scatter(pred_loc[0], pred_loc[1], pred_loc[2], color='red', s=180, marker='*', label='Detected Internal Crack/Void')
            ax.legend(loc="upper left")
            
            defect_location_placeholder.metric(label="Predicted Defect Location (X, Y, Z)", value=f"[{pred_loc[0]:.2f}, {pred_loc[1]:.2f}, {pred_loc[2]:.2f}]")
            confidence_placeholder.metric(label="Inverse Localization Accuracy Confidence", value=f"{confidence:.1f} %")
        
        ax.set_title("Volumetric State & Real-Time NDT Defect Scan")
        fig.colorbar(sc, ax=ax, shrink=0.5, label='State Value')
        
        plot_placeholder.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Click 'RUN INVERSE DIAGNOSTIC LOOP' to simulate internal damage and trigger real-time physical tomography.")
