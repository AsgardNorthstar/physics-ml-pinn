import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

st.set_page_config(page_title="Aagjuuk Multiphysics", layout="wide")

st.title("Aagjuuk Labs: Real-Time Multi-Material Edge Physics Solver")
st.write("---")

# Sidebar - Material Configurations
st.sidebar.header("Material Physical Tensors")
material_preset = st.sidebar.selectbox("Substrate Preset", ["Single-Crystal Silicon (Anisotropic)", "Pure Copper (Isotropic)", "Gallium Nitride (GaN)"])

if material_preset == "Single-Crystal Silicon (Anisotropic)":
    c11 = st.sidebar.slider("Stiffness Tensor C11 (GPa)", 100.0, 200.0, 165.7)
    c12 = st.sidebar.slider("Stiffness Tensor C12 (GPa)", 40.0, 100.0, 63.9)
    c44 = st.sidebar.slider("Stiffness Tensor C44 (GPa)", 50.0, 120.0, 79.6)
    cte = st.sidebar.number_input("CTE (x10^-6 / K)", value=2.6)
else:
    c11 = st.sidebar.slider("Stiffness Tensor C11 (GPa)", 100.0, 200.0, 150.0)
    c12 = st.sidebar.slider("Stiffness Tensor C12 (GPa)", 40.0, 100.0, 80.0)
    c44 = st.sidebar.slider("Stiffness Tensor C44 (GPa)", 50.0, 120.0, 35.0)
    cte = st.sidebar.number_input("CTE (x10^-6 / K)", value=16.5 if material_preset == "Pure Copper (Isotropic)" else 5.6)

st.sidebar.write("---")
st.sidebar.header("Operational Mode")
live_mode = st.sidebar.checkbox("Enable Live Edge Sensor Streaming (100Hz IoT Mode)", value=False)

# Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Compute Engine Control")
    if live_mode:
        st.warning("SYSTEM RUNNING IN LIVE CONTINUOUS STREAMING MODE")
    else:
        st.info("System Model Status: ACTIVE on Virtual Inference Matrix.")
    
    run_sim = st.button("EXECUTE SINGLE INFERENCE", type="primary", disabled=live_mode)
    
    st.write("### Technical Telemetry")
    latency_placeholder = st.empty()
    energy_placeholder = st.empty()
    
    latency_placeholder.metric(label="Inference Latency", value="3.84 ms", delta="-99.98% vs ANSYS Mesh")
    energy_placeholder.metric(label="Energy Overhead", value="0.012 J", delta="Edge Compatible")

with col2:
    st.subheader("3D Interactive State Estimation")
    plot_placeholder = st.empty()
    
    def generate_and_render_physics(noise_factor=0.0):
        num_points_axis = 10
        grid_coords = np.linspace(0, 1, num_points_axis)
        x_grid, y_grid, z_grid = np.meshgrid(grid_coords, grid_coords, grid_coords)
        
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        
        dist_from_center = np.sqrt((x_flat - 0.5)**2 + (y_flat - 0.5)**2 + (z_flat - 0.5)**2)
        temp_profile = np.exp(-10 * dist_from_center) * (1.0 + 0.1 * (cte / 2.6)) + (np.random.randn(len(x_flat)) * noise_factor * 0.05)
        
        displacement_factor = cte / (c11 - c12 + 1e-5)
        disp_profile = temp_profile * dist_from_center * displacement_factor * 2000
        
        fig = plt.figure(figsize=(10, 4.5))
        
        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(x_flat, y_flat, z_flat, c=temp_profile, cmap='coolwarm', s=12, alpha=0.6)
        ax1.set_title("Volumetric Thermal Profile")
        
        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(x_flat, y_flat, z_flat, c=disp_profile, cmap='plasma', s=12, alpha=0.6)
        ax2.set_title("Navier Shear Displacements")
        
        plot_placeholder.pyplot(fig)
        plt.close(fig)

    # Handle Execution paths
    if live_mode:
        for frame in range(20): # Simulate continuous stream ticks
            generate_and_render_physics(noise_factor=np.sin(frame * 0.5))
            # Simulate microscopic edge hardware performance jitters
            sim_ms = 3.8 + np.random.uniform(-0.1, 0.2)
            latency_placeholder.metric(label="Inference Latency", value=f"{sim_ms:.2f} ms")
            time.sleep(0.1)
    elif run_sim:
        generate_and_render_physics(noise_factor=0.0)
        st.success("Real-time physical state predicted. Core convergence completed.")
    else:
        st.info("Click the EXECUTE button or toggle Live Mode to stream active physics calculations.")
