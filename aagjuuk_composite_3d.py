import deepxde as dde
import numpy as np
import tensorflow as tf

# 1. Spatial and Temporal Domains
geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define Composite Multi-Material Multi-Physics PDE System
def composite_pde(x, y):
    # Inputs: x[:, 0]=x, x[:, 1]=y, x[:, 2]=z, x[:, 3]=t
    # Outputs: y[:, 0]=T (Temp), y[:, 1]=u, y[:, 2]=v, y[:, 3]=w (3D Displacements)
    T, u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    
    # Extract coordinates
    X, Y, Z, t = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    
    # --- Dynamic Material Coefficient Mapping ---
    # We define a sharp transition boundary at the interface z = 0.5
    # Below z = 0.5: Anisotropic Silicon
    # Above z = 0.5: Isotropic Copper
    is_copper = tf.greater_equal(Z, 0.5)
    
    # Elastic Constants mapping
    C11 = tf.where(is_copper, 150.0, 165.7)  # GPa
    C12 = tf.where(is_copper, 80.0, 63.9)    # GPa
    C44 = tf.where(is_copper, 35.0, 79.6)    # GPa
    
    # Thermal Expansion Constants mapping
    alpha_thermal = tf.where(is_copper, 16.5e-6, 2.6e-6)
    
    # Thermal Diffusivity mapping (Heat conducts faster in Copper)
    alpha_diffusivity = tf.where(is_copper, 0.11, 0.05)
    
    # --- 1. Composite Thermal Dissipation Component ---
    dT_t = dde.grad.jacobian(y, x, i=0, j=3)
    dT_xx = dde.grad.hessian(y, x, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, i=0, j=1)
    dT_zz = dde.grad.hessian(y, x, i=0, j=2)
    thermal_residual = dT_t - alpha_diffusivity * (dT_xx + dT_yy + dT_zz)
    
    # --- 2. Structural Strain Calculations ---
    du_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=2, j=1)
    dw_z = dde.grad.jacobian(y, x, i=3, j=2)
    
    du_y = dde.grad.jacobian(y, x, i=1, j=1)
    du_z = dde.grad.jacobian(y, x, i=1, j=2)
    dv_x = dde.grad.jacobian(y, x, i=2, j=0)
    dv_z = dde.grad.jacobian(y, x, i=2, j=2)
    dw_x = dde.grad.jacobian(y, x, i=3, j=0)
    dw_y = dde.grad.jacobian(y, x, i=3, j=1)
    
    gamma_yz = dv_z + dw_y
    gamma_xz = du_z + dw_x
    gamma_xy = du_y + dv_x
    
    # --- Hooke's Law with spatially variable coefficients ---
    sigma_xx = C11 * du_x + C12 * (dv_y + dw_z) - (C11 + 2*C12) * alpha_thermal * T
    sigma_yy = C11 * dv_y + C12 * (du_x + dw_z) - (C11 + 2*C12) * alpha_thermal * T
    sigma_zz = C11 * dw_z + C12 * (du_x + dv_y) - (C11 + 2*C12) * alpha_thermal * T
    
    sigma_yz = C44 * gamma_yz
    sigma_xz = C44 * gamma_xz
    sigma_xy = C44 * gamma_xy
    
    # --- Navier Equilibrium Residual Forces ---
    stress_x_residual = (
        dde.grad.jacobian(sigma_xx, x, i=0, j=0) +
        dde.grad.jacobian(sigma_xy, x, i=0, j=1) +
        dde.grad.jacobian(sigma_xz, x, i=0, j=2)
    )
    
    stress_y_residual = (
        dde.grad.jacobian(sigma_xy, x, i=0, j=0) +
        dde.grad.jacobian(sigma_yy, x, i=0, j=1) +
        dde.grad.jacobian(sigma_yz, x, i=0, j=2)
    )
    
    stress_z_residual = (
        dde.grad.jacobian(sigma_xz, x, i=0, j=0) +
        dde.grad.jacobian(sigma_yz, x, i=0, j=1) +
        dde.grad.jacobian(sigma_zz, x, i=0, j=2)
    )
    
    return [thermal_residual, stress_x_residual, stress_y_residual, stress_z_residual]

# 3. Boundary & Initial Setup
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

def init_thermal(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]) * np.sin(np.pi * x[:, 2:3])

ic_thermal = dde.icbc.IC(geomtime, init_thermal, lambda _, on_initial: on_initial, component=0)

data = dde.data.TimePDE(
    geomtime, 
    composite_pde, 
    [bc, ic_thermal], 
    num_domain=8000,   # Increased evaluation points to resolve the sharp interface boundaries
    num_boundary=600, 
    num_initial=600
)

# 4. Neural Network Scaling
net = dde.nn.FNN([4] + [128] * 5 + [4], "tanh", "Glorot normal")

# 5. Compile and Train
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=6000)

model.save("aagjuuk_composite_model")
print("Composite Multi-Material Physics Solver successfully executed.")
