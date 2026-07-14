import deepxde as dde
import numpy as np
import torch

# 1. Define Spatial (3D Cuboid) and Temporal Domains
# We model a solid cube [0, 1] x [0, 1] x [0, 1] over a time interval [0, 1.0]
space_domain = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
time_domain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

# 2. Define the 3D Physical Law (The PDE)
def pde(x, y):
    # Inputs:
    # x[:, 0:1] = x-coordinate
    # x[:, 1:2] = y-coordinate
    # x[:, 2:3] = z-coordinate
    # x[:, 3:4] = time (t)
    # Output:
    # y = predicted temperature (T)
    
    # Calculate first derivative with respect to time (dT/dt)
    dT_t = dde.grad.jacobian(y, x, i=0, j=3)
    
    # Calculate second spatial derivatives (3D Laplacian components)
    dT_xx = dde.grad.hessian(y, x, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, i=1, j=1)
    dT_zz = dde.grad.hessian(y, x, i=2, j=2)
    
    # Thermal diffusivity constant (alpha) 
    alpha = 0.05
    
    # 3D Heat Equation: dT/dt - alpha * (dT/dx^2 + dT/dy^2 + dT/dz^2) = 0
    return dT_t - alpha * (dT_xx + dT_yy + dT_zz)

# 3. Boundary Conditions (BC) & Initial Conditions (IC)
# Boundary: Keep all six faces of the 3D cube cooled at 0 degrees
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# Initial: A localized heat spike concentrated in the exact center of the 3D cube
def initial_condition(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]) * np.sin(np.pi * x[:, 2:3])

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

# 4. Generate Training Constraints
# We use more domain points to effectively map the extra spatial dimension
data = dde.data.TimePDE(
    geomtime, 
    pde, 
    [bc, ic], 
    num_domain=8000,    # Physics evaluation points across the 3D volume over time
    num_boundary=600,   # Spatial face constraints
    num_initial=600     # Initial state constraints
)

# 5. Multilayer Perceptron Architecture
# Input layer takes 4 dimensions (x, y, z, t). Output is 1 dimension (Temperature).
net = dde.nn.FNN([4] + [64] * 4 + [1], "tanh", "Glorot normal")

# 6. Compile and Train with Adam Optimizer
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

# Save the trained model parameters
model.save("pinn_thermal_3d_model")
print("3D PINN Thermal Solver successfully trained and saved.")
