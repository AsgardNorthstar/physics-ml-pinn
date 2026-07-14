import deepxde as dde
import numpy as np
import torch

# 1. Define Spatial (2D Rectangle) and Temporal Domains
# We model a square wafer slice [0, 1] x [0, 1] over a time interval [0, 1]
space_domain = dde.geometry.Rectangle([0, 0], [1, 1])
time_domain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

# 2. Define the 2D Physical Law (The PDE)
def pde(x, y):
    # x[:, 0:1] = x-coordinate
    # x[:, 1:2] = y-coordinate
    # x[:, 2:3] = time (t)
    # y = predicted temperature (T)
    
    # Calculate first derivative with respect to time
    dT_t = dde.grad.jacobian(y, x, i=0, j=2)
    
    # Calculate second spatial derivatives (Laplacian components)
    dT_xx = dde.grad.hessian(y, x, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, i=1, j=1)
    
    # Thermal diffusivity constant (alpha) representing silicon/metal material properties
    alpha = 0.05
    
    # Heat equation: dT/dt - alpha * (dT/dx^2 + dT/dy^2) = 0
    return dT_t - alpha * (dT_xx + dT_yy)

# 3. Boundary Conditions (BC) & Initial Conditions (IC)
# Boundary: Keep the edges of the wafer cooled at 0 degrees (Dirichlet Boundary)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# Initial: A high-heat concentration peak right in the center of the wafer at t=0
def initial_condition(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

# 4. Generate Training Constraints
data = dde.data.TimePDE(
    geomtime, 
    pde, 
    [bc, ic], 
    num_domain=5000,    # Random physics evaluation points across the 2D plane over time
    num_boundary=400,   # Spatial edge constraints
    num_initial=400     # Initial state constraints
)

# 5. Multilayer Perceptron Architecture
# Input layer takes 3 dimensions (x, y, t). Output is 1 dimension (Temperature).
net = dde.nn.FNN([3] + [32] * 4 + [1], "tanh", "Glorot normal")

# 6. Compile and Train with Adam Optimizer
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=8000)

# Save the trained weight states
model.save("pinn_thermal_2d_model")
print("2D PINN Thermal Solver successfully trained and saved.")
