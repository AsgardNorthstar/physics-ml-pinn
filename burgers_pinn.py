import deepxde as dde
import numpy as np

# 1. Define the geometry of your material and the time domain
# We will simulate a 1D material over a time interval [0, 1]
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define the Physics (The PDE)
# We embed the physical laws of heat/fluid flow directly into the network's loss function
def pde(x, y):
    # x[:, 0:1] is spatial coordinate, x[:, 1:2] is time
    # y is the predicted temperature/velocity
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    # Burgers' equation representing physical dissipation: 
    # Change in state over time + convection - diffusion = 0
    return dy_t + y * dy_x - (0.01 / np.pi) * dy_xx

# 3. Boundary & Initial Conditions
# We define how the material behaves at its borders (e.g., fixed cold temperature at edges)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 4. Create the Training Dataset
data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)

# 5. Build the Neural Network Architecture
# We use a simple 4-layer feedforward network
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# 6. Compile and Train
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

# Save the model
model.save("pinn_thermal_model")
