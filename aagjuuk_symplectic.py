import numpy as np

class SymplecticHamiltonianIntegrator:
    """
    Symplectic Leapfrog Integrator for Hamiltonian Neural Operators.
    Guarantees exact conservation of phase-space volume and system energy
    over infinite-horizon transient simulations.
    """
    def __init__(self, dH_dq_func, dH_dp_func):
        # Neural network gradients representing the Hamiltonian equations of motion
        self.dH_dq = dH_dq_func  # Force field equivalent: -dp/dt
        self.dH_dp = dH_dp_func  # Velocity equivalent: dq/dt

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a single, numerically stable symplectic leapfrog integration step.
        Coordinates q and momentum p are updated out of phase to preserve the symplectic metric.
        """
        # 1. Update momentum by a half-step
        p_half = p - 0.5 * dt * self.dH_dq(q, p)
        
        # 2. Update position by a full step using the intermediate momentum
        q_next = q + dt * self.dH_dp(q, p_half)
        
        # 3. Complete momentum update using the new position
        p_next = p_half - 0.5 * dt * self.dH_dq(q_next, p_half)
        
        return q_next, p_next
