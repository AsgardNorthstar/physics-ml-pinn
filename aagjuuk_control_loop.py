import numpy as np

class ThermalStressController:
    """
    Active Closed-Loop Controller for Industrial Hardware.
    Uses Proportional-Integral-Derivative (PID) mechanics mapped to
    predicted structural stress outputs from the Aagjuuk PINN.
    """
    def __init__(self, target_stress_limit=15.0, kp=0.4, ki=0.1, kd=0.05):
        self.target_stress = target_stress_limit
        self.kp = kp  # Proportional Gain
        self.ki = ki  # Integral Gain
        self.kd = kd  # Derivative Gain
        
        self.integral_error = 0.0
        self.last_error = 0.0

    def compute_control_action(self, current_stress_max, current_laser_power):
        """
        Calculates the required power adjustment for laser annealing or substrate heaters
        to safely bring peak stress back below the critical damage threshold.
        """
        # Error is the delta between maximum estimated stress and our structural safety limit
        error = self.target_stress - current_stress_max
        
        self.integral_error += error
        derivative = error - self.last_error
        self.last_error = error
        
        # PID Output calculation
        adjustment = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        # Apply adjustment to current power setting (clamped to physical hardware safe limits)
        new_power = np.clip(current_laser_power + adjustment, 10.0, 100.0)
        return new_power, error

print("Aagjuuk Active Closed-Loop Controller loaded.")
