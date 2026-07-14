import numpy as np

class LatentDataAssimilator:
    """
    Microsecond-scale latent state corrector utilizing a localized
    differentiable Kalman projection to merge noisy telemetry with physics predictions.
    """
    def __init__(self, state_dim: int, observation_dim: int):
        self.I = np.eye(state_dim)
        # Process noise covariance (trust in our physical model)
        self.Q = np.eye(state_dim) * 1e-5 
        # Measurement noise covariance (trust in physical hardware sensors)
        self.R = np.eye(observation_dim) * 1e-2 

    def assimilate(
        self, 
        predicted_state: np.ndarray, 
        covariance: np.ndarray, 
        measurement: np.ndarray, 
        observation_operator: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes a localized, matrix-free Kalman correction step.
        """
        # Project state covariance into observation space
        H = observation_operator
        HT = H.T
        
        # S = H * P * H_T + R
        S = H @ covariance @ HT + self.R
        
        # Compute exact Kalman gain: K = P * H_T * S^-1
        K = covariance @ HT @ np.linalg.inv(S)
        
        # Correct the predicted state based on physical observation anomaly
        innovation = measurement - (H @ predicted_state)
        corrected_state = predicted_state + K @ innovation
        
        # Update state error covariance: P = (I - K * H) * P
        corrected_covariance = (self.I - K @ H) @ covariance + self.Q
        
        return corrected_state, corrected_covariance
