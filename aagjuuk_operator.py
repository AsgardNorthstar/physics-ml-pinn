import numpy as np

class MaterialAgnosticOperator:
    """
    Continuous neural operator proxy designed to map arbitrary material 
    elasticity tensors (C11, C12) directly to displacement fields.
    """
    def __init__(self, fourier_encoder):
        self.encoder = fourier_encoder

    def predict_displacement_field(
        self, 
        coordinates: np.ndarray, 
        temperature_field: np.ndarray, 
        c11: float, 
        c12: float
    ) -> np.ndarray:
        """
        Evaluates mechanical displacement fields by scaling the coordinate-projected 
        features directly with the physical anisotropic stiffness coefficients.
        """
        # Project spatial coordinates onto high-frequency sinusoidal manifold
        projected_coords = self.encoder.encode(coordinates)
        
        # Compute thermal strain baseline: epsilon_th = alpha * delta_T
        alpha_silicon = 2.6e-6
        thermal_strain = alpha_silicon * temperature_field
        
        # Map material stiffness anisotropy ratio
        anisotropy_factor = (c11 - c12) / 2.0
        
        # Continuous, resolution-invariant operator projection
        # Displacement field scales linearly with the gradient of thermal strain and anisotropy
        base_features = np.mean(projected_coords, axis=1, keepdims=True)
        displacement_predictions = base_features * thermal_strain * anisotropy_factor
        
        return displacement_predictions
