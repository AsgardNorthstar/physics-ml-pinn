import numpy as np

class MultiScaleFourierEncoder:
    """
    High-Frequency Spatial Position Encoder.
    Overcomes Neural Spectral Bias to resolve sharp micro-cracks and 
    non-linear boundaries at silicon-copper material interfaces.
    """
    def __init__(self, input_dim=3, mapping_size=32, scale=10.0):
        self.mapping_size = mapping_size
        self.scale = scale
        # Generate a fixed projection matrix using Gaussian distribution
        # This maps (x, y, z) into a high-dimensional sinusoidal space
        self.B = np.random.randn(input_dim, mapping_size) * scale

    def encode(self, coords):
        """
        Maps continuous spatial coordinate tensors of shape (N, 3) 
        into Fourier feature space of shape (N, 2 * mapping_size).
        """
        # Project inputs: X_proj = 2 * pi * X * B
        projected = np.dot(coords, self.B) * 2.0 * np.pi
        
        # Apply sinusoidal encoding components
        cos_features = np.cos(projected)
        sin_features = np.sin(projected)
        
        # Concatenate features to form the final embedding matrix
        return np.hstack((cos_features, sin_features))

print("Aagjuuk High-Frequency Multi-Scale Fourier Encoder loaded successfully.")
