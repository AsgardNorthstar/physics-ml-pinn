import numpy as np
import pytest
from aagjuuk_fourier_encoder import MultiScaleFourierEncoder
from aagjuuk_control_loop import ThermalStressController
from aagjuuk_inverse_solver import locate_internal_defect

def test_fourier_encoder_dimensions():
    """Validates coordinate projection matrix shapes and type consistency."""
    encoder = MultiScaleFourierEncoder(input_dim=3, mapping_size=16, scale=1.0)
    mock_coords = np.random.rand(10, 3)
    
    encoded = encoder.encode(mock_coords)
    
    # Expected output size is 2 * mapping_size (cos + sin components)
    assert encoded.shape == (10, 32)
    assert isinstance(encoded, np.ndarray)

def test_fourier_encoder_invalid_input():
    """Verifies that the encoder handles dimensional mismatch gracefully."""
    encoder = MultiScaleFourierEncoder(input_dim=3)
    invalid_coords = np.random.rand(10, 2)  # Missing Z-coordinate
    
    with pytest.raises(ValueError):
        encoder.encode(invalid_coords)

def test_control_loop_nan_protection():
    """Asserts that the hardware controller triggers a safety fallback when receiving NaN."""
    controller = ThermalStressController(target_stress_limit=25.0)
    
    # Simulate a corrupted sensor reading yielding NaN
    safe_power, error = controller.compute_control_action(current_stress_max=np.nan, current_laser_power=50.0)
    
    assert safe_power < 50.0
    assert safe_power == 25.0 or safe_power == 15.0  # Assures safety backup power reduction
    assert error == 0.0

def test_inverse_solver_accuracy():
    """Confirms the inverse engine resolves coordinates within the spatial domain."""
    sensor_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    temperatures = np.array([300.0, 300.0, 350.0])  # Heat anomaly near top sensor
    
    predicted_xyz, confidence = locate_internal_defect(temperatures, sensor_coords, stiffness_tensor=165.7)
    
    assert predicted_xyz is not None
    assert predicted_xyz.shape == (3,)
    # Verify the target coordinate falls inside our [0, 1] unit cell boundary
    assert np.all(predicted_xyz >= 0.0) and np.all(predicted_xyz <= 1.0)
