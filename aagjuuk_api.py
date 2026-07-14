from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from aagjuuk_fourier_encoder import MultiScaleFourierEncoder
from aagjuuk_control_loop import ThermalStressController

app = FastAPI(
    title="Aagjuuk Labs: Multi-Material Edge Physics API",
    version="1.0.0",
    description="Production-grade endpoints for real-time continuous thermal stress calculations."
)

# Define strict input validation models using Pydantic
class CoordinatePayload(BaseModel):
    coordinates: list[list[float]] = Field(..., description="An (N, 3) matrix of coordinates [x, y, z]")
    scale: float = Field(default=5.0, description="Sinusoidal scale factor for Fourier projections")

class PIDPayload(BaseModel):
    current_stress: float = Field(..., description="Peak observed stress in GPa")
    current_laser_power: float = Field(..., description="Active laser power percentage")

@app.post("/v1/physics/fourier-project")
async def project_coordinates(payload: CoordinatePayload):
    """
    Evaluates multi-scale Fourier feature projections to bypass neural spectral bias.
    """
    coords_array = np.array(payload.coordinates)
    
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise HTTPException(status_code=420, detail="Coordinate input must be of shape (N, 3).")
        
    try:
        encoder = MultiScaleFourierEncoder(input_dim=3, scale=payload.scale)
        projected = encoder.encode(coords_array)
        return {
            "status": "success",
            "dimensions": projected.shape,
            "embeddings": projected.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/v1/control/pid-step")
async def process_control_step(payload: PIDPayload):
    """
    Calculates the non-linear, anti-windup PID control output for laser regulation.
    """
    try:
        controller = ThermalStressController(target_stress_limit=25.0)
        next_power, error = controller.compute_control_action(
            current_stress_max=payload.current_stress,
            current_laser_power=payload.current_laser_power
        )
        return {
            "status": "success",
            "adjusted_laser_power": next_power,
            "error_offset": error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Actuator loop failure: {str(e)}")
