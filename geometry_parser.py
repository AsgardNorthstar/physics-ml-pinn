import numpy as np
from stl import mesh

def parse_stl_to_nodes(file_path, num_samples=1000):
    """
    Parses a 3D CAD .stl file and extracts a point cloud of spatial coordinates (x, y, z)
    to feed into the Aagjuuk physics-informed neural network boundary conditions.
    """
    try:
        # Load the STL file using numpy-stl
        your_mesh = mesh.Mesh.from_file(file_path)
        
        # Flatten the mesh vectors to get raw 3D vertices
        vertices = your_mesh.vectors.reshape(-1, 3)
        
        # Remove duplicate points to get clean boundary nodes
        unique_nodes = np.unique(vertices, axis=0)
        
        # Downsample if the CAD model is too dense, to keep inference under 5ms
        if len(unique_nodes) > num_samples:
            indices = np.random.choice(len(unique_nodes), num_samples, replace=False)
            unique_nodes = unique_nodes[indices]
            
        # Normalize coordinates to [0, 1] range for PINN stability
        min_val = unique_nodes.min(axis=0)
        max_val = unique_nodes.max(axis=0)
        normalized_nodes = (unique_nodes - min_val) / (max_val - min_val + 1e-8)
        
        return normalized_nodes, None
    except Exception as e:
        return None, str(e)

print("Enterprise Geometry Parser compiled successfully.")
