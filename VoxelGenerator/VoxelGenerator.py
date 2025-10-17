import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stl_to_voxel(filename, resolution=10):
    """
    Convert an STL file to a 3D voxel grid
    
    Args:
        filename: Path to the STL file
        resolution: Number of voxels along each axis (higher = more detail, slower)
    
    Returns:
        voxel_grid: 3D numpy array of boolean values
        bounds: Tuple of (min_coords, max_coords)
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(filename)
    
    # Get the bounding box from all vertices
    # stl_mesh.vectors has shape (n_triangles, 3, 3) - flatten to get all points
    all_points = stl_mesh.vectors.reshape(-1, 3)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    
    # Create voxel grid
    dims = max_coords - min_coords
    voxel_size = dims / resolution
    
    # Initialize empty voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # For each triangle in the mesh
    for triangle in stl_mesh.vectors:
        # Get the bounding box of the triangle
        tri_min = triangle.min(axis=0)
        tri_max = triangle.max(axis=0)
        
        # Convert to voxel coordinates
        voxel_min = np.floor((tri_min - min_coords) / voxel_size).astype(int)
        voxel_max = np.ceil((tri_max - min_coords) / voxel_size).astype(int)
        
        # Clamp to grid bounds
        voxel_min = np.maximum(voxel_min, 0)
        voxel_max = np.minimum(voxel_max, resolution - 1)
        
        # Fill voxels that intersect with the triangle
        for i in range(voxel_min[0], voxel_max[0] + 1):
            for j in range(voxel_min[1], voxel_max[1] + 1):
                for k in range(voxel_min[2], voxel_max[2] + 1):
                    # Get voxel center in world coordinates
                    voxel_center = min_coords + np.array([i, j, k]) * voxel_size + voxel_size / 2
                    
                    # Simple point-in-triangle test (can be improved)
                    # For now, mark voxel as occupied if its center is near the triangle
                    if point_near_triangle(voxel_center, triangle, voxel_size.max()):
                        voxel_grid[i, j, k] = True
    
    return voxel_grid, (min_coords, max_coords)

def point_near_triangle(point, triangle, threshold):
    """
    Simple check if a point is near a triangle
    """
    # Calculate distances to each vertex
    distances = np.linalg.norm(triangle - point, axis=1)
    return distances.min() < threshold * 2

def display_voxels(voxel_grid):
    """
    Display the 3D voxel grid
    
    Args:
        voxel_grid: 3D numpy array of boolean values
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of filled voxels
    filled = np.where(voxel_grid)
    
    # Plot voxels
    ax.voxels(voxel_grid, facecolors='cyan', edgecolors='k', alpha=0.7)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Voxel Grid ({voxel_grid.shape[0]}x{voxel_grid.shape[1]}x{voxel_grid.shape[2]})')
    
    # Print statistics
    total_voxels = voxel_grid.size
    filled_voxels = voxel_grid.sum()
    print(f"Total voxels: {total_voxels}")
    print(f"Filled voxels: {filled_voxels}")
    print(f"Fill percentage: {100 * filled_voxels / total_voxels:.2f}%")
    
    plt.show()

# Main execution
if __name__ == "__main__":
    stl_file = "000154_vertebrae.stl"
    
    # Adjust resolution as needed (higher = more detail but slower)
    # Start with 30-50 for faster processing
    resolution = 40
    
    try:
        print(f"Loading STL file: {stl_file}")
        print(f"Converting to voxel grid (resolution: {resolution})...")
        
        voxel_grid, bounds = stl_to_voxel(stl_file, resolution=resolution)
        
        print("Displaying voxels...")
        display_voxels(voxel_grid)
        
    except FileNotFoundError:
        print(f"Error: File '{stl_file}' not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
