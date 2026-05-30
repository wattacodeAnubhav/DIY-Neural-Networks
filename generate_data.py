import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_blobs

def save_dataset(X, y, filename):
    """Formats matrices and saves them as CSV files."""
    os.makedirs('sample_data', exist_ok=True)
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['label'] = y.astype(int)
    
    filepath = f'sample_data/{filename}.csv'
    df.to_csv(filepath, index=False)
    print(f"✅ Generated {filepath}")

def generate_3d_plane(n_samples):
    """Creates two distinct clusters separated by a clear 3D gap."""
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=3, random_state=42, cluster_std=1.2)
    save_dataset(X, y, '3d_plane')

def generate_3d_spherical(n_samples):
    """Creates a dense inner ball trapped inside a hollow outer shell."""
    N = n_samples // 2
    
    # Class 0: Inner Sphere (Dense ball)
    inner_sphere = np.random.normal(0, 0.5, (N, 3))
    
    # Class 1: Outer Sphere (Hollow shell)
    u = np.random.normal(0, 1, (N, 3))
    d = np.linalg.norm(u, axis=1, keepdims=True)
    outer_sphere = (u / d) * 3.0 + np.random.normal(0, 0.2, (N, 3))
    
    X = np.vstack((inner_sphere, outer_sphere))
    y = np.hstack((np.zeros(N), np.ones(N)))
    save_dataset(X, y, '3d_spherical')

def generate_3d_helix(n_samples):
    """Creates two intertwined spirals extending along the Z-axis."""
    N = n_samples // 2
    t = np.linspace(0, 4 * np.pi, N) # Two full rotations
    
    # Class 0: Strand A
    x_a = np.cos(t) + np.random.randn(N) * 0.1
    y_a = np.sin(t) + np.random.randn(N) * 0.1
    z_a = t / (4 * np.pi) * 5 # Stretch along Z-axis
    strand_a = np.column_stack((x_a, y_a, z_a))
    
    # Class 1: Strand B (Offset by Pi so they intertwine)
    x_b = np.cos(t + np.pi) + np.random.randn(N) * 0.1
    y_b = np.sin(t + np.pi) + np.random.randn(N) * 0.1
    z_b = t / (4 * np.pi) * 5
    strand_b = np.column_stack((x_b, y_b, z_b))
    
    X = np.vstack((strand_a, strand_b))
    y = np.hstack((np.zeros(N), np.ones(N)))
    save_dataset(X, y, '3d_helix')

if __name__ == "__main__":
    print("Generating 3D Neural Network Sandbox Datasets...\n")
    np.random.seed(42)
    
    # Generate 600 total points for each dataset (300 per class)
    total_samples = 600 
    
    generate_3d_plane(total_samples)
    generate_3d_spherical(total_samples)
    generate_3d_helix(total_samples)
    
    print("\n🎉 All 3D datasets generated successfully!")

