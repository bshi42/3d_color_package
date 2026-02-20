import os
import glob
import numpy as np
from PIL import Image

# Configuration
TEXTURES_DIR = "/Users/leyangloh/Downloads/3D_Fish/mussels_31/textures"
TARGET_SIZE = (64, 64) # Resize for features

def pca_numpy(data, n_modes=10):
    # data: (N_samples, M_features)
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Use randomized SVD or standard SVD
    # For shape (31, 64*64*3) = (31, 12288), standard SVD on centered_data is fine 
    # but computing Covariance (12k x 12k) is bad. 
    # We want svd(X) where X is N x M.
    # U, S, Vt = svd(X)
    # Scores = U * S
    
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    eigenvalues = (S ** 2) / (data.shape[0] - 1)
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    scores = np.dot(centered_data, Vt.T)
    return scores, explained_variance_ratio, Vt

def load_textures(tex_dir):
    files = glob.glob(os.path.join(tex_dir, "*.png"))
    data = []
    ids = []
    
    print(f"Loading {len(files)} textures...")
    for f in sorted(files):
        basename = os.path.basename(f)
        # ID is usually the filename without extension, but some have extra suffix?
        # Filenames like UF_IZ_438751.png
        model_id = os.path.splitext(basename)[0]
        
        try:
            img = Image.open(f).convert('RGB')
            img = img.resize(TARGET_SIZE)
            # Flatten: (64, 64, 3) -> 12288 features
            arr = np.array(img).flatten()
            data.append(arr)
            ids.append(model_id)
        except Exception as e:
            print(f"Error loading {basename}: {e}")
            
    return np.array(data), ids

def main():
    print(f"Scanning {TEXTURES_DIR}...")
    texture_data, ids = load_textures(TEXTURES_DIR)
    
    if len(ids) == 0:
        print("No textures found.")
        return
        
    n_samples = len(ids)
    print(f"Loaded {n_samples} textures. Feature vector size: {texture_data.shape[1]}")
    
    print("Running PCA on texture features...")
    # Normalize pixel values 0-1 for stability? Not strictly necessary for PCA but good practice.
    texture_data = texture_data / 255.0
    
    scores, explained_var, _ = pca_numpy(texture_data)
    
    print("Explained variance ratio (first 5 components):")
    for i, var in enumerate(explained_var[:5]):
        print(f"PC{i+1}: {var:.4f}")
    print(f"Total explained by 5 components: {sum(explained_var[:5]):.4f}")
    
    # Farthest Point Sampling on PC1-PC5 (captures color & pattern)
    # Using more PCs because texture variance might be more distributed
    n_pcs_fps = 5 
    print(f"\nSelecting 10 Visually Diverse Mussels (based on Texture PC1-PC{n_pcs_fps})...")
    
    pc_data = scores[:, :n_pcs_fps]
    
    n_select = 10
    if n_samples < n_select:
        print("Not enough samples.")
        return
        
    # Start with outlier
    dist_from_center = np.linalg.norm(pc_data, axis=1)
    start_idx = np.argmax(dist_from_center)
    
    selected_indices = [start_idx]
    min_dists = np.full(n_samples, np.inf)
    
    # Calculate dist from the first
    dists = np.linalg.norm(pc_data - pc_data[start_idx], axis=1)
    min_dists = np.minimum(min_dists, dists)
    
    for _ in range(n_select - 1):
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)
        dists = np.linalg.norm(pc_data - pc_data[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists)
        
    print("\n--- Diverse Texture Group ---")
    diverse_ids = [ids[idx] for idx in selected_indices]
    for i, mid in enumerate(diverse_ids):
        print(f"{i+1}. {mid}")

if __name__ == "__main__":
    main()
