import os
import json
import glob
import numpy as np

# Configuration
LMs_DIR = "/Users/leyangloh/Downloads/3D_Fish/mussels_31/LMs"

def load_landmarks(lm_dir):
    files = glob.glob(os.path.join(lm_dir, "*.mrk.json"))
    landmarks = {}
    
    for f in sorted(files):
        basename = os.path.basename(f)
        model_id = basename.replace(".mrk.json", "")
        
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                
            if 'markups' in data and len(data['markups']) > 0:
                control_points = data['markups'][0]['controlPoints']
                points = []
                for cp in control_points:
                    pos = cp['position']
                    pid = int(cp['id'])
                    points.append((pid, pos))
                
                points.sort(key=lambda x: x[0])
                coords = np.array([p[1] for p in points])
                landmarks[model_id] = coords
            else:
                print(f"Warning: No markups found in {basename}")
                
        except Exception as e:
            print(f"Error loading {basename}: {e}")
            
    return landmarks

def procrustes_numpy(ref, target):
    """
    Procrustes analysis using numpy (translation, scaling, rotation).
    Returns transformed target and disparity.
    MATCHES scipy.spatial.procrustes behavior.
    """
    # 1. Translation
    ref_mean = np.mean(ref, axis=0)
    target_mean = np.mean(target, axis=0)
    ref_centered = ref - ref_mean
    target_centered = target - target_mean
    
    # 2. Scaling
    ref_scale = np.linalg.norm(ref_centered)
    target_scale = np.linalg.norm(target_centered)
    
    ref_norm = ref_centered / ref_scale
    target_norm = target_centered / target_scale
    
    # 3. Rotation (Orthogonal Procrustes)
    U, s, Vt = np.linalg.svd(np.dot(target_norm.T, ref_norm))
    R = np.dot(U, Vt)
    
    target_transformed = np.dot(target_norm, R)
    
    # Return matched target to ref's scale? 
    # Scipy returns both standardized.
    # We want to return target aligned to ref.
    # If we want simple geometric alignment without size normalization, we skip scale div.
    # But GPA usually involves size normalization.
    
    return target_transformed, ref_norm

def generalized_procrustes_analysis(shapes, max_iter=10, tol=1e-5):
    n_shapes = len(shapes)
    if n_shapes == 0:
        return [], None
    
    # Flatten to list of arrays
    shapes_arr = [s.copy() for s in shapes]
    
    # Initial alignment: center and scale all
    for i in range(n_shapes):
        center = np.mean(shapes_arr[i], axis=0)
        shapes_arr[i] -= center
        scale = np.linalg.norm(shapes_arr[i])
        shapes_arr[i] /= scale
        
    mean_shape = shapes_arr[0].copy()
    
    for iteration in range(max_iter):
        aligned_shapes = []
        for s in shapes_arr:
            # Align s to mean_shape
            # Orthogonal Procrustes
            U, _, Vt = np.linalg.svd(np.dot(s.T, mean_shape))
            R = np.dot(U, Vt)
            aligned_s = np.dot(s, R)
            aligned_shapes.append(aligned_s)
            
        shapes_arr = aligned_shapes
        new_mean_shape = np.mean(np.array(shapes_arr), axis=0)
        
        # Normalize mean size
        new_mean_shape /= np.linalg.norm(new_mean_shape)
        
        # Align new mean to old mean to prevent drift
        U, _, Vt = np.linalg.svd(np.dot(new_mean_shape.T, mean_shape))
        R = np.dot(U, Vt)
        new_mean_shape = np.dot(new_mean_shape, R)
        
        diff = np.linalg.norm(new_mean_shape - mean_shape)
        mean_shape = new_mean_shape
        
        if diff < tol:
            break
            
    return shapes_arr, mean_shape

def pca_numpy(data, n_modes=10):
    # data: (N_samples, M_features)
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Covariance matrix? Or SVD on data directly?
    # SVD on centered data: X = U S V^T
    # PCA scores = U * S
    # Components = V^T
    
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    # Eigenvalues = S^2 / (N-1)
    eigenvalues = (S ** 2) / (data.shape[0] - 1)
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Scores (Coordinates in PC space)
    scores = np.dot(centered_data, Vt.T)
    # Alternatively: scores = U * S
    
    return scores, explained_variance_ratio, Vt

def main():
    print(f"Loading landmarks from {LMs_DIR}...")
    lms_dict = load_landmarks(LMs_DIR)
    ids = list(lms_dict.keys())
    raw_shapes = list(lms_dict.values())
    
    if not raw_shapes:
        print("No landmarks loaded.")
        return

    n_points = raw_shapes[0].shape[0]
    print(f"Loaded {len(ids)} specimens with {n_points} landmarks each.")
    
    print("Running Generalized Procrustes Analysis...")
    aligned_shapes, mean_shape = generalized_procrustes_analysis(raw_shapes)
    
    # Flatten shapes for PCA: (n_samples, n_points * 3)
    data_matrix = np.array([s.flatten() for s in aligned_shapes])
    
    print("Running PCA...")
    scores, explained_var, components = pca_numpy(data_matrix)
    
    print(f"PCA output shape: {scores.shape}")
    print("Explained variance ratio (first 5 components):")
    for i, var in enumerate(explained_var[:5]):
        print(f"PC{i+1}: {var:.4f}")
    
    # Find closest pairs using Euclidean distance on first 3 PCs
    pc_data = scores[:, :3]
    n_samples = len(ids)
    
    pairs = []
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(pc_data[i] - pc_data[j])
            pairs.append((ids[i], ids[j], dist))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    pairs.sort(key=lambda x: x[2])
    
    print("\n--- Closest Model Pairs (Most Similar) ---")
    print("(Based on Euclidean distance in PC1-PC3 space)")
    for p in pairs[:10]:
        print(f"{p[0]} <-> {p[1]} : dist={p[2]:.4f}")
        
    # Simple Clustering (Single Linkage like)
    # Find groups where members are connected by distance < threshold
    # Let's try a few thresholds to find a group of size 3-6
    
    print("\n--- Identified Groups (Clusters) ---")
    
    sorted_dists = [p[2] for p in pairs]
    # Pick a threshold? e.g. 10th percentile or fixed
    # Visually from previous, 0.02 seems like a good tight threshold
    
    thresholds = [0.030, 0.032, 0.034, 0.036, 0.038, 0.040]
    
    for thresh in thresholds:
        adj = dist_matrix < thresh
        np.fill_diagonal(adj, False)
        
        # Find connected components
        visited = set()
        groups = []
        
        for i in range(n_samples):
            if i not in visited:
                # BFS/DFS
                stack = [i]
                group = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    group.append(node)
                    
                    neighbors = np.where(adj[node])[0]
                    for nb in neighbors:
                        if nb not in visited:
                            stack.append(nb)
                if len(group) > 2: # interconnected group
                    groups.append(group)
        
        if groups:
            print(f"\nThreshold {thresh}: Found {len(groups)} groups (>2 members)")
            for g in groups:
                member_ids = [ids[idx] for idx in g]
                print(f"  Group ({len(g)} members): {', '.join(sorted(member_ids))}")

    # Diversity Selection: Farthest Point Sampling (FPS)
    print("\n--- Selecting 10 Diverse Mussels (Farthest Point Sampling) ---")
    n_select = 10
    if n_samples < n_select:
        print(f"Not enough samples ({n_samples}) to select {n_select}.")
    else:
        # 1. Start with the point closest to the mean (closest to 0 in PC space)
        # Or start with a random point, or the most extreme point.
        # Closest to mean is a good "representative" start, but for diversity maybe most extreme.
        # Let's start with the one most distant from the centroid (the outlier) to ensure range.
        dist_from_center = np.linalg.norm(pc_data, axis=1)
        start_idx = np.argmax(dist_from_center)
        
        selected_indices = [start_idx]
        min_dists = np.full(n_samples, np.inf)
        
        # Calculate dist from the first selected point
        dists = np.linalg.norm(pc_data - pc_data[start_idx], axis=1)
        min_dists = np.minimum(min_dists, dists)
        
        for _ in range(n_select - 1):
            # Select the point with the maximum distance to the set of selected points
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)
            
            # Update minimum distances
            dists = np.linalg.norm(pc_data - pc_data[next_idx], axis=1)
            min_dists = np.minimum(min_dists, dists)
            
        diverse_ids = [ids[idx] for idx in selected_indices]
        print(f"Selected {n_select} diverse mussels:")
        for i, mid in enumerate(diverse_ids):
            print(f"  {i+1}. {mid}")

if __name__ == "__main__":
    main()
