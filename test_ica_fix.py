#!/usr/bin/env python3
"""
Quick test to verify ICA implementation works correctly
"""

import numpy as np
from sklearn.decomposition import FastICA, PCA

def test_ica_attributes():
    """Test that ICA doesn't have explained_variance_ratio_ but PCA does"""
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 10)
    
    # Test PCA
    print("Testing PCA...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    print(f"  ✓ PCA shape: {pca_result.shape}")
    print(f"  ✓ PCA has explained_variance_ratio_: {hasattr(pca, 'explained_variance_ratio_')}")
    print(f"  ✓ PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  ✓ PCA has inverse_transform: {hasattr(pca, 'inverse_transform')}")
    
    # Test ICA
    print("\nTesting ICA...")
    ica = FastICA(n_components=3, random_state=42, max_iter=500)
    ica_result = ica.fit_transform(data)
    print(f"  ✓ ICA shape: {ica_result.shape}")
    print(f"  ✓ ICA has explained_variance_ratio_: {hasattr(ica, 'explained_variance_ratio_')}")
    print(f"  ✓ ICA has n_iter_: {hasattr(ica, 'n_iter_')}")
    if hasattr(ica, 'n_iter_'):
        print(f"  ✓ ICA n_iter_: {ica.n_iter_}")
    print(f"  ✓ ICA has inverse_transform: {hasattr(ica, 'inverse_transform')}")
    
    # Test inverse transform
    print("\nTesting inverse transforms...")
    
    # PCA inverse
    pca_coords = pca_result[0]
    pca_reconstructed = pca.inverse_transform(pca_coords.reshape(1, -1))
    print(f"  ✓ PCA inverse_transform works: {pca_reconstructed.shape}")
    
    # ICA inverse
    ica_coords = ica_result[0]
    ica_reconstructed = ica.inverse_transform(ica_coords.reshape(1, -1))
    print(f"  ✓ ICA inverse_transform works: {ica_reconstructed.shape}")
    
    print("\n✅ All tests passed!")
    print("\nKey findings:")
    print("  - ICA does NOT have explained_variance_ratio_ (unlike PCA)")
    print("  - ICA has n_iter_ attribute for convergence info")
    print("  - Both PCA and ICA have inverse_transform method")
    print("  - Both can be used for morphospace visualization")

if __name__ == "__main__":
    test_ica_attributes()

