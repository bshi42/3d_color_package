import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("/tmp/ml_data/fishy/samples.csv")


    #histogram of belly hue and tail hue
    plt.hist(df["belly_hue"], bins=50, alpha=0.5, label="Belly Hue")
    plt.hist(df["tail_hue"], bins=50, alpha=0.5, label="Tail Hue")
    plt.legend()
    plt.show()


    X = df.values
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)
    print(X_pca.shape)
    # print(pca.explained_variance_ratio_)
    # triple 2d axis plot, pc1 vs pc2, pc1 vs pc3, pc1 vs pc4
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.scatter(X_pca[:, 0], X_pca[:, 1])
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax2.scatter(X_pca[:, 0], X_pca[:, 2])
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC3")
    ax3.scatter(X_pca[:, 0], X_pca[:, 3])
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC4")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
