import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#parametri
mu = [0, 0] 
sigma = [[1, 0.5], 
         [0.5, 1]]

#generazione campione
np.random.seed(50)
n_samples = 1000
data = np.random.multivariate_normal(mu, sigma, n_samples)

#PCA
pca = PCA(n_components=2)
data_transformed = pca.fit_transform(data)


fig, axes = plt.subplots(1, 2, figsize=(6, 3))

#originali
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.7, c='blue', edgecolor='k')
axes[0].set_title("Dati Originali")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].grid(True)

#trasformati
axes[1].scatter(data_transformed[:, 0], data_transformed[:, 1], alpha=0.7, c='green', edgecolor='k')
axes[1].set_title("Dati Trasformati (PCA)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].grid(True)

plt.tight_layout()
plt.show()