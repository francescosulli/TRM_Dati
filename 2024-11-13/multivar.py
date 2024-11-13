import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#parametri
mu = [0, 0]  #media
sigma = [[1, 0.5], [0.5, 1]]  #matrice delle cov

#griglia
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

#distribuzione normale multivariata
rv = multivariate_normal(mu, sigma)
Z = rv.pdf(pos)

#estraz del campione
sample = rv.rvs(size=500)

plt.figure(figsize=(6, 4))

#distribuzione normale multivariata
contour = plt.contourf(X, Y, Z, 50, cmap='twilight_r', alpha=0.5)

#campione estratto
plt.scatter(sample[:, 0], sample[:, 1], alpha=0.6, color='red', s=5)

plt.title('Distribuzione Normale Multivariata con Campione Estratto')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(contour)
plt.show()