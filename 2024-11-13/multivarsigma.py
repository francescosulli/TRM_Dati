import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#parametri
mu = [0, 0]  #media
sigma = [[1, 0.5], [0.5, 1]]  #matrice delle covarianze

#griglia
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

#normale multivariata
rv = multivariate_normal(mu, sigma)
Z = rv.pdf(pos)

#campione estratto
sample = rv.rvs(size=500)

#calcolare la densità alla media (mu)
mu_density = rv.pdf(mu)

#calcolare le soglie di confidenza (densità)
confidence_levels = [0.6827, 0.9545, 0.9973]
levels = [mu_density * (1 - ci) for ci in confidence_levels]

levels.sort()

#calcolare la densità per ogni punto nel campione
sample_densities = rv.pdf(sample)

#inizializzare contatori
counts = [0, 0, 0]

#conta quanti punti cadono in ciascun intervallo
for density in sample_densities:
    if density >= levels[0]:
        counts[0] += 1
    if density >= levels[1]:
        counts[1] += 1
    if density >= levels[2]:
        counts[2] += 1

print(f"Punti dentro il 68% (1 sigma): {counts[2]}")
print(f"Punti dentro il 95% (2 sigma): {counts[1]}")
print(f"Punti dentro il 99.7% (3 sigma): {counts[0]}")

#plot delle aree di confidenza
plt.figure(figsize=(6, 4))
plt.contour(X, Y, Z, levels=levels, colors=['blue', 'green', 'red'], linewidths=2)

# campione estratto
plt.scatter(sample[:, 0], sample[:, 1], alpha=0.6, color='red', s=5)

plt.title('Distribuzione Normale Multivariata con Aree di Confidenza')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()