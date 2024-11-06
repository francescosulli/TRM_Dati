import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm

p = 0.5           #probabilità per la distribuzione binomiale
lambda_val = 10   #parametro medio per la distribuzione di Poisson
n_values = [10, 20, 50, 100]  # Valori crescenti di n per testare l'approssimazione

#grafico Binomiale
plt.figure(figsize=(14, 6))
for i, n in enumerate(n_values, 1):
    mean_binom = n * p
    std_binom = np.sqrt(n * p * (1 - p))

    #binomiale
    x_binom = np.arange(0, n + 1)
    y_binom = binom.pmf(x_binom, n, p)

    #normale approx
    x_norm = np.linspace(0, n, 100)
    y_norm = norm.pdf(x_norm, mean_binom, std_binom)

    plt.subplot(2, len(n_values) // 2, i)
    plt.bar(x_binom, y_binom, color='skyblue', label='Binomiale', alpha=0.6)
    plt.plot(x_norm, y_norm, color='red', label='Normale approssimata')
    plt.title(f'Distribuzione Binomiale con n={n}')
    plt.xlabel('k')
    plt.ylabel('Probabilità')
    plt.legend()

plt.tight_layout()
plt.show()

#grafico per Poisson
plt.figure(figsize=(6, 4))
mean_poisson = lambda_val
std_poisson = np.sqrt(lambda_val)

#distribuz di Poisson
x_poisson = np.arange(0, lambda_val * 3)
y_poisson = poisson.pmf(x_poisson, lambda_val)

#distribuzione normale aprox
x_norm_poisson = np.linspace(0, lambda_val * 3, 100)
y_norm_poisson = norm.pdf(x_norm_poisson, mean_poisson, std_poisson)

plt.bar(x_poisson, y_poisson, color='lightgreen', label='Poisson', alpha=0.6)
plt.plot(x_norm_poisson, y_norm_poisson, color='red', label='Normale approssimata')
plt.title(f'Distribuzione di Poisson con λ={lambda_val}')
plt.xlabel('k')
plt.ylabel('Probabilità')
plt.legend()
plt.show()