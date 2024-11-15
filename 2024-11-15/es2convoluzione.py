import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

#prametri grid
grid_size = 51
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]
dA = dx * dy

#funz a gradino
step_function = np.where((np.abs(X) <= 0.5) & (np.abs(Y) <= 0.5), 1, 0)

#parametr distribuz gauss bivariata
mu = [0, 0]
sigma_x = 0.1
sigma_y = 0.2
rho = 0.333
cov_matrix = [[sigma_x**2, rho * sigma_x * sigma_y],
              [rho * sigma_x * sigma_y, sigma_y**2]]

#distribuz gauss bivariata
gaussian_bivariate = multivariate_normal(mean=mu, cov=cov_matrix)
gaussian_pdf = gaussian_bivariate.pdf(np.dstack((X, Y)))

#metodo 1: Convoluzione con integrazione numerica
convolution_trapezoid = np.zeros_like(step_function, dtype=np.float64)

for i in range(grid_size):
    for j in range(grid_size):
        # Traslare la funzione a gradino
        shifted_step = np.roll(np.roll(step_function,
                                       i - grid_size // 2,
                                       axis=0),
                               j - grid_size // 2,
                               axis=1)
        # Calcolare il prodotto punto a punto
        product = shifted_step * gaussian_pdf
        # Integrazione sulle due coordinate
        integral_y = np.trapezoid(product, y, axis=0)
        convolution_trapezoid[i, j] = np.trapezoid(integral_y, x)

#metodo 2: Convoluzione con convolve2d
convolution_convolve2d = convolve2d(step_function,
                                    gaussian_pdf,
                                    mode='same',
                                    boundary='fill',
                                    fillvalue=0)

convolution_convolve2d /= (np.sum(convolution_convolve2d) * dA)

#metodo 3: Convoluzione MC
N_samples = 10000000
step_samples_x = np.random.uniform(-0.5, 0.5, N_samples)
step_samples_y = np.random.uniform(-0.5, 0.5, N_samples)
gaussian_samples = np.random.multivariate_normal(mu, cov_matrix, N_samples)
mc_samples = np.column_stack(
    (step_samples_x, step_samples_y)) + gaussian_samples

hist, x_edges, y_edges = np.histogram2d(mc_samples[:, 0],
                                        mc_samples[:, 1],
                                        bins=grid_size,
                                        range=[[-1, 1], [-1, 1]])
convolution_monte_carlo = hist / (dx * dy * N_samples)

#distribuz marginalizzate per ciascun metodo
#met1
marginal_x_trapezoid = np.trapezoid(convolution_trapezoid, y, axis=0)
marginal_y_trapezoid = np.trapezoid(convolution_trapezoid, x, axis=1)

#met2
marginal_x_convolve2d = np.trapezoid(convolution_convolve2d, y, axis=0)
marginal_y_convolve2d = np.trapezoid(convolution_convolve2d, x, axis=1)

#met3
marginal_x_mc = np.sum(convolution_monte_carlo, axis=0) * dy
marginal_y_mc = np.sum(convolution_monte_carlo, axis=1) * dx

#plot
fig, axs = plt.subplots(3, 3, figsize=(6, 4))

# 1a riga: Funzioni di partenza
g1 = axs[0, 0].contourf(X, Y, step_function, cmap='inferno')
axs[0, 0].set_title('Funzione a gradino')
g2 = axs[0, 1].contourf(X, Y, gaussian_pdf, cmap='inferno')
axs[0, 1].set_title('Distribuzione Gaussiana')
axs[0, 2].axis('off')  # Slot vuoto

# 2a riga: Convoluzioni
g3 = axs[1, 0].contourf(X, Y, convolution_trapezoid, cmap='inferno', levels=100)
axs[1, 0].set_title('Convoluzione con np.trapezoid')
g4 = axs[1, 1].contourf(X, Y, convolution_convolve2d, cmap='inferno', levels=100)
axs[1, 1].set_title('Convoluzione con convolve2d')
g5 = axs[1, 2].contourf(Y, X, convolution_monte_carlo, cmap='inferno', levels=100)
axs[1, 2].set_title('Convoluzione Monte Carlo')

# 3a riga: Marginalizzazioni
axs[2, 0].plot(x, marginal_x_trapezoid, label='Marginal X (trapezoid)', color='blue')
axs[2, 0].plot(y, marginal_y_trapezoid, label='Marginal Y (trapezoid)', color='red')
axs[2, 0].hist(mc_samples[:,0], bins=51, density=True, alpha=0.5, label="MC Samples X", color='blue', linestyle='dashed')
axs[2, 0].hist(mc_samples[:,1], bins=51, density=True, alpha=0.5, label="MC Samples Y", color='red', linestyle='dashed')
axs[2, 0].set_title('Marginali np.trapezoid')
axs[2, 0].legend()

axs[2, 1].plot(x, marginal_x_convolve2d, label='Marginal X (conv2d)', color='blue')
axs[2, 1].plot(y, marginal_y_convolve2d, label='Marginal Y (conv2d)', color='red')
axs[2, 1].hist(mc_samples[:,0], bins=51, density=True, alpha=0.5, label="MC Samples X", color='blue', linestyle='dashed')
axs[2, 1].hist(mc_samples[:,1], bins=51, density=True, alpha=0.5, label="MC Samples Y", color='red', linestyle='dashed')
axs[2, 1].set_title('Marginali convolve2d')
axs[2, 1].legend()

axs[2, 2].plot(x, marginal_x_mc, label='Marginal X (MC)', color='blue')
axs[2, 2].plot(y, marginal_y_mc, label='Marginal Y (MC)', color='red')
axs[2, 2].hist(mc_samples[:,0], bins=51, density=True, alpha=0.5, label="MC Samples X", color='blue', linestyle='dashed')
axs[2, 2].hist(mc_samples[:,1], bins=51, density=True, alpha=0.5, label="MC Samples Y", color='red', linestyle='dashed')
axs[2, 2].set_title('Marginali Monte Carlo')
axs[2, 2].legend()

#colorbars
fig.colorbar(g1, ax=axs[0, 0], orientation='vertical')
fig.colorbar(g2, ax=axs[0, 1], orientation='vertical')
fig.colorbar(g3, ax=axs[1, 0], orientation='vertical')
fig.colorbar(g4, ax=axs[1, 1], orientation='vertical')
fig.colorbar(g5, ax=axs[1, 2], orientation='vertical')

plt.tight_layout()
plt.show()