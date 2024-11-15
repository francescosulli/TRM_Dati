import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

#param della grid
grid_size = 51
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

dx = x[1]-x[0]
dy = y[1]-y[0]
dA = dx*dy
#funz a gradino
step_function = np.where((np.abs(X) <= 0.5) & (np.abs(Y) <= 0.5), 1, 0)

#param distribuz gauss bivariata
mu = [0, 0]
sigma_x = 0.1
sigma_y = 0.2
rho = 0.333
cov_matrix = [[sigma_x**2, rho * sigma_x * sigma_y],
              [rho * sigma_x * sigma_y, sigma_y**2]]

#distribuz gauss bivariata
gaussian_bivariate = multivariate_normal(mean=mu, cov=cov_matrix)
gaussian_pdf = gaussian_bivariate.pdf(np.dstack((X, Y)))

#metodo 1:convoluzione con integraz numerica
convolution_trapz = np.zeros_like(step_function, dtype=np.float64)
dx = dy = x[1] - x[0]  #passo della griglia

for i in range(grid_size):
    for j in range(grid_size):
        #traslare la funz a gradino
        shifted_step = np.roll(np.roll(step_function,
                                       i - grid_size // 2,
                                       axis=0),
                               j - grid_size // 2,
                               axis=1)
        #calc prodotto punto a punto
        product = shifted_step * gaussian_pdf
        #integraz sulle due coord
        integral_y = np.trapezoid(product, y, axis=0)
        convolution_trapz[i, j] = np.trapezoid(integral_y, x)

#metodo 2:convoluz con convolve2d
convolution_convolve2d = convolve2d(step_function,
                                    gaussian_pdf,
                                    mode='same',
                                    boundary='fill',
                                    fillvalue=0)

convolution_convolve2d = convolution_convolve2d /(np.sum(convolution_convolve2d )*dA)

#metodo 3:convoluz MC
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

#plot
fig, axs = plt.subplots(2, 3, figsize=(6, 4))

#1a riga:funzioni di partenza
g1 = axs[0, 0].contourf(X, Y, step_function, cmap='inferno')
axs[0, 0].set_title('Funzione a gradino')
g2 = axs[0, 1].contourf(X, Y, gaussian_pdf, cmap='inferno')
axs[0, 1].set_title('Distribuzione Gaussiana')
axs[0, 2].axis('off')  #slot vuoto

#2a riga:convoluzioni
g3 = axs[1, 0].contourf(X, Y, convolution_trapz, cmap='inferno', levels=100)
axs[1, 0].set_title('Convoluzione con np.trapezoid')
g4 = axs[1, 1].contourf(X, Y, convolution_convolve2d, cmap='inferno', levels=100)
axs[1, 1].set_title('Convoluzione con convolve2d')
g5 = axs[1, 2].contourf(Y, X, convolution_monte_carlo, cmap='inferno', levels=100)
axs[1, 2].set_title('Convoluzione Monte Carlo')

#colorbar
fig.colorbar(g1, ax=axs[0, 0], orientation='vertical')
fig.colorbar(g2, ax=axs[0, 1], orientation='vertical')
fig.colorbar(g3, ax=axs[1, 0], orientation='vertical')
fig.colorbar(g4, ax=axs[1, 1], orientation='vertical')
fig.colorbar(g5, ax=axs[1, 2], orientation='vertical')

plt.tight_layout()
plt.show()