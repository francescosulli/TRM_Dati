import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Parametri della Gaussiana e della Lognormale
mu_gauss, sigma_gauss = 0, 1  # Media e deviazione standard della Gaussiana
mu_lognorm, sigma_lognorm = 0, 0.5  # Parametri della Lognormale

# Definizione della PDF convoluta
def pdf_convolution(x):
    def integrand(t):
        gauss = stats.norm.pdf(t, mu_gauss, sigma_gauss)
        lognorm = stats.lognorm.pdf(x - t, sigma_lognorm, scale=np.exp(mu_lognorm))
        return gauss * lognorm

    # Limiti dell'integrazione
    lower_limit = -10  # Limite inferiore (ad esempio -10)
    upper_limit = 10   # Limite superiore (ad esempio 10)
    return integrate.quad(integrand, lower_limit, upper_limit)[0]

# Calcolo della PDF convoluta su un intervallo
x_values = np.linspace(-5, 10, 500)  # Intervallo di campionamento
pdf_values = np.array([pdf_convolution(x) for x in x_values])

# Calcolo della CDF dalla PDF
cdf_values = np.cumsum(pdf_values) * (x_values[1] - x_values[0])
cdf_values /= cdf_values[-1]  # Normalizzazione

# Metodo della trasformata inversa
def inverse_transform_sampling(n_samples):
    random_uniform = np.random.rand(n_samples)  # Campioni uniformi [0, 1]
    return np.interp(random_uniform, cdf_values, x_values)

# Generazione dei campioni
n_samples = 10000
samples = inverse_transform_sampling(n_samples)

# Visualizzazione
plt.figure(figsize=(12, 6))

# PDF convoluta
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label="PDF Convoluta", color="blue")
plt.title("PDF della Convoluzione")
plt.xlabel("x")
plt.ylabel("Densità")
plt.legend()

# Istogramma dei campioni
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.7, label="Campioni")
plt.plot(x_values, pdf_values, label="PDF Convoluta", color="blue")
plt.title("Istogramma dei Campioni")
plt.xlabel("x")
plt.ylabel("Densità")
plt.legend()

plt.tight_layout()
plt.show()