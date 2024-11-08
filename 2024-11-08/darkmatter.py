import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# Parametri della distribuzione lognormale per la massa di DM
mu_DM_true = 14  # Cambiato da 1e14 a 14 per evitare overflow
sigma_DM_true = 0.5  # Deviazione standard della distribuzione lognormale

# Parametri dell'errore di misura
mu_DM_oss = 0  # Media dell'errore (zero)
sigma_DM_oss = 1e13  # Deviazione standard dell'errore

# Numero di campioni
num_samples = 1000

# 1. Simulazione Monte Carlo
# Generazione della massa di DM intrinseca da una lognormale
mass_DM = lognorm.rvs(sigma_DM_true, scale=np.exp(mu_DM_true), size=num_samples)

# Aggiunta dell'errore di misura da una normale
mass_DM_obs = mass_DM + np.random.normal(mu_DM_oss, sigma_DM_oss, num_samples)

# 2. Distribuzione teorica della Massa Osservata
# La distribuzione di masse osservate si ottiene convolvendo la lognormale e la normale
# La deviazione standard combinata si calcola come sqrt(sigma_DM_true^2 + sigma_DM_oss^2)
sigma_DM_obs = np.sqrt(sigma_DM_true**2 + sigma_DM_oss**2)

# Plot della distribuzione osservata tramite simulazione e distribuzione teorica
x = np.linspace(min(mass_DM_obs), max(mass_DM_obs), 500)

# Calcolo della distribuzione teorica analitica
pdf_DM_obs = norm.pdf(x, mu_DM_true, sigma_DM_obs)

# Grafico
plt.hist(mass_DM_obs, bins=30, density=True, alpha=0.6, color='skyblue', label="Campione Massa DM Osservata (Monte Carlo)")
plt.plot(x, pdf_DM_obs, 'r', linewidth=2, label="Distribuzione Teorica (Convoluzione)")
plt.xlabel("Massa di Materia Oscura Osservata (kg)")
plt.ylabel("Densità di Probabilità")
plt.legend()
plt.title("Distribuzione della Massa di Materia Oscura Osservata")
plt.show()