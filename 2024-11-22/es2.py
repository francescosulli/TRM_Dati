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
    lower_limit = -10
    upper_limit = 10
    return integrate.quad(integrand, lower_limit, upper_limit)[0]

# Costruzione della distribuzione target
x_values = np.linspace(-5, 10, 500)
pdf_values = np.array([pdf_convolution(x) for x in x_values])

# Trova il massimo della PDF convoluta per il coefficiente c
max_pdf = max(pdf_values)

# Distribuzione proposta: una gaussiana ampia
proposal_mu, proposal_sigma = 2.5, 3  # Parametri della gaussiana proposta
proposal_dist = stats.norm(loc=proposal_mu, scale=proposal_sigma)

# Funzione di accettazione
def rejection_sampling(n_samples):
    samples = []
    while len(samples) < n_samples:
        # Genera un campione dalla distribuzione proposta
        x = proposal_dist.rvs()
        # Calcola la probabilità di accettazione
        f_x = pdf_convolution(x)  # PDF target
        g_x = proposal_dist.pdf(x)  # PDF della proposta
        u = np.random.rand()  # Campione uniforme [0, 1]
        # Accetta il campione se u < f(x) / (c * g(x))
        if u < f_x / (max_pdf * g_x):
            samples.append(x)
    return np.array