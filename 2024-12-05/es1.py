import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

# Caricamento dati
data = np.genfromtxt("Esercizio4.csv", delimiter=",", names=True)
lnMtot = np.log(data["Mtot"])
lnMstar = np.log(data["Mstar"])
lnMgas = np.log(data["Mgas"])

# Controllo di NaN o infiniti
if np.any(np.isnan([lnMtot, lnMstar, lnMgas])):
    print("Errore: dati contengono NaN!")
    exit()
if np.any(np.isinf([lnMtot, lnMstar, lnMgas])):
    print("Errore: dati contengono infiniti!")
    exit()

# Log-verosimiglianza
def log_likelihood(theta, lnMtot, lnMstar, lnMgas):
    A1, B1, sigma1, A2, B2, sigma2, rho = theta
    if sigma1 <= 0 or sigma2 <= 0 or abs(rho) >= 1:  # Evita valori non validi
        return -np.inf

    mu1 = A1 + B1 * lnMtot
    mu2 = A2 + B2 * lnMtot
    cov_matrix = np.array([[sigma1**2, rho * sigma1 * sigma2], 
                           [rho * sigma1 * sigma2, sigma2**2]])
    inv_cov = np.linalg.inv(cov_matrix)
    diff = np.vstack([lnMstar - mu1, lnMgas - mu2])
    log_det_cov = np.log(np.linalg.det(cov_matrix))

    try:
        logL = -0.5 * np.sum(diff.T @ inv_cov * diff.T) - 0.5 * log_det_cov - np.log(2 * np.pi) * len(lnMtot)
    except np.linalg.LinAlgError:
        return -np.inf
    return logL

# Prior
def log_prior(theta):
    A1, B1, sigma1, A2, B2, sigma2, rho = theta
    if not (-10 < A1 < 10 and -10 < B1 < 10 and 0 < sigma1 < 10 and
            -10 < A2 < 10 and -10 < B2 < 10 and 0 < sigma2 < 10 and 
            -1 < rho < 1):
        return -np.inf
    return 0.0  # Prior uniforme

# Posteriori
def log_posterior(theta, lnMtot, lnMstar, lnMgas):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, lnMtot, lnMstar, lnMgas)

# Parametri iniziali
nwalkers = 100
ndim = 7
initial_pos = np.random.uniform(-1, 1, (nwalkers, ndim))
initial_pos[:, 2] = np.random.uniform(0.1, 2, nwalkers)  # sigma1
initial_pos[:, 5] = np.random.uniform(0.1, 2, nwalkers)  # sigma2
initial_pos[:, 6] = np.random.uniform(-0.5, 0.5, nwalkers)  # rho

# Sampling con emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(lnMtot, lnMstar, lnMgas))
sampler.run_mcmc(initial_pos, 3000, progress=True)

# Campioni
samples = sampler.get_chain(discard=500, thin=15, flat=True)

# Corner plot
labels = ["A1", "B1", "sigma1", "A2", "B2", "sigma2", "rho"]
fig = corner.corner(samples, labels=labels)
plt.show()

# Propagazione dell'incertezza
lnMtot_pred = np.linspace(np.min(lnMtot), np.max(lnMtot), 100)
mu1_pred = samples[:, 0][:, None] + samples[:, 1][:, None] * lnMtot_pred
mu2_pred = samples[:, 3][:, None] + samples[:, 4][:, None] * lnMtot_pred

mu1_mean = np.mean(mu1_pred, axis=0)
mu1_conf_int = np.percentile(mu1_pred, [16, 84], axis=0)

mu2_mean = np.mean(mu2_pred, axis=0)
mu2_conf_int = np.percentile(mu2_pred, [16, 84], axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.fill_between(lnMtot_pred, mu1_conf_int[0], mu1_conf_int[1], alpha=0.3, label="1σ intervallo (M*)")
plt.fill_between(lnMtot_pred, mu2_conf_int[0], mu2_conf_int[1], alpha=0.3, label="1σ intervallo (Mgas)")
plt.plot(lnMtot_pred, mu1_mean, label="Valor medio (M*)", color="C0")
plt.plot(lnMtot_pred, mu2_mean, label="Valor medio (Mgas)", color="C1")
plt.scatter(lnMtot, lnMstar, label="Dati M*", color="C0", s=10)
plt.scatter(lnMtot, lnMgas, label="Dati Mgas", color="C1", s=10)
plt.xlabel("ln M_tot")
plt.ylabel("ln M_* / ln M_gas")
plt.legend()
plt.show()