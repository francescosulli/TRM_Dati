import numpy as np
import emcee
import matplotlib.pyplot as plt

#upload
data = np.genfromtxt("Esercizio3.csv", delimiter=",", names=True)
heights = data["Altezza"]
errors = data["Errore"]

#funzione likelihood
def log_likelihood(params, heights, errors):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  #per evitare valori fisicamente non validi
    return -0.5 * np.sum(((heights - mu) / np.sqrt(sigma**2 + errors**2))**2 + 
                         np.log(2 * np.pi * (sigma**2 + errors**2)))

#funz prior
def log_prior(params):
    mu, sigma = params
    if 0 < sigma < 50 and 140 < mu < 210:
        return 0.0
    return -np.inf

#funzione emcee
def log_probability(params, heights, errors):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, heights, errors)

#inizializzazione campionamento
ndim = 2  #N parametri: mu e sigma
nwalkers = 50
nsteps = 5000

#valori iniziali casuali walker
initial_guesses = np.random.normal([170, 10], [5, 2], size=(nwalkers, ndim))
#i picchi alti e bassi negli step dei walker dipendono dai valori della gaussiana, in particolare li ho messi abbastanza grandi

#creazione sampler emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(heights, errors))

#esecuzione campionamento
sampler.run_mcmc(initial_guesses, nsteps, progress=True)

#estraz dei risultati
samples = sampler.get_chain(discard=0, thin=10, flat=True)
#il discard serve a togliere i picchi iniziali, il thin conserva un campione ogni 15 passi

#val medi e degli intervalli di conf
mu_mcmc, sigma_mcmc = np.mean(samples, axis=0)
mu_std, sigma_std = np.std(samples, axis=0)

print(f"Valore medio inferito (mu): {mu_mcmc:.2f} ± {mu_std:.2f}")
print(f"Deviazione standard inferita (sigma): {sigma_mcmc:.2f} ± {sigma_std:.2f}")

#output

fig, axes = plt.subplots(2, figsize=(5, 3), sharex=True)
labels = ["mu", "sigma"]

for i in range(ndim):
    ax = axes[i]
    for walker in sampler.chain[:, :, i]:
        ax.plot(walker, alpha=0.5, lw=0.5)
    ax.set_ylabel(labels[i])
    ax.axhline(np.mean(samples[:, i]), color="r", linestyle="--", label="Media finale")
    ax.legend()

axes[-1].set_xlabel("Step")
plt.tight_layout()
plt.show()

import corner
fig = corner.corner(samples, labels=["mu", "sigma"], truths=[mu_mcmc, sigma_mcmc])
plt.show()