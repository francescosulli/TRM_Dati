import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

# 1. Caricamento dei dati
data = np.genfromtxt("Esercizio4.csv", delimiter=",", names=True)
lnMtot = np.log(data["Mass"])  # ln M_tot (massa totale)
lnMstar = np.log(data["Mstar"])  # ln M_star (massa stellare)

# 2. Controlla se ci sono valori NaN o infiniti nei dati
if np.any(np.isnan(lnMtot)) or np.any(np.isnan(lnMstar)):
    print("Errore: dati contengono NaN!")
    exit()
if np.any(np.isinf(lnMtot)) or np.any(np.isinf(lnMstar)):
    print("Errore: dati contengono infiniti!")
    exit()

# 3. Definizione della log-likelihood
def log_likelihood(theta, lnMtot, lnMstar):
    A, B, sigma_lnMstar = theta
    if sigma_lnMstar <= 0:  # Evita valori non validi
        return -np.inf
    mu = A + B * lnMtot
    try:
        logL = -0.5 * np.sum(((lnMstar - mu) / sigma_lnMstar)**2 + np.log(2 * np.pi * sigma_lnMstar**2))
    except Exception as e:
        print(f"Errore nel calcolo della log-verosimiglianza: {e}")
        return -np.inf
    return logL

# 4. Definizione dei prior
def log_prior(theta):
    A, B, sigma_lnMstar = theta
    # Assicurati che sigma_lnMstar sia positivo e all'interno di un intervallo ragionevole
    if sigma_lnMstar <= 0 or sigma_lnMstar > 10:
        return -np.inf  # Prior invalido
    # Limita i parametri A e B a intervalli ragionevoli
    if not (-10 < A < 10 and -10 < B < 10):
        return -np.inf  # Prior invalido
    return 0.0  # Prior uniforme

# 5. Definizione della distribuzione a posteriori
def log_posterior(theta, lnMtot, lnMstar):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, lnMtot, lnMstar)

# 6. Impostazione dei parametri iniziali per emcee
nwalkers = 50  # Numero di walker
ndim = 3  # Numero di parametri (A, B, sigma_lnMstar)

# Impostazione di intervalli più sicuri per i parametri iniziali
A_initial = np.random.uniform(-1, 1, size=nwalkers)
B_initial = np.random.uniform(-1, 1, size=nwalkers)
sigma_initial = np.random.uniform(0.1, 2, size=nwalkers)  # Evitiamo valori troppo piccoli o troppo grandi

pos = np.array([A_initial, B_initial, sigma_initial]).T  # Impostiamo i parametri iniziali

# 7. Esecuzione del campionamento con emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(lnMtot, lnMstar))

# Aggiungi un controllo per il progresso e per il controllo dei NaN
for _ in sampler.sample(pos, iterations=2000, progress=True):
    if np.any(np.isnan(sampler.chain)) or np.any(np.isinf(sampler.chain)):
        print("NaN o infini rilevati nei parametri durante il campionamento!")
        break

# 8. Estrazione dei campioni
samples = sampler.get_chain(discard=500, thin=15, flat=True)

# 9. Analisi e visualizzazione dei risultati
# Corner plot dei parametri
fig = corner.corner(samples, labels=["A", "B", "sigma_lnMstar"], truths=None)
plt.show()

# 10. Propagazione dell'incertezza nello spazio delle osservabili
lnMtot_pred = np.linspace(np.min(lnMtot), np.max(lnMtot), 100)

# Calcolo del valor medio e intervalli di confidenza
mu_pred = samples[:, 0][:, None] + samples[:, 1][:, None] * lnMtot_pred
sigma_pred = np.mean(samples[:, 2])  # Deviazione standard media

mu_mean = np.mean(mu_pred, axis=0)
mu_conf_int = np.percentile(mu_pred, [16, 84], axis=0)

# Plot del valor medio e intervallo di confidenza
plt.plot(lnMtot_pred, mu_mean, label="Valor medio")
plt.fill_between(lnMtot_pred, mu_conf_int[0], mu_conf_int[1], alpha=0.3, label="1σ intervallo")
plt.scatter(lnMtot, lnMstar, color="red", s=10, label="Dati")
plt.xlabel("ln M_tot")
plt.ylabel("ln M_*")
plt.legend()
plt.show()

# 11. Grafici dei parametri durante gli step di campionamento
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Grafico per il parametro A
axes[0].plot(sampler.chain[:, :, 0], color="C0", alpha=0.3)
axes[0].set_ylabel("A")
axes[0].set_title("Evoluzione di A durante il campionamento")

# Grafico per il parametro B
axes[1].plot(sampler.chain[:, :, 1], color="C1", alpha=0.3)
axes[1].set_ylabel("B")
axes[1].set_title("Evoluzione di B durante il campionamento")

# Grafico per il parametro sigma_lnMstar
axes[2].plot(sampler.chain[:, :, 2], color="C2", alpha=0.3)
axes[2].set_ylabel(r"$\sigma_{lnM_*}$")
axes[2].set_title("Evoluzione di $\sigma_{lnM_*}$ durante il campionamento")

plt.tight_layout()
plt.show()