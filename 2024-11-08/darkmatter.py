import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy.integrate import cumulative_trapezoid


xarr = np.linspace(-5, 5, 10000)

#parametri DM osservata (distribuzione normale)
obs_dm_mean = 0.
obs_dm_sigma = 0.3

#parametri DM vera (distribuzione lognormale)
true_dm_mean = np.log(0.6)
true_dm_sigma = 0.7

#PDF teoriche
obs_dm_pdf = norm.pdf(xarr, loc=obs_dm_mean, scale=obs_dm_sigma)
true_dm_pdf = lognorm.pdf(xarr, true_dm_sigma, scale=np.exp(true_dm_mean))

#MC
n_samples = 100000
sample_obs_dm = np.random.normal(size=n_samples, loc=obs_dm_mean, scale=obs_dm_sigma)
sample_true_dm = np.random.lognormal(size=n_samples, mean=true_dm_mean, sigma=true_dm_sigma)
sample_combined = sample_obs_dm + sample_true_dm

#plot singoli
plt.figure(figsize=(6, 4))
plt.hist(sample_obs_dm, bins=71, density=True, alpha=0.5, label='Campioni Errore di Misura')
plt.hist(sample_true_dm, bins=71, density=True, alpha=0.5, label='Campioni DM Vera')
plt.plot(xarr, obs_dm_pdf, label='PDF Errore di Misura')
plt.plot(xarr, true_dm_pdf, label='PDF DM Vera')
plt.xlabel('Valore')
plt.ylabel('Densità')
plt.legend()
plt.title("Distribuzioni Singole")
plt.xlim(-2.5, 5)
plt.show()

#plot distribuzione combinata osservata
plt.figure(figsize=(6, 4))
plt.hist(sample_combined, bins=71, density=True, alpha=0.5, label='Campioni Massa DM Osservata')
plt.plot(xarr, obs_dm_pdf, label='PDF Errore di Misura')
plt.plot(xarr, true_dm_pdf, label='PDF DM Vera')

#convoluzione tra le due PDF teoriche
convoluzione = np.convolve(true_dm_pdf, obs_dm_pdf, mode='same')
convoluzione_normalizzata = convoluzione / np.trapezoid(convoluzione, xarr)
plt.plot(xarr, convoluzione_normalizzata, 'r-', label='Convoluzione PDF teoriche')

#calc cumulativa
yarr = convoluzione_normalizzata
cumulativa = cumulative_trapezoid(yarr, xarr, initial=0)

#quantili della cumulativa
quantili = [0.16, 0.5, 0.84]
percentili = [np.interp(q, cumulativa, xarr) for q in quantili]

# Stampa solo i numeri, senza np.float64
print('Quantili teorici (PDF):', [p.item() for p in percentili])

#quantili dei combinati
campioni_quantili = np.percentile(sample_combined, [16, 50, 84])
print('Quantili campioni:', campioni_quantili)

plt.xlabel('Valore')
plt.ylabel('Densità')
plt.legend()
plt.title("Distribuzione della Massa DM Osservata")
plt.xlim(-2.5, 5)
plt.show()