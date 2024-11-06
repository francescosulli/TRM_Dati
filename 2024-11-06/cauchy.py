import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000   #numero totale
n_trials = 1000    #numero di prove (campioni da cui calcolare la media)
n_bins = 50        #numero bin

#funzione per media
def cauchy_sample_means(sample_size, n_trials):
    means = []
    for _ in range(n_trials):
        sample = np.random.standard_cauchy(sample_size)  #campioni da Cauchy
        means.append(np.mean(sample))
    return means

#generazione e visual dei risultati
sample_sizes = [10, 50, 100, 500]  #diverse dimensioni di campioni
plt.figure(figsize=(12, 8))

#print
for i, sample_size in enumerate(sample_sizes):
    means = cauchy_sample_means(sample_size, n_trials)
    plt.subplot(2, 2, i+1)
    plt.hist(means, bins=n_bins, density=True, alpha=0.75, label=f'Media - N={sample_size}')
    plt.title(f'Media di Cauchy - Dimensione campione N={sample_size}')
    plt.xlabel('Media')
    plt.ylabel('Densità')
    plt.legend()

plt.tight_layout()
plt.show()