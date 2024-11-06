import numpy as np
import matplotlib.pyplot as plt

#parametri
n_samples = 3000        #numero di elementi per ciascun campione
n_realizations = 2000   #numero di realizzazioni

mean_values = []

#esegui realizzazioni
for _ in range(n_realizations):
    # Estrai numeri casuali dalla distribuzione uniforme [0,1]
    sample = np.random.uniform(0, 1, n_samples)
    # Calcola la media del campione
    sample_mean = np.mean(sample)
    # Aggiungi la media alla lista
    mean_values.append(sample_mean)

# Converti la lista in un array numpy per comodità
mean_values = np.array(mean_values)

# Plot della distribuzione dei valori medi
plt.hist(mean_values, bins=30, density=True, color='skyblue', edgecolor='black')
plt.xlabel('Valore medio')
plt.ylabel('Densità di probabilità')
plt.title(f'Distribuzione dei valori medi (1000 realizzazioni di campioni di 2000 elementi)')
plt.show()