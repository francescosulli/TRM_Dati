import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Esercizio1.csv", header=None, names=["altezza", "errore"])

#correz dati
data["altezza"] = pd.to_numeric(data["altezza"], errors="coerce")
data["errore"] = pd.to_numeric(data["errore"], errors="coerce")

#rimozione eventuali colonne
data = data.dropna()


altezze = data["altezza"].values
errori = data["errore"].values

#funz posterior
def calcola_posterior(media_grid, altezze, errori):
    """
    Calcola il posterior per una griglia di valori di media dati i dati osservati e i loro errori.
    """
    likelihood = np.prod([
        np.exp(-0.5 * ((altezza - media_grid) / errore)**2) / (errore * np.sqrt(2 * np.pi))
        for altezza, errore in zip(altezze, errori)
    ], axis=0)
    prior = np.ones_like(media_grid)
    posterior = likelihood * prior
    return posterior / np.sum(posterior)  #normalizzaz

#griglia
media_grid = np.linspace(min(altezze) - 3, max(altezze) + 3, 1000)

#posterior
posterior = calcola_posterior(media_grid, altezze, errori)

#valore medio massimo a posteriori (MAP)
media_map = media_grid[np.argmax(posterior)]


plt.figure(figsize=(5, 3))
plt.plot(media_grid, posterior, label="Posterior", color="blue")
plt.axvline(media_map, color="red", linestyle="--", label=f"MAP = {media_map:.2f}")
plt.xlabel("Media")
plt.ylabel("Densità di probabilità")
plt.title("Posterior del Valore Medio")
plt.legend()
plt.grid()
plt.show()

print(f"Il valore medio massimo a posteriori (MAP) è: {media_map:.2f}")