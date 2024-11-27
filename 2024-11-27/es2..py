import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Esercizio2.csv", header=None, names=["M1", "errore_M1", "M2", "errore_M2"])

#conversione colonne numeri
data["M1"] = pd.to_numeric(data["M1"], errors="coerce")
data["errore_M1"] = pd.to_numeric(data["errore_M1"], errors="coerce")
data["M2"] = pd.to_numeric(data["M2"], errors="coerce")
data["errore_M2"] = pd.to_numeric(data["errore_M2"], errors="coerce")

#rimozione righe sbagliate
data = data.dropna()

M1 = data["M1"].values
errore_M1 = data["errore_M1"].values
M2 = data["M2"].values
errore_M2 = data["errore_M2"].values

#posterior
def calcola_posterior_massa(media_grid, M1, errore_M1, M2, errore_M2):
    """
    Calcola il posterior per la massa dei neutrini data la differenza tra M1 e M2
    con errori associati.
    """
   
    massa_neutrini = M1 - M2
    errore_massa = np.sqrt(errore_M1**2 + errore_M2**2)

    
    likelihood = np.prod([
        np.exp(-0.5 * ((massa_neutrini - media_grid) / errore_massa)**2) / (errore_massa * np.sqrt(2 * np.pi))
        for massa_neutrini, errore_massa in zip(massa_neutrini, errore_massa)
    ], axis=0)

    #prior
    prior = np.ones_like(media_grid)

    posterior = likelihood * prior
    return posterior / np.sum(posterior)  

#griglia
media_grid = np.linspace(min(M1 - M2) - 3, max(M1 - M2) + 3, 1000)

#posterior
posterior = calcola_posterior_massa(media_grid, M1, errore_M1, M2, errore_M2)

#assimo a posteriori (MAP)
massa_map = media_grid[np.argmax(posterior)]

plt.figure(figsize=(10, 6))
plt.plot(media_grid, posterior, label="Posterior", color="blue")
plt.axvline(massa_map, color="red", linestyle="--", label=f"MAP = {massa_map:.2f}")
plt.xlabel("Massa dei Neutrini")
plt.ylabel("Densità di probabilità")
plt.title("Posterior della Massa dei Neutrini")
plt.legend()
plt.grid()
plt.show()

print(f"La massa dei neutrini massimo a posteriori (MAP) è: {massa_map:.2f}")