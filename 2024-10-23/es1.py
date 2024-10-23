import pandas as pd
import numpy as np

# Creazione di una Pandas Series con 10 numeri interi casuali compresi tra 1 e 100
serie = pd.Series(np.random.randint(1, 101, 10))

# Calcolo della media, deviazione standard e somma
media = serie.mean()
deviazione_standard = serie.std()
somma = serie.sum()

# Filtraggio dei valori minori di 50
valori_minori_50 = serie[serie < 50]

# Risultati
print("Serie:\n", serie)
print("Media:", media)
print("Deviazione standard:", deviazione_standard)
print("Somma:", somma)
print("Valori minori di 50:\n", valori_minori_50)