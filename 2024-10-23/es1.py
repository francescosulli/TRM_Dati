import pandas as pd
import numpy as np

#serie
serie = pd.Series(np.random.randint(1, 101, 10))

#calcolo
media = serie.mean()
deviazione_standard = serie.std()
somma = serie.sum()

#confronto
valori_minori_50 = serie[serie < 50]

#plot
print("Serie:\n", serie)
print("Media:", media)
print("Deviazione standard:", deviazione_standard)
print("Somma:", somma)
print("Valori minori di 50:\n", valori_minori_50)