import pandas as pd
import numpy as np

serie = pd.Series(np.random.randint(1, 101, 10))

media = serie.mean()
deviazione_standard = serie.std()
somma = serie.sum()

valori_minori_50 = serie[serie < 50]

print("Serie:\n", serie)
print("Media:", media)
print("Deviazione standard:", deviazione_standard)
print("Somma:", somma)
print("Valori minori di 50:\n", valori_minori_50)