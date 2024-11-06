import numpy as np

#parametri
N = 15  #numero di domande
p = 1/4  #probabilità di rispondere correttamente

risposte_corrette = np.random.binomial(N, p)

print(f"Numero di risposte corrette con scelta casuale: {risposte_corrette}")