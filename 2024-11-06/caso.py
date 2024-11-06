#import
import matplotlib.pyplot as plt
import timeit


#generatore
def MCRNG(seed, A, C, M, num_samples=100):
    numbers = []
    current = seed
    for _ in range(num_samples):
        current = (A * current + C) % M
        numbers.append(current)
    return numbers

#arametri
seed = 10
A = 7
C = 3
M = 2**11

sequence = MCRNG(seed, A, C, M)

#tempo esecuzione
execution_time = timeit.timeit(lambda: MCRNG(seed, A, C, M), number=100)
print(f"Tempo di esecuzione per generare il campione: {execution_time:.5f} secondi")

#grafico
plt.figure(figsize=(6, 4))
plt.scatter(range(len(sequence)), sequence, s=10, color='blue')
plt.title("Sequenza di Numeri Pseudo-casuali Generata con LCG")
plt.xlabel("Indice")
plt.ylabel("Valore")
plt.show()