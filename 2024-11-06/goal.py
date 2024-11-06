import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

#parametro distribuzione di Poisson
lambda_val = 2.5

#numero max goal
max_goals = 10

#calcola le probabilità di ottenere 0, 1, ..., max_goals
goals = np.arange(0, max_goals + 1)
probabilities = poisson.pmf(goals, lambda_val)

#print
for goal, prob in zip(goals, probabilities):
    print(f"Probabilità di {goal} gol: {prob:.4f}")

#grafico
plt.bar(goals, probabilities, color='skyblue')
plt.xlabel('Numero di gol')
plt.ylabel('Probabilità')
plt.title(f'Distribuzione di Poisson per gol in una partita (lambda = {lambda_val})')
plt.show()