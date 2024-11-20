import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#dati
filename = "Nemo_6670.dat"
data = pd.read_csv(filename, comment='#', delim_whitespace=True)

#estrazione colonne
columns = ['MsuH', 'm_ini', 'logL', 'logTe', 'M_ass', 'b_ass', 'y_ass', 'm_app', 'b-y', 'dist', 'abs_dist', 'ID_parent', 'age_parent']
data.columns = columns

#debug
print(data.head())

#calcolare PCA
features = ['logL', 'logTe', 'M_ass', 'b_ass', 'y_ass', 'm_app', 'b-y', 'dist', 'abs_dist']
x = data[features].values
x = StandardScaler().fit_transform(x)  #standardizzazione dei dati

#PCA
pca = PCA()
principal_components = pca.fit_transform(x)

#Calcolo varianza
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

#numero componenti sono necessarie
threshold = 0.95  #soglia del 95%
n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

print(f"Numero di componenti necessarie per descrivere il 95% della varianza: {n_components}")

#grafico varianza
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance_ratio, marker='o', linestyle='--', label="Cumulativa")
plt.axhline(y=threshold, color='r', linestyle='-', label=f"{int(threshold * 100)}% Varianza")
plt.xlabel("Numero di componenti principali")
plt.ylabel("Varianza spiegata cumulativa")
plt.title("PCA - Varianza spiegata cumulativa")
plt.legend()
plt.grid()
plt.show()

#grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#grafico HR
axs[0, 0].scatter(data['b-y'], data['M_ass'], c=data['logL'], cmap='viridis', s=10)
axs[0, 0].invert_yaxis()
axs[0, 0].set_xlabel('Colore (b-y)')
axs[0, 0].set_ylabel('Massa [M⊙]')
axs[0, 0].set_title('Diagramma HR - Dati iniziali')
axs[0, 0].grid()
fig.colorbar(axs[0, 0].collections[0], ax=axs[0, 0], label='log(Luminosità)')

#grafico HR con PCA completa
x_full_pca = principal_components[:, :n_components]
axs[0, 1].scatter(data['b-y'], data['M_ass'], c=x_full_pca[:, 0], cmap='viridis', s=10)
axs[0, 1].invert_yaxis()
axs[0, 1].set_xlabel('Colore (b-y)')
axs[0, 1].set_ylabel('Massa [M⊙]')
axs[0, 1].set_title('Diagramma HR - PCA completa')
axs[0, 1].grid()
fig.colorbar(axs[0, 1].collections[0], ax=axs[0, 1], label='PC1')

#grafico HR con PCA parziale
x_partial_pca = principal_components[:, :n_components]
axs[1, 0].scatter(data['b-y'], data['M_ass'], c=x_partial_pca[:, 0], cmap='viridis', s=10)
axs[1, 0].invert_yaxis()
axs[1, 0].set_xlabel('Colore (b-y)')
axs[1, 0].set_ylabel('Massa [M⊙]')
axs[1, 0].set_title('Diagramma HR - PCA parziale')
axs[1, 0].grid()
fig.colorbar(axs[1, 0].collections[0], ax=axs[1, 0], label='PC1')

axs[1, 1].axis('off')

plt.tight_layout()
plt.show()