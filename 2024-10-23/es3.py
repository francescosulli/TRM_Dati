import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Nemo_6670.dat'

#carica il file
try:
    df = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, 
                     names=["MsuH", "m_ini", "logL", "logTe", "M_ass", "b_ass", "y_ass", 
                            "m_app", "b_y", "dist", "abs_dist", "ID_parent", "age_parent"])
    print("File caricato con successo!")
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
    exit()
except Exception as e:
    print(f"Errore durante il caricamento del file: {e}")
    exit()

#controllo dati
if df.empty:
    print("Errore: Il DataFrame è vuoto. Controlla il file.")
    exit()

print("Prime righe del DataFrame:")
print(df.head())

if 'M_ass' not in df.columns or 'b_y' not in df.columns or 'age_parent' not in df.columns:
    print("Errore: Il file non contiene le colonne 'M_ass', 'b_y' o 'age_parent'.")
    print("Colonne disponibili:", df.columns)
    exit()

#bin di età
df['age_bin'] = pd.cut(df['age_parent'], bins=7)
labels = df['age_bin'].unique()

#grafico per ciascun bin di età
plt.figure(figsize=(10, 6))

for label in labels:
    subset = df[df['age_bin'] == label]
    plt.scatter(subset['b_y'], subset['M_ass'], label=f'Età: {label}', alpha=0.7)

#etichette
plt.title('Magnitudine Assoluta (M_ass) vs Colore (b-y) per diversi Bins di Età')
plt.xlabel('Colore (b-y)')
plt.ylabel('Magnitudine Assoluta (M_ass)')
plt.gca().invert_yaxis()  # Inverte l'asse y
plt.legend(title='Bins di Età')
plt.grid(True)
plt.show()