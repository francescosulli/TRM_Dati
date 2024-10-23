import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specifica il percorso al file .dat
file_path = 'Nemo_6670.dat'

# Prova a caricare il file
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

# Controlla che il DataFrame contenga dati
if df.empty:
    print("Errore: Il DataFrame è vuoto. Controlla il file.")
    exit()

# Mostra le prime righe del DataFrame per debug
print("Prime righe del DataFrame:")
print(df.head())

# Controlla che le colonne 'M_ass', 'b_y' e 'm_ini' esistano nel DataFrame
if 'M_ass' not in df.columns or 'b_y' not in df.columns or 'm_ini' not in df.columns:
    print("Errore: Il file non contiene le colonne 'M_ass', 'b_y' o 'm_ini'.")
    print("Colonne disponibili:", df.columns)
    exit()

# Normalizza i valori di massa per il gradiente di colori
mass_values = df['m_ini']
norm = plt.Normalize(mass_values.min(), mass_values.max())
colors = plt.cm.viridis(norm(mass_values))  # Scegli una colormap

# Crea il grafico scatter
plt.figure(figsize=(8, 6))
plt.scatter(df['b_y'], df['M_ass'], color=colors, alpha=0.7)

# Etichette e titolo
plt.title('Magnitudine Assoluta (M_ass) vs Colore (b-y)')
plt.xlabel('Colore (b-y)')
plt.ylabel('Magnitudine Assoluta (M_ass)')
plt.gca().invert_yaxis()  # Inverte l'asse y per convenzione astronomica

# Aggiungi una barra dei colori
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=plt.gca())
cbar.set_label('Massa (m_ini)')

# Mostra il grafico
plt.grid(True)
plt.show()