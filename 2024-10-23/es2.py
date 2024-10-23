import pandas as pd
import matplotlib.pyplot as plt


file_path = 'Nemo_6670.dat'

try:
    # Carica il file e ignora la riga di commento
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

# Controlla dati
if df.empty:
    print("Errore: Il DataFrame è vuoto. Controlla il file.")
    exit()

# Mostra le prime righe del DataFrame per debug
print("Prime righe del DataFrame:")
print(df.head())

# Controlla che le colonne 'M_ass' e 'b_y' esistano
if 'M_ass' not in df.columns or 'b_y' not in df.columns:
    print("Errore: Il file non contiene le colonne 'M_ass' o 'b_y'.")
    print("Colonne disponibili:", df.columns)
    exit()

#grafico scatter
plt.figure(figsize=(8,6))
plt.scatter(df['b_y'], df['M_ass'], color='blue', alpha=0.7, label='Dati')

# Etichette
plt.title('Magnitudine Assoluta (M_ass) vs Colore (b-y)')
plt.xlabel('Colore (b-y)')
plt.ylabel('Magnitudine Assoluta (M_ass)')
plt.gca().invert_yaxis()  # Inverte l'asse y

#legenda
plt.legend()

# grafico
plt.grid(True)
plt.show()