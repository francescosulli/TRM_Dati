import pandas as pd
import matplotlib.pyplot as plt

# Specifica il percorso al file .dat
file_path = 'Nemo_6670.dat'

# Carica il file con pandas utilizzando il separatore appropriato (spazi multipli)
# Il file contiene un'intestazione che inizia con '#', quindi saltiamo la prima riga
try:
    # Salta la riga di commento che inizia con '#'
    df = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None, 
                     names=["MsuH", "m_ini", "logL", "logTe", "M_ass", "b_ass", "y_ass", 
                            "m_app", "b_y", "dist", "abs_dist", "ID_parent", "age_parent"])
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
    exit()

# Controlla che le colonne 'M_ass' e 'b_y' esistano nel DataFrame
if 'M_ass' not in df.columns or 'b_y' not in df.columns:
    print("Errore: Il file non contiene le colonne 'M_ass' o 'b_y'.")
    exit()

# Crea il grafico scatter
plt.figure(figsize=(8,6))
plt.scatter(df['b_y'], df['M_ass'], color='blue', alpha=0.7, label='Dati')

# Etichette e titolo
plt.title('Magnitudine Assoluta (M_ass) vs Colore (b-y)')
plt.xlabel('Colore (b-y)')
plt.ylabel('Magnitudine Assoluta (M_ass)')
plt.gca().invert_yaxis()  # Inverte l'asse y per convenzione astronomica

# Mostra legenda
plt.legend()

# Mostra il grafico
plt.grid(True)
plt.show()