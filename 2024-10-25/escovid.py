import pandas as pd
import matplotlib.pyplot as plt

#file
file_path = "dpc-covid19-ita-regioni.csv"
data = pd.read_csv(file_path)

#scegli regione
regione = "Friuli Venezia Giulia"
data_regione = data.loc[data["denominazione_regione"] == regione].copy()

#converti data in formato datetime 
data_regione["data"] = pd.to_datetime(data_regione["data"])

#seleziona variabili
variabili = ["ricoverati_con_sintomi", "terapia_intensiva", "totale_ospedalizzati", 
                        "isolamento_domiciliare", "totale_positivi"]

#grafico
plt.figure(figsize=(12, 8))
for var in variabili:
    plt.plot(data_regione["data"], data_regione[var], label=var)

#etichette
plt.title(f"Andamento COVID-19 in {regione}")
plt.xlabel("Data")
plt.ylabel("Numero di casi")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()