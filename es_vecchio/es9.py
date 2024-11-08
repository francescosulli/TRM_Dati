import matplotlib.pyplot as plt

# Dati ipotetici
anni_precisi = list(range(1920, 1975, 5))  # Intervalli di 5 anni
costi_medi_auto = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0]  # Costo medio in migliaia di dollari
redditi_medi = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]  # Reddito medio in migliaia di dollari

# Calcolo del rapporto costo medio auto / reddito medio
accessibilita_auto = [costo / reddito for costo, reddito in zip(costi_medi_auto, redditi_medi)]

# Creazione del grafico
plt.figure(figsize=(12, 6))
plt.plot(anni_precisi, accessibilita_auto, color='green', marker='o', linestyle='-', label="Accessibilità auto (prezzo/reddito)")
plt.title("Accessibilità delle automobili rispetto al reddito medio (1920-1970)")
plt.xlabel("Anno")
plt.ylabel("Rapporto Costo Auto / Reddito Medio")
plt.xticks(anni_precisi, rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

plt.show()