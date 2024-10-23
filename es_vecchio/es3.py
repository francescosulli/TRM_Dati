# Creazione del dizionario "lista della spesa"
lista_della_spesa = {
    "Mele": 5,
    "Banane": 7,
    "Pane": 2,
    "Latte": 1,
    "Uova": 12,
    "Pasta": 3
}

# Iterazione sulle coppie chiave/valore e stampa
for prodotto, quantita in lista_della_spesa.items():
    print(f"{prodotto}: {quantita}")