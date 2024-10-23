# Creazione della lista dei 30 numeri pari
pari = [x for x in range(0, 60, 2)]  # Genera numeri pari da 0 a 58

# Creazione della lista dei 40 numeri dispari
dispari = [x for x in range(1, 80, 2)]  # Genera numeri dispari da 1 a 79

# Unione delle due liste
lista_unione = pari + dispari

# Inizializzazione del contatore per il ciclo while
i = 0

# Ciclo while per scorrere la lista finché ci sono numeri pari
while i < len(lista_unione) and lista_unione[i] % 2 == 0:
    print(f"{lista_unione[i]} è pari")
    i += 1