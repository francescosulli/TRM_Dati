# Lista degli elementi
numeri = [2, 3, 4, 5, 12, 13, 14, 15, 0, 1, 22, 23, 24, 25, 32, 30, 34, 35]

# Iterazione a partire dal terzo elemento (indice 2)
for i in range(2, len(numeri)):
    elemento = numeri[i]

    if 3 < elemento < 20:
        # Somma dei due elementi precedenti
        risultato = numeri[i-2] + numeri[i-1]
    elif 20 < elemento < 33:
        # Somma dei due elementi successivi (attenzione agli ultimi elementi della lista)
        if i+2 < len(numeri):
            risultato = numeri[i+1] + numeri[i+2]
        else:
            # Se non ci sono due elementi successivi, somma solo i successivi disponibili
            risultato = numeri[i+1] if i+1 < len(numeri) else elemento
    else:
        # L'elemento stesso viene restituito
        risultato = elemento

    print(f"Elemento {i} ({elemento}): {risultato}")
