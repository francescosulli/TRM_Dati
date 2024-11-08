import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#parametri
media_altezza_vera = 175  #media della distribuzione
dev_std_altezza_vera = 6  #deviazione std

#parametro dell'errore di misura
dev_std_errore = 2

#generazione
num_campioni = 1000000  #N individui
altezza_vera = np.random.normal(media_altezza_vera, dev_std_altezza_vera, num_campioni)
errore_misurazione = np.random.normal(0, dev_std_errore, num_campioni)
altezza_misurata = altezza_vera + errore_misurazione

#calcolo distribuzione teorica delle altezze osservate
media_altezza_misurata = media_altezza_vera  #media invariata
dev_std_altezza_misurata = np.sqrt(dev_std_altezza_vera**2 + dev_std_errore**2)  #propagazione dell'errore

#grafico
x = np.linspace(155, 195, 500)
pdf_altezza_misurata = norm.pdf(x, media_altezza_misurata, dev_std_altezza_misurata)

plt.hist(altezza_misurata, bins=30, density=True, alpha=0.6, color='skyblue', label="Campione Altezza Misurata")
plt.plot(x, pdf_altezza_misurata, 'r', linewidth=2, label="Distribuzione Teorica")
plt.xlabel("Altezza (cm)")
plt.ylabel("Densità di Probabilità")
plt.legend()
plt.title("Distribuzione delle Altezze Misurate")
plt.show()