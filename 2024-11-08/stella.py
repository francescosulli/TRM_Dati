import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import fftconvolve

# Parametri
lum_mean = 1e30         # Luminosità media in erg/s
lum_std = 1e29          # Deviazione standard della luminosità in erg/s
error_std = 1e29        # Deviazione standard dell'errore di misura in erg/s
sample_size = 2000      # Numero di stelle

# Generazione di un campione per la distribuzione della luminosità vera
true_luminosity = np.random.normal(loc=lum_mean, scale=lum_std, size=sample_size)
measurement_error = np.random.normal(loc=0, scale=error_std, size=sample_size)
observed_luminosity = true_luminosity + measurement_error

# Predizione teorica tramite convoluzione
# Definiamo un intervallo di valori per la luminosità
lum_range = np.linspace(lum_mean - 4*lum_std, lum_mean + 4*lum_std, 1000)

# Distribuzione teorica della luminosità vera e dell'errore
true_luminosity_pdf = norm.pdf(lum_range, loc=lum_mean, scale=lum_std)
measurement_error_pdf = norm.pdf(lum_range, loc=lum_mean, scale=error_std)

# Convoluzione delle due distribuzioni
convoluted_pdf = np.convolve(true_luminosity_pdf, measurement_error_pdf, mode='same')
convoluted_pdf /= np.trapz(convoluted_pdf, lum_range)  # Normalizza la distribuzione risultante

# Visualizzazione
plt.hist(observed_luminosity, bins=30, alpha=0.6, color='blue', density=True, label='Luminosità Osservata (Simulazione)')
plt.hist(true_luminosity, bins=30, alpha=0.3, color='red', density=True, label='Luminosità Osservata (Simulazione)')
plt.plot(lum_range, convoluted_pdf, color='orange', label='Predizione Teorica (Convoluzione)')
plt.axvline(x=lum_mean, color='red', linestyle='--', label='Media Vera')
plt.xlabel("Luminosità (erg/s)")
plt.ylabel("Densità di Probabilità")
plt.legend()
plt.title("Distribuzione della Luminosità Osservata con Convoluzione Teorica")
plt.show()