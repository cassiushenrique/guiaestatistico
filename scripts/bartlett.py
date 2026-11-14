# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mpg  = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
cyl  = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
disp = np.array([160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1, 318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121])

# 1. Variâncias por grupo
g = [mpg[cyl == c] for c in (4, 6, 8)]
ni = np.array([len(v) for v in g]); vi = np.array([v.var(ddof=1) for v in g])
print("Variâncias (4, 6, 8 cilindros):", np.round(vi, 4))

# 2. Normalidade dentro de cada grupo
for c, v in zip((4, 6, 8), g):
    W, p = stats.shapiro(v)
    print(f"Shapiro-Wilk ({c} cil.): W = {W:.4f}, p-valor = {p:.4f}")

# 3. Teste de Bartlett
res = stats.bartlett(*g)
print(f"\nBartlett: K² = {res.statistic:.4f}, gl = 2, p-valor = {res.pvalue:.5f}")

# 4. Cálculo manual da estatística
N = ni.sum(); k = len(ni)
sp2 = np.sum((ni - 1) * vi) / (N - k)                       # variância combinada
num = (N - k) * np.log(sp2) - np.sum((ni - 1) * np.log(vi))
den = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / (ni - 1)) - 1 / (N - k))
K2 = num / den
print(f"sp² = {sp2:.4f} | numerador = {num:.4f} | denominador = {den:.4f} | K² = {K2:.4f} | p = {stats.chi2.sf(K2, k - 1):.5f}")

# 5. Gráfico das variâncias
plt.bar(["4", "6", "8"], vi, color="lightgray", edgecolor="gray")
plt.axhline(sp2, color="red", linestyle="--"); plt.ylabel("Variância"); plt.xlabel("Cilindros"); plt.show()
