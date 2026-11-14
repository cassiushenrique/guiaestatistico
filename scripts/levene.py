# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mpg  = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
cyl  = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
disp = np.array([160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1, 318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121])

# 1. Grupos: consumo por número de cilindros
grupos = {c: mpg[cyl == c] for c in (4, 6, 8)}
print("cil   n   média  mediana     DP")
for c, g in grupos.items():
    print(f"{c:>3} {len(g):>3} {g.mean():7.3f} {np.median(g):8.3f} {g.std(ddof=1):7.3f}")

# 2. Diagnóstico gráfico
plt.boxplot(list(grupos.values())); plt.xticks([1, 2, 3], ["4", "6", "8"])
plt.xlabel("Cilindros"); plt.ylabel("mpg"); plt.show()

# 3. Brown–Forsythe (centro = mediana) e Levene clássico (centro = média)
g = list(grupos.values())
bf = stats.levene(*g, center="median")
lv = stats.levene(*g, center="mean")
print(f"\nBrown-Forsythe (mediana): W = {bf.statistic:.4f}, p-valor = {bf.pvalue:.5f}")
print(f"Levene (média):           W = {lv.statistic:.4f}, p-valor = {lv.pvalue:.5f}")

# 4. Cálculo manual: ANOVA sobre os desvios absolutos em torno da mediana
Z = [np.abs(v - np.median(v)) for v in g]
print("Médias de Z por grupo:", [round(float(z.mean()), 3) for z in Z], "| média geral:", round(float(np.concatenate(Z).mean()), 3))
F, pF = stats.f_oneway(*Z)
print(f"ANOVA sobre Z: F = {F:.4f}, p-valor = {pF:.5f}")

# 5. Bartlett para comparação (sensível a não normalidade)
bt = stats.bartlett(*g)
print(f"Bartlett: K² = {bt.statistic:.4f}, p-valor = {bt.pvalue:.5f}")
