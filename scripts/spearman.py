# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: potência (hp) e consumo (mpg)
hp  = np.array([110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109])
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
n = len(hp)

# 1. Dispersão dos dados e dos postos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(hp, mpg); ax[0].set_xlabel("Potência (hp)"); ax[0].set_ylabel("Consumo (mpg)")
ax[1].scatter(stats.rankdata(hp), stats.rankdata(mpg), color="darkgreen")
ax[1].set_xlabel("Posto da potência"); ax[1].set_ylabel("Posto do consumo")
plt.show()

# 2. Correlação de Spearman
rho, p = stats.spearmanr(hp, mpg)
t = rho * np.sqrt((n - 2) / (1 - rho**2))
print(f"rho = {rho:.5f} | t = {t:.4f}, gl = {n - 2}, p-valor = {p:.3e}")

# 3. Spearman é o Pearson calculado sobre os postos (postos médios nos empates)
rx, ry = stats.rankdata(hp), stats.rankdata(mpg)
print(f"Pearson dos postos = {stats.pearsonr(rx, ry).statistic:.5f}")
d2 = np.sum((rx - ry)**2)
print(f"Soma de d² = {d2:.1f} | fórmula simplificada 1 - 6Σd²/(n(n²-1)) = {1 - 6*d2/(n*(n**2 - 1)):.5f} (aproximada, pois há empates)")
print("Valores repetidos: hp =", n - len(np.unique(hp)), "| mpg =", n - len(np.unique(mpg)))

# 4. Comparação com Pearson
r = stats.pearsonr(hp, mpg).statistic
print(f"Pearson (dados originais) = {r:.5f}")

# 5. Intervalo de confiança de 95% por bootstrap (percentil)
rng = np.random.default_rng(2026)
boot = []
for _ in range(5000):
    i = rng.integers(0, n, n)
    boot.append(stats.spearmanr(hp[i], mpg[i])[0])
print("IC 95% bootstrap:", np.round(np.percentile(boot, [2.5, 97.5]), 4))

# 6. Exemplo do texto (n = 6, sem empates)
x = np.array([10, 20, 30, 40, 50, 60.]); y = np.array([1.2, 0.9, 1.5, 2.8, 2.1, 9.5])
r6, p6 = stats.spearmanr(x, y)
print(f"\nn = 6: postos de y = {stats.rankdata(y).astype(int).tolist()}, Σd² = {np.sum((stats.rankdata(x)-stats.rankdata(y))**2):.0f}, rho = {r6:.4f}, Pearson = {stats.pearsonr(x, y).statistic:.4f}")
