# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: peso (wt, em 1000 libras) e consumo (mpg)
wt  = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78])
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
n = len(wt)

# 1. Gráfico de dispersão (sempre antes do coeficiente)
plt.scatter(wt, mpg); b, a = np.polyfit(wt, mpg, 1)
plt.plot(np.sort(wt), a + b * np.sort(wt), color="red")
plt.xlabel("Peso (1000 lb)"); plt.ylabel("Consumo (mpg)"); plt.show()

# 2. Correlação de Pearson e teste de hipóteses
res = stats.pearsonr(wt, mpg)
r = res.statistic
t = r * np.sqrt((n - 2) / (1 - r**2))
print(f"r = {r:.5f} | t = {t:.4f}, gl = {n - 2}, p-valor = {res.pvalue:.3e}")

# 3. Intervalo de confiança de 95% (transformação z de Fisher)
z = np.arctanh(r); ep = 1 / np.sqrt(n - 3); zc = stats.norm.ppf(0.975)
ic = np.tanh([z - zc * ep, z + zc * ep])
print(f"z de Fisher = {z:.4f} | erro-padrão = {ep:.4f} | IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}]")
print(f"r² = {r**2:.4f}")

# 4. Cálculo pela definição
sxy = np.sum((wt - wt.mean()) * (mpg - mpg.mean()))
sxx = np.sum((wt - wt.mean())**2); syy = np.sum((mpg - mpg.mean())**2)
print(f"Sxy = {sxy:.4f} | Sxx = {sxx:.4f} | Syy = {syy:.4f} | r = {sxy / np.sqrt(sxx * syy):.5f}")

# 5. Normalidade das duas variáveis (para o teste e o IC)
for nome, v in [("wt", wt), ("mpg", mpg)]:
    W, pw = stats.shapiro(v)
    print(f"Shapiro-Wilk ({nome}): W = {W:.4f}, p-valor = {pw:.4f}")

# 6. Exemplo do texto (n = 5)
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 4, 5, 4, 5.])
r5 = stats.pearsonr(x, y)
print(f"\nn = 5: r = {r5.statistic:.4f}, t = {r5.statistic*np.sqrt(3/(1-r5.statistic**2)):.4f}, p-valor = {r5.pvalue:.4f}")
