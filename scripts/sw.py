# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars (32 automóveis); disp = cilindrada em pol³
mpg  = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
cyl  = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
disp = np.array([160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1, 318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121])
x = disp
n = len(x)

# 1. Estatísticas descritivas
q = np.percentile(x, [0, 25, 50, 75, 100])
print("Mín, Q1, Mediana, Q3, Máx:", np.round(q, 2))
print("Média:", round(x.mean(), 2), "| DP:", round(x.std(ddof=1), 4), "| n:", n)

# 2. Diagnóstico gráfico (sempre antes do teste)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].hist(x, bins=8, density=True, color="lightgray", edgecolor="white")
t = np.linspace(x.min(), x.max(), 200)
ax[0].plot(t, stats.norm.pdf(t, x.mean(), x.std(ddof=1)), color="red")
ax[0].set_title("Histograma")
stats.probplot(x, dist="norm", plot=ax[1]); ax[1].set_title("Gráfico Q-Q")
ax[2].boxplot(x); ax[2].set_title("Boxplot")
plt.show()

# 3. Teste de Shapiro–Wilk
W, p = stats.shapiro(x)
print(f"\nShapiro-Wilk: W = {W:.5f}, p-valor = {p:.5f}")
print("Evidência contra normalidade" if p < 0.05 else "Sem evidência suficiente contra normalidade")

# 4. Transformação logarítmica
W2, p2 = stats.shapiro(np.log(x))
print(f"log(x):       W = {W2:.5f}, p-valor = {p2:.5f}")

# 5. Assimetria populacional e amostral
m = x.mean()
assim = np.sum((x - m)**3) / n / (np.sum((x - m)**2) / n)**1.5
print("Assimetria (populacional):", round(assim, 4), "| (amostral):", round(stats.skew(x, bias=False), 4))

# 6. Exemplo manual com n = 6 (coeficientes a_i tabulados por Shapiro e Wilk)
y = np.array([2.1, 2.8, 3.0, 3.4, 3.9, 4.0])
a = np.array([0.6431, 0.2806, 0.0875])
b = sum(a[i] * (y[5 - i] - y[i]) for i in range(3))
S2 = np.sum((y - y.mean())**2)
print(f"\nManual: b = {b:.4f}, b² = {b**2:.4f}, S² = {S2:.2f}, W ≈ {b**2 / S2:.4f}")
Wy, py = stats.shapiro(y)
print(f"scipy:  W = {Wy:.4f}, p-valor = {py:.4f}")

# 7. Valor-p pela aproximação de Royston (n >= 12), para conferência
ln = np.log(n)
mu = 0.0038915*ln**3 - 0.083751*ln**2 - 0.31082*ln - 1.5861
sg = np.exp(0.0030302*ln**2 - 0.082676*ln - 0.4803)
z = (np.log(1 - W) - mu) / sg
print(f"\nRoyston: mu = {mu:.4f}, sigma = {sg:.4f}, Z = {z:.3f}, p = {1 - stats.norm.cdf(z):.4f}")
