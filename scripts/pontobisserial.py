# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: câmbio (am: 0 = automático, 1 = manual) e consumo (mpg)
am  = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
n = len(mpg)

# 1. Descritivas por grupo
g0, g1 = mpg[am == 0], mpg[am == 1]
print(f"Automático (0): n = {len(g0)}, média = {g0.mean():.4f}, DP = {g0.std(ddof=1):.4f}")
print(f"Manual (1):     n = {len(g1)}, média = {g1.mean():.4f}, DP = {g1.std(ddof=1):.4f}")

# 2. Correlação ponto-bisserial
res = stats.pointbiserialr(am, mpg)
r = res.statistic
t = r * np.sqrt((n - 2) / (1 - r**2))
print(f"\nr_pb = {r:.5f} | t = {t:.4f}, gl = {n - 2}, p-valor = {res.pvalue:.6f}")

# 3. Cálculo pela fórmula
s_n = mpg.std(ddof=0)                                   # desvio-padrão com divisor n
r_formula = (g1.mean() - g0.mean()) / s_n * np.sqrt(len(g1) * len(g0) / n**2)
print(f"Fórmula: ({g1.mean():.4f} - {g0.mean():.4f}) / {s_n:.4f} x raiz({len(g1)} x {len(g0)} / {n}²) = {r_formula:.5f}")

# 4. Equivalências: Pearson com variável 0/1 e teste t de Student (variâncias iguais)
print(f"Pearson(am, mpg) = {stats.pearsonr(am, mpg).statistic:.5f}")
tt = stats.ttest_ind(g1, g0, equal_var=True)
print(f"Teste t de Student: t = {tt.statistic:.4f}, p-valor = {tt.pvalue:.6f}")
tw = stats.ttest_ind(g1, g0, equal_var=False)
print(f"Teste t de Welch:   t = {tw.statistic:.4f}, p-valor = {tw.pvalue:.6f}")

# 5. Intervalo de confiança de 95% (z de Fisher) e r²
z = np.arctanh(r); ep = 1 / np.sqrt(n - 3); zc = stats.norm.ppf(0.975)
ic = np.tanh([z - zc * ep, z + zc * ep])
print(f"IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}] | r² = {r**2:.4f}")

# 6. Pressupostos: normalidade em cada grupo e igualdade de variâncias
for nome, g in [("automático", g0), ("manual", g1)]:
    W, pw = stats.shapiro(g)
    print(f"Shapiro-Wilk ({nome}): W = {W:.4f}, p-valor = {pw:.4f}")
lv = stats.levene(g0, g1, center="median")
print(f"Brown-Forsythe: W = {lv.statistic:.4f}, p-valor = {lv.pvalue:.4f}")

# 7. Exemplo do texto (n = 6)
x = np.array([0, 0, 0, 1, 1, 1]); y = np.array([2, 4, 6, 5, 7, 9.])
r6 = stats.pointbiserialr(x, y)
print(f"\nn = 6: r_pb = {r6.statistic:.4f}, p-valor = {r6.pvalue:.4f}")

# 8. Gráfico
plt.boxplot([g0, g1]); plt.xticks([1, 2], ["Automático (0)", "Manual (1)"]); plt.ylabel("Consumo (mpg)"); plt.show()
