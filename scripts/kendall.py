# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: potência (hp) e consumo (mpg)
hp  = np.array([110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109])
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
n = len(hp)

def pares(x, y):
    """Conta pares concordantes, discordantes e empatados."""
    C = D = Tx = Ty = Txy = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0: Txy += 1
            elif dx == 0: Tx += 1
            elif dy == 0: Ty += 1
            elif dx == dy: C += 1
            else: D += 1
    return C, D, Tx, Ty, Txy

# 1. Contagem dos pares
C, D, Tx, Ty, Txy = pares(hp, mpg)
total = n * (n - 1) // 2
print(f"Pares: total = {total} | concordantes = {C} | discordantes = {D} | empates só em hp = {Tx} | só em mpg = {Ty} | em ambos = {Txy}")

# 2. tau-a (ignora empates) e tau-b (corrige empates)
tau_a = (C - D) / total
tau_b_manual = (C - D) / np.sqrt((C + D + Tx) * (C + D + Ty))
print(f"tau-a = {tau_a:.5f} | tau-b (manual) = {tau_b_manual:.5f}")

# 3. Teste de hipóteses (tau-b, aproximação normal com correção de empates)
res = stats.kendalltau(hp, mpg)
print(f"tau-b = {res.statistic:.5f}, z = {stats.norm.isf(res.pvalue / 2) * np.sign(res.statistic):.4f}, p-valor = {res.pvalue:.3e}")

# 4. Comparação com Spearman e Pearson
print(f"Spearman = {stats.spearmanr(hp, mpg)[0]:.5f} | Pearson = {stats.pearsonr(hp, mpg).statistic:.5f}")

# 5. Intervalo de confiança de 95% por bootstrap (percentil)
rng = np.random.default_rng(2026)
boot = []
for _ in range(5000):
    i = rng.integers(0, n, n)
    boot.append(stats.kendalltau(hp[i], mpg[i]).statistic)
print("IC 95% bootstrap:", np.round(np.percentile(boot, [2.5, 97.5]), 4))

# 6. Exemplo do texto (n = 5, sem empates)
x = np.array([1, 2, 3, 4, 5]); y = np.array([2, 1, 4, 3, 5])
C5, D5, *_ = pares(x, y)
r5 = stats.kendalltau(x, y)
print(f"\nn = 5: C = {C5}, D = {D5}, tau = {r5.statistic:.2f}, p-valor exato = {r5.pvalue:.4f}")

# 7. Gráfico de dispersão
plt.scatter(hp, mpg); plt.xlabel("Potência (hp)"); plt.ylabel("Consumo (mpg)"); plt.show()
