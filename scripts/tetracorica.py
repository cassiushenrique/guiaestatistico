# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

def phi2(x, y, r):
    """Função de distribuição acumulada da normal bivariada padrão com correlação r."""
    x, y = np.clip([x, y], -8, 8)
    return stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([x, y])

def tetracorica(a, b, c, d):
    """Tabela [[a, b], [c, d]]: a = ambos positivos, d = ambos negativos."""
    n = a + b + c + d
    h = stats.norm.ppf(1 - (a + b) / n)          # limiar da variável das linhas
    k = stats.norm.ppf(1 - (a + c) / n)          # limiar da variável das colunas
    f = lambda r: phi2(-h, -k, r) - a / n        # P(X > h, Y > k) deve igualar a proporção observada
    return optimize.brentq(f, -0.999, 0.999, xtol=1e-7), h, k

# 1. Dados: 200 alunos, aprovação (nota >= 60) em duas provas
#                 Prova 2: aprovado  reprovado
a, b = 70, 30   # Prova 1 aprovado
c, d = 20, 80   # Prova 1 reprovado
n = a + b + c + d
print(f"n = {n} | aprovados na Prova 1: {(a + b) / n:.2f} | aprovados na Prova 2: {(a + c) / n:.2f}")

# 2. Correlação tetracórica
r_t, h, k = tetracorica(a, b, c, d)
print(f"Limiares: h = {h:.4f}, k = {k:.4f}")
print(f"Correlação tetracórica = {r_t:.4f}")

# 3. Comparação com o coeficiente phi (Pearson das variáveis 0/1) e medidas da tabela
phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
rc = (a * d) / (b * c)
print(f"Phi = {phi:.4f} | razão de chances = {rc:.3f}")
print(f"Aproximação do cosseno: cos(pi / (1 + raiz(ad/bc))) = {np.cos(np.pi / (1 + np.sqrt(rc))):.4f}")

# 4. Teste de associação na tabela (qui-quadrado sem correção)
qui2, p, gl, esp = stats.chi2_contingency([[a, b], [c, d]], correction=False)
print(f"Qui-quadrado = {qui2:.3f}, gl = {gl}, p-valor = {p:.2e} (note que qui² = n x phi² = {n * phi**2:.3f})")

# 5. Intervalo de confiança de 95% por bootstrap (reamostragem da tabela)
rng = np.random.default_rng(2026)
boot = []
for _ in range(1000):
    ab, bb, cb, db = rng.multinomial(n, np.array([a, b, c, d]) / n)
    boot.append(tetracorica(ab, bb, cb, db)[0])
print("Erro-padrão bootstrap:", round(float(np.std(boot, ddof=1)), 4), "| IC 95%:", np.round(np.percentile(boot, [2.5, 97.5]), 3))

# 6. O efeito do ponto de corte: mesma correlação latente (0,70), cortes diferentes
for corte in [0.0, 1.0, 1.5]:
    p11 = phi2(-corte, -corte, 0.70); p1 = 1 - stats.norm.cdf(corte)
    print(f"corte = {corte:.1f}: proporção acima = {p1:.3f} | phi teórico = {(p11 - p1**2) / (p1 * (1 - p1)):.3f} | tetracórica = 0.700")

# 7. Gráfico: tabela observada
plt.imshow([[a, b], [c, d]], cmap="Greens")
for i, linha in enumerate([[a, b], [c, d]]):
    for j, v in enumerate(linha): plt.text(j, i, str(v), ha="center", va="center", fontsize=16)
plt.xticks([0, 1], ["P2 aprovado", "P2 reprovado"]); plt.yticks([0, 1], ["P1 aprovado", "P1 reprovado"]); plt.show()
