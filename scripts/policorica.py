# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

def phi2(x, y, r):
    """Função de distribuição acumulada da normal bivariada padrão com correlação r."""
    x, y = np.clip([x, y], -8, 8)
    return stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([x, y])

def policorica(tab):
    """Correlação policórica em duas etapas: limiares pelas marginais, r por máxima verossimilhança."""
    tab = np.asarray(tab, dtype=float); n = tab.sum()
    a = np.r_[-np.inf, stats.norm.ppf(np.cumsum(tab.sum(axis=1))[:-1] / n), np.inf]   # limiares das linhas
    b = np.r_[-np.inf, stats.norm.ppf(np.cumsum(tab.sum(axis=0))[:-1] / n), np.inf]   # limiares das colunas
    def probs(r):
        F = np.array([[phi2(ai, bj, r) if np.isfinite(ai) or ai > 0 else 0.0 for bj in b] for ai in a])
        F[0, :] = 0; F[:, 0] = 0
        return F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    menos_logv = lambda r: -np.sum(tab * np.log(np.clip(probs(r), 1e-12, None)))
    res = optimize.minimize_scalar(menos_logv, bounds=(-0.99, 0.99), method="bounded", options={"xatol": 1e-6})
    return res.x, a[1:-1], b[1:-1], probs(res.x) * n

# 1. Dados: 250 clientes; satisfação com o produto (linhas) e avaliação do atendimento (colunas)
#                  atendimento: ruim  regular  bom
tab = np.array([[40, 25, 10],      # satisfação baixa
                [20, 45, 30],      # satisfação média
                [ 5, 25, 50]])     # satisfação alta
n = tab.sum()
print("Totais das linhas:", tab.sum(axis=1), "| das colunas:", tab.sum(axis=0), "| n =", n)

# 2. Correlação policórica
r_p, lim_lin, lim_col, esperado = policorica(tab)
print("Limiares das linhas:", np.round(lim_lin, 4), "| das colunas:", np.round(lim_col, 4))
print(f"Correlação policórica = {r_p:.4f}")
print("Frequências esperadas pelo modelo:\n", np.round(esperado, 1))

# 3. Qualidade do ajuste do modelo normal bivariado latente
G2 = 2 * np.sum(tab * np.log(tab / esperado)); gl = tab.size - 1 - (tab.shape[0] - 1) - (tab.shape[1] - 1) - 1
print(f"Razão de verossimilhanças G² = {G2:.3f}, gl = {gl}, p-valor = {stats.chi2.sf(G2, gl):.4f}")

# 4. Comparação com coeficientes calculados sobre os códigos 1, 2, 3
x = np.repeat([1, 2, 3], tab.sum(axis=1))
y = np.concatenate([np.repeat([1, 2, 3], linha) for linha in tab])
print(f"Pearson dos códigos = {stats.pearsonr(x, y).statistic:.4f} | Spearman = {stats.spearmanr(x, y)[0]:.4f} | tau-b de Kendall = {stats.kendalltau(x, y).statistic:.4f}")

# 5. Teste de associação (qui-quadrado de independência)
qui2, p, glq, _ = stats.chi2_contingency(tab)
print(f"Qui-quadrado = {qui2:.2f}, gl = {glq}, p-valor = {p:.2e}")

# 6. Intervalo de confiança de 95% por bootstrap (reamostragem da tabela)
rng = np.random.default_rng(2026)
boot = [policorica(rng.multinomial(n, tab.ravel() / n).reshape(3, 3))[0] for _ in range(300)]
print("Erro-padrão bootstrap:", round(float(np.std(boot, ddof=1)), 4), "| IC 95%:", np.round(np.percentile(boot, [2.5, 97.5]), 3))

# 7. Gráfico: tabela observada
plt.imshow(tab, cmap="Greens")
for i in range(3):
    for j in range(3): plt.text(j, i, str(tab[i, j]), ha="center", va="center", fontsize=14)
plt.xticks([0, 1, 2], ["ruim", "regular", "bom"]); plt.yticks([0, 1, 2], ["baixa", "média", "alta"])
plt.xlabel("Atendimento"); plt.ylabel("Satisfação"); plt.show()
