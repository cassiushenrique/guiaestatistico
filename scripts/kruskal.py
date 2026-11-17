# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools

# Dados: conjunto InsectSprays (contagem de insetos em 72 parcelas, 6 inseticidas)
contagem = np.array([10,7,20,14,14,12,10,23,17,20,14,13,  11,17,21,11,16,14,17,17,19,21,7,13,
                     0,1,7,2,3,1,2,1,3,0,1,4,            3,5,12,6,4,3,5,5,5,5,2,4,
                     3,5,3,5,3,6,1,1,3,2,6,4,            11,9,15,22,15,16,13,10,26,26,24,13.])
spray = np.repeat(list("ABCDEF"), 12)
grupos = [contagem[spray == s] for s in "ABCDEF"]
N = len(contagem); k = len(grupos)
print("spray   n   média  mediana   DP   variância")
for s, g in zip("ABCDEF", grupos):
    print(f"  {s}    {len(g)}  {g.mean():6.3f}  {np.median(g):7.1f}  {g.std(ddof=1):5.2f}  {g.var(ddof=1):8.2f}")

# 1. Por que não usar a ANOVA: variância cresce com a média
lev = stats.levene(*grupos, center="median")
print(f"\nBrown-Forsythe: W = {lev.statistic:.4f}, p-valor = {lev.pvalue:.5f}")
print("Razão entre a maior e a menor variância:", round(max(g.var(ddof=1) for g in grupos) / min(g.var(ddof=1) for g in grupos), 2))
for s, g in zip("ABCDEF", grupos):
    print(f"  Shapiro-Wilk (spray {s}): p-valor = {stats.shapiro(g).pvalue:.4f}")

# 2. Postos de todas as 72 observações
postos = stats.rankdata(contagem)
Ri = np.array([postos[spray == s].sum() for s in "ABCDEF"])
ni = np.array([len(g) for g in grupos])
print(f"\nSoma dos postos por spray: {Ri}")
print(f"Posto médio por spray: {np.round(Ri / ni, 3)} | posto médio geral = {(N+1)/2:.1f}")
print(f"Soma total = {Ri.sum():.1f} = N(N+1)/2 = {N*(N+1)/2:.1f}")

# 3. Estatística H
H = 12 / (N * (N + 1)) * np.sum(Ri**2 / ni) - 3 * (N + 1)
emp = np.unique(contagem, return_counts=True)[1]
correcao = 1 - np.sum(emp**3 - emp) / (N**3 - N)
print(f"\nH (sem correção) = 12/({N}x{N+1}) x {np.sum(Ri**2/ni):.2f} - 3x{N+1} = {H:.4f}")
print(f"Correção para empates: 1 - {np.sum(emp**3-emp)}/({N}³-{N}) = {correcao:.5f}")
print(f"H corrigido = {H/correcao:.4f} | gl = {k-1} | p-valor = {stats.chi2.sf(H/correcao, k-1):.3e}")
print("scipy:", stats.kruskal(*grupos))

# 4. Comparação com a ANOVA clássica
f = stats.f_oneway(*grupos)
print(f"\nANOVA de um fator: F = {f.statistic:.4f}, p-valor = {f.pvalue:.3e}")
w = np.array([len(g) / g.var(ddof=1) for g in grupos]); mw = np.sum(w * np.array([g.mean() for g in grupos])) / w.sum()
A_ = np.sum(w * (np.array([g.mean() for g in grupos]) - mw)**2) / (k - 1)
B_ = 2 * (k - 2) / (k**2 - 1) * np.sum((1 - w / w.sum())**2 / (ni - 1))
Fw = A_ / (1 + B_); glw = (k**2 - 1) / (3 * np.sum((1 - w / w.sum())**2 / (ni - 1)))
print(f"ANOVA de Welch: F = {Fw:.4f}, gl = ({k-1}; {glw:.3f}), p-valor = {stats.f.sf(Fw, k-1, glw):.3e}")

# 5. Comparações múltiplas de Dunn, com ajuste de Bonferroni
print("\nTeste de Dunn (valores-p ajustados por Bonferroni):")
sigma = np.sqrt((N * (N + 1) / 12 - np.sum(emp**3 - emp) / (12 * (N - 1))))
pares = list(itertools.combinations(range(k), 2))
for i, j in pares:
    z = (Ri[i]/ni[i] - Ri[j]/ni[j]) / (sigma * np.sqrt(1/ni[i] + 1/ni[j]))
    p = min(1, 2 * stats.norm.sf(abs(z)) * len(pares))
    marca = " *" if p < 0.05 else ""
    print(f"  {'ABCDEF'[i]} vs {'ABCDEF'[j]}: z = {z:>7.3f} | p ajustado = {p:.5f}{marca}")

# 6. Tamanho do efeito
eta2 = (H / correcao - k + 1) / (N - k)
print(f"\nEpsilon² = H / [(N²-1)/(N+1)] = {(H/correcao)/((N**2-1)/(N+1)):.4f}")
print(f"Eta² baseado em H = (H - k + 1)/(N - k) = {eta2:.4f}")

# 7. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].boxplot(grupos); ax[0].set_xticks(range(1, 7)); ax[0].set_xticklabels(list("ABCDEF"))
for j, g in enumerate(grupos, 1): ax[0].scatter(np.full(len(g), j), g, alpha=0.5)
ax[0].set_xlabel("Inseticida"); ax[0].set_ylabel("Contagem de insetos")
ax[1].bar(list("ABCDEF"), Ri / ni, color="lightgray", edgecolor="gray")
ax[1].axhline((N + 1) / 2, color="red", ls="--"); ax[1].set_ylabel("Posto médio"); ax[1].set_xlabel("Inseticida")
plt.show()
