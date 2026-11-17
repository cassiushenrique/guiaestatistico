# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools

# Dados: 10 provadores avaliaram 4 formulações de um produto, em escala de 0 a 10 (dados fictícios).
# Cada provador avaliou TODAS as formulações: é um delineamento em blocos.
notas = np.array([[4.8, 5.6, 7.2, 5.8], [7.4, 8.2, 8.4, 7.5], [6.5, 6.5, 7.8, 6.0],
                  [7.2, 7.1, 8.0, 7.0], [7.2, 7.4, 9.1, 7.4], [6.0, 7.2, 8.0, 5.5],
                  [7.7, 7.6, 7.4, 7.2], [5.9, 6.1, 7.8, 6.4], [4.9, 5.2, 6.5, 6.4],
                  [5.4, 6.2, 6.9, 6.3]])
n, k = notas.shape                       # n = blocos (provadores), k = tratamentos (formulações)
form = ["A", "B", "C", "D"]
print(f"{n} provadores x {k} formulações")
print("Formulação   média  mediana")
for j, f in enumerate(form):
    print(f"    {f}        {notas[:, j].mean():.3f}   {np.median(notas[:, j]):6.2f}")

# 1. Os postos são calculados DENTRO de cada provador
postos = np.array([stats.rankdata(linha) for linha in notas])
print("\nPostos dentro de cada provador (1 = pior, 4 = melhor):")
for i in range(n):
    print(f"  provador {i+1:>2}: notas {notas[i]} -> postos {postos[i].astype(int) if np.all(postos[i] == postos[i].astype(int)) else postos[i]}")
Rj = postos.sum(axis=0)
print(f"\nSoma dos postos por formulação: {Rj}")
print(f"Posto médio: {np.round(Rj / n, 3)} | esperado sob H0 = {(k+1)/2:.1f}")

# 2. Estatística de Friedman
Q = 12 / (n * k * (k + 1)) * np.sum(Rj**2) - 3 * n * (k + 1)
print(f"\nQ = 12/({n}x{k}x{k+1}) x {np.sum(Rj**2):.1f} - 3x{n}x{k+1} = {Q:.4f}")
# Correção para empates dentro dos blocos (dois provadores deram notas iguais a duas formulações)
soma_emp = sum(np.sum(c**3 - c) for linha in notas for c in [np.unique(linha, return_counts=True)[1]])
correcao = 1 - soma_emp / (n * k * (k**2 - 1))
print(f"Empates dentro dos blocos: correção = 1 - {soma_emp}/({n}x{k}x({k}²-1)) = {correcao:.5f}")
print(f"Q corrigido = {Q/correcao:.4f} | gl = k - 1 = {k-1} | p-valor = {stats.chi2.sf(Q/correcao, k-1):.6f}")
print("scipy:", stats.friedmanchisquare(*notas.T))

# 3. Coeficiente de concordância de Kendall: o quanto os provadores concordam
W = (Q / correcao) / (n * (k - 1))
print(f"\nW de Kendall = Q / [n(k-1)] = {W:.4f}  (0 = nenhuma concordância, 1 = concordância total)")

# 4. Comparações múltiplas: Wilcoxon pareado com ajuste de Bonferroni
pares = list(itertools.combinations(range(k), 2))
print("\nComparações pareadas (Wilcoxon, ajuste de Bonferroni):")
for i, j in pares:
    p = stats.wilcoxon(notas[:, i], notas[:, j]).pvalue
    pa = min(1, p * len(pares))
    print(f"  {form[i]} vs {form[j]}: diferença mediana = {np.median(notas[:, i] - notas[:, j]):>5.2f} "
          f"| p = {p:.4f} | ajustado = {pa:.4f}{' *' if pa < 0.05 else ''}")

# 5. Diferença mínima significativa entre postos médios (Conover)
A = np.sum(postos**2); B = np.sum(Rj**2) / n
t_crit = stats.t.ppf(0.975, (n - 1) * (k - 1))
dms = t_crit * np.sqrt(2 * n * (A - B) / ((n - 1) * (k - 1)))
print(f"\nDiferença mínima entre SOMAS de postos (Conover, 5%) = {dms:.4f}")
for i, j in pares:
    d = abs(Rj[i] - Rj[j])
    print(f"  |R{form[i]} - R{form[j]}| = {d:.1f}{' > DMS: difere' if d > dms else ''}")

# 6. Comparação com a ANOVA de medidas repetidas
media_geral = notas.mean()
SQ_trat = n * np.sum((notas.mean(axis=0) - media_geral)**2)
SQ_bloco = k * np.sum((notas.mean(axis=1) - media_geral)**2)
SQ_res = np.sum((notas - media_geral)**2) - SQ_trat - SQ_bloco
F = (SQ_trat / (k - 1)) / (SQ_res / ((n - 1) * (k - 1)))
print(f"\nANOVA de medidas repetidas: F({k-1}, {(n-1)*(k-1)}) = {F:.4f}, p-valor = {stats.f.sf(F, k-1, (n-1)*(k-1)):.6f}")
print(f"Shapiro-Wilk dos resíduos: p-valor = {stats.shapiro((notas - notas.mean(axis=0) - notas.mean(axis=1)[:, None] + media_geral).ravel()).pvalue:.4f}")

# 7. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(n): ax[0].plot(range(1, k + 1), notas[i], marker="o", color="gray", alpha=0.6)
ax[0].plot(range(1, k + 1), notas.mean(axis=0), marker="o", color="red", lw=3)
ax[0].set_xticks(range(1, k + 1)); ax[0].set_xticklabels(form); ax[0].set_xlabel("Formulação"); ax[0].set_ylabel("Nota")
ax[1].bar(form, Rj / n, color="lightgray", edgecolor="gray")
ax[1].axhline((k + 1) / 2, color="red", ls="--"); ax[1].set_ylabel("Posto médio"); ax[1].set_xlabel("Formulação")
plt.show()
