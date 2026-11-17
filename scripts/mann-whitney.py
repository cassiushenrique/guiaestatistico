# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: tempo até o reparo (horas) registrado por duas equipes de manutenção (dados fictícios).
# A distribuição é assimétrica à direita, como é comum em tempos de atendimento.
equipeA = np.array([1.6, 4.1, 2.8, 3.4, 5.2, 2.2, 2.4, 3.0, 1.2, 3.2, 1.8, 4.5, 3.7, 2.6, 2.0, 6.6])
equipeB = np.array([60.0, 2.6, 2.0, 7.4, 4.5, 3.0, 4.2, 8.5, 6.7, 4.9, 3.9, 3.3, 3.6, 6.1, 5.6, 5.2])
n1, n2 = len(equipeA), len(equipeB)
for nome, g in [("equipe A", equipeA), ("equipe B", equipeB)]:
    print(f"{nome}: n = {len(g)} | média = {g.mean():.3f} | mediana = {np.median(g):.2f} "
          f"| DP = {g.std(ddof=1):.3f} | máximo = {g.max()}")

# 1. Por que não usar o teste t aqui
for nome, g in [("equipe A", equipeA), ("equipe B", equipeB)]:
    print(f"Shapiro-Wilk ({nome}): p-valor = {stats.shapiro(g).pvalue:.6f}")
tw = stats.ttest_ind(equipeA, equipeB, equal_var=False)
print(f"Teste t de Welch: t = {tw.statistic:.4f}, p-valor = {tw.pvalue:.4f}  (não detecta diferença)")

# 2. Postos: a base do método
todos = np.concatenate([equipeA, equipeB])
postos = stats.rankdata(todos)
R1 = postos[:n1].sum(); R2 = postos[n1:].sum()
print(f"\nSoma dos postos: equipe A = {R1:.1f} | equipe B = {R2:.1f} | total = {R1 + R2:.1f} = N(N+1)/2 = {(n1+n2)*(n1+n2+1)/2:.1f}")
print("Maior tempo da equipe B (60,0 h) recebe o posto", int(postos[n1]), "- e não mais que isso: é só o maior.")

# 3. Estatística U
U1 = R1 - n1 * (n1 + 1) / 2
U2 = R2 - n2 * (n2 + 1) / 2
print(f"\nU1 = {R1:.1f} - {n1}x{n1+1}/2 = {U1:.1f} | U2 = {U2:.1f} | U1 + U2 = {U1+U2:.1f} = n1 x n2 = {n1*n2}")

# 4. Teste (o scipy devolve U da primeira amostra)
res = stats.mannwhitneyu(equipeA, equipeB, alternative="two-sided")
print(f"\nMann-Whitney: U = {res.statistic:.1f}, p-valor = {res.pvalue:.5f}")

# 5. Aproximação normal, com correção de continuidade e de empates
N = n1 + n2
mu_U = n1 * n2 / 2
emp = np.unique(todos, return_counts=True)[1]
correcao = np.sum(emp**3 - emp)
var_U = n1 * n2 / 12 * ((N + 1) - correcao / (N * (N - 1)))
z = (U1 - mu_U + 0.5) / np.sqrt(var_U) if U1 < mu_U else (U1 - mu_U - 0.5) / np.sqrt(var_U)
print(f"Aproximação normal: E(U) = {mu_U:.1f} | Var(U) = {var_U:.4f} | z = {z:.4f} | p-valor = {2*stats.norm.cdf(-abs(z)):.5f}")
print(f"Empates: {int(np.sum(emp > 1))} valores repetidos")

# 6. Tamanho do efeito: probabilidade de superioridade e delta de Cliff
PS = U2 / (n1 * n2)          # P(uma observação de B ser maior que uma de A)
print(f"\nProbabilidade de superioridade P(B > A) = {PS:.4f}")
print(f"Delta de Cliff = 2 x {PS:.4f} - 1 = {2*PS - 1:.4f}")
print(f"r = z / raiz(N) = {abs(z)/np.sqrt(N):.4f}")

# 7. Estimativa de Hodges-Lehmann: mediana das diferenças entre todos os pares
difs = np.array([b - a for b in equipeB for a in equipeA])
print(f"\nDeslocamento de Hodges-Lehmann (mediana das {len(difs)} diferenças) = {np.median(difs):.3f} h")
print(f"Diferença entre as medianas = {np.median(equipeB) - np.median(equipeA):.3f} h")

# 8. O efeito do valor extremo
semext = equipeB[equipeB < 50]
print(f"\nSem a observação de 60 h (n = {len(semext)}):")
print(f"  teste t de Welch: p-valor = {stats.ttest_ind(equipeA, semext, equal_var=False).pvalue:.5f}")
print(f"  Mann-Whitney:     p-valor = {stats.mannwhitneyu(equipeA, semext).pvalue:.5f}")
print(f"Com log dos tempos, teste t de Welch: p-valor = {stats.ttest_ind(np.log(equipeA), np.log(equipeB), equal_var=False).pvalue:.5f}")

# 9. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].boxplot([equipeA, equipeB]); ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["equipe A", "equipe B"])
for j, g in enumerate([equipeA, equipeB], 1): ax[0].scatter(np.full(len(g), j), g, alpha=0.6)
ax[0].set_ylabel("Tempo até o reparo (h)")
ax[1].scatter(postos[:n1], np.full(n1, 1), alpha=0.7); ax[1].scatter(postos[n1:], np.full(n2, 2), alpha=0.7)
ax[1].set_yticks([1, 2]); ax[1].set_yticklabels(["equipe A", "equipe B"]); ax[1].set_xlabel("Posto")
plt.show()
