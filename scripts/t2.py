# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: conjunto PlantGrowth (peso seco de plantas, em g, sob três condições)
controle = np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14])
trat2    = np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26])
n1, n2 = len(controle), len(trat2)
m1, m2 = controle.mean(), trat2.mean()
s1, s2 = controle.std(ddof=1), trat2.std(ddof=1)
print(f"controle: n = {n1} | média = {m1:.4f} | DP = {s1:.4f} | variância = {s1**2:.4f}")
print(f"trat2:    n = {n2} | média = {m2:.4f} | DP = {s2:.4f} | variância = {s2**2:.4f}")
print(f"Diferença das médias (controle - trat2) = {m1 - m2:.4f} g")

# 1. Variância combinada e estatística de teste de Student
sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
ep = np.sqrt(sp2 * (1 / n1 + 1 / n2))
t = (m1 - m2) / ep; gl = n1 + n2 - 2
print(f"\nVariância combinada = [{n1-1} x {s1**2:.4f} + {n2-1} x {s2**2:.4f}] / {n1+n2-2} = {sp2:.4f}")
print(f"Erro-padrão da diferença = raiz({sp2:.4f} x (1/{n1} + 1/{n2})) = {ep:.4f}")
print(f"t = {m1 - m2:.4f} / {ep:.4f} = {t:.4f} com {gl} gl | p-valor = {2*stats.t.sf(abs(t), gl):.5f}")
print("scipy (Student):", stats.ttest_ind(controle, trat2, equal_var=True))

# 2. Intervalo de confiança de 95% para a diferença
tc = stats.t.ppf(0.975, gl); ic = (m1 - m2) + np.array([-1, 1]) * tc * ep
print(f"IC 95% da diferença = [{ic[0]:.4f}; {ic[1]:.4f}]")

# 3. Versão de Welch (não supõe variâncias iguais)
ep_w = np.sqrt(s1**2 / n1 + s2**2 / n2)
gl_w = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
t_w = (m1 - m2) / ep_w
print(f"\nWelch: erro-padrão = {ep_w:.4f} | gl = {gl_w:.4f} | t = {t_w:.4f} | p-valor = {2*stats.t.sf(abs(t_w), gl_w):.5f}")
print("scipy (Welch):", stats.ttest_ind(controle, trat2, equal_var=False))

# 4. Pressupostos
for nome, g in [("controle", controle), ("trat2", trat2)]:
    W, pw = stats.shapiro(g); print(f"Shapiro-Wilk ({nome}): W = {W:.4f}, p-valor = {pw:.4f}")
lev = stats.levene(controle, trat2, center="median")
print(f"Brown-Forsythe: W = {lev.statistic:.4f}, p-valor = {lev.pvalue:.4f} | razão entre variâncias = {max(s1,s2)**2/min(s1,s2)**2:.4f}")

# 5. Tamanho do efeito
d = (m1 - m2) / np.sqrt(sp2)
g_hedges = d * (1 - 3 / (4 * (n1 + n2) - 9))
print(f"\nd de Cohen = {d:.4f} | g de Hedges (corrigido) = {g_hedges:.4f}")

# 6. Alternativa não paramétrica
print("Mann-Whitney:", stats.mannwhitneyu(controle, trat2).pvalue.round(5))

# 7. Poder do teste
delta = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
print(f"Poder para o efeito observado (alfa = 0,05): {stats.nct.sf(tc, gl, delta) + stats.nct.cdf(-tc, gl, delta):.4f}")

# 8. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot([controle, trat2]); ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["controle", "trat2"])
for j, g in enumerate([controle, trat2], 1): ax[0].scatter(np.full(len(g), j), g, alpha=0.6)
ax[0].set_ylabel("Peso seco (g)")
ax[1].errorbar([0, 1], [m1, m2], yerr=[tc*s1/np.sqrt(n1), tc*s2/np.sqrt(n2)], fmt="o", capsize=8)
ax[1].set_xlim(-0.5, 1.5); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["controle", "trat2"])
ax[1].set_ylabel("Média com IC de 95%")
plt.show()
