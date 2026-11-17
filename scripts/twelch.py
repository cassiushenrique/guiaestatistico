# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: consumo (mpg) por tipo de câmbio
# am: 0 = automático, 1 = manual
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
                10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4,
                15.8, 19.7, 15, 21.4])
am  = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
x, y = mpg[am == 0], mpg[am == 1]
n1, n2 = len(x), len(y); m1, m2 = x.mean(), y.mean(); s1, s2 = x.std(ddof=1), y.std(ddof=1)
print(f"automático: n = {n1} | média = {m1:.4f} | DP = {s1:.4f} | variância = {s1**2:.4f}")
print(f"manual:     n = {n2} | média = {m2:.4f} | DP = {s2:.4f} | variância = {s2**2:.4f}")
print(f"Razão entre as variâncias = {s2**2/s1**2:.4f} | razão entre os tamanhos = {n1/n2:.4f}")

# 1. Erro-padrão de Welch: cada grupo entra com a SUA variância
ep = np.sqrt(s1**2 / n1 + s2**2 / n2)
t = (m1 - m2) / ep
print(f"\nErro-padrão de Welch = raiz({s1**2:.4f}/{n1} + {s2**2:.4f}/{n2}) = {ep:.4f}")
print(f"t = ({m1:.4f} - {m2:.4f}) / {ep:.4f} = {t:.4f}")

# 2. Graus de liberdade de Welch-Satterthwaite
a_, b_ = s1**2 / n1, s2**2 / n2
gl = (a_ + b_)**2 / (a_**2 / (n1 - 1) + b_**2 / (n2 - 1))
print(f"\nGraus de liberdade = ({a_:.4f} + {b_:.4f})² / [{a_:.4f}²/{n1-1} + {b_:.4f}²/{n2-1}] = {gl:.4f}")
print(f"(entre {min(n1, n2) - 1} e {n1 + n2 - 2}, os limites possíveis)")
print(f"Valor-p = {2 * stats.t.sf(abs(t), gl):.6f}")
print("scipy (Welch):", stats.ttest_ind(x, y, equal_var=False))

# 3. Intervalo de confiança de 95% para a diferença
tc = stats.t.ppf(0.975, gl); ic = (m1 - m2) + np.array([-1, 1]) * tc * ep
print(f"IC 95% da diferença = [{ic[0]:.4f}; {ic[1]:.4f}] mpg")

# 4. Comparação com o teste de Student (supõe variâncias iguais)
s2p = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
ep_s = np.sqrt(s2p * (1 / n1 + 1 / n2)); t_s = (m1 - m2) / ep_s
print(f"\nStudent: variância combinada = {s2p:.4f} | erro-padrão = {ep_s:.4f} | t = {t_s:.4f}"
      f" | gl = {n1+n2-2} | p = {2*stats.t.sf(abs(t_s), n1+n2-2):.6f}")
print("scipy (Student):", stats.ttest_ind(x, y, equal_var=True))
print(f"Diferença entre os erros-padrão: {ep:.4f} (Welch) contra {ep_s:.4f} (Student)")

# 5. Por que Student erra aqui: grupo MENOR tem variância MAIOR
print(f"\nO grupo com mais observações ({n1}) tem a MENOR variância ({s1**2:.2f}).")
print("Nesse arranjo, o teste de Student subestima o erro-padrão e o valor-p fica otimista.")

# 6. Pressupostos
for nome, g in [("automático", x), ("manual", y)]:
    W, pw = stats.shapiro(g); print(f"Shapiro-Wilk ({nome}): W = {W:.4f}, p-valor = {pw:.4f}")
lev = stats.levene(x, y, center="median")
print(f"Brown-Forsythe: W = {lev.statistic:.4f}, p-valor = {lev.pvalue:.4f}")

# 7. Tamanho do efeito (com desvio-padrão combinado e com o d robusto de Glass)
print(f"\nd de Cohen = {(m1-m2)/np.sqrt(s2p):.4f} | delta de Glass (DP do controle) = {(m1-m2)/s1:.4f}")

# 8. Simulação: o que acontece com cada teste quando as variâncias diferem
rng = np.random.default_rng(2026)
rej_s = rej_w = 0
for _ in range(20000):
    g1 = rng.normal(0, 1, 20); g2 = rng.normal(0, 3, 8)       # médias iguais, variâncias diferentes
    rej_s += stats.ttest_ind(g1, g2, equal_var=True).pvalue < 0.05
    rej_w += stats.ttest_ind(g1, g2, equal_var=False).pvalue < 0.05
print(f"\nSimulação com médias IGUAIS (n = 20 e 8; DP = 1 e 3; 20000 repetições):")
print(f"  taxa de rejeição do Student = {rej_s/20000:.4f} | de Welch = {rej_w/20000:.4f} | nominal = 0,05")

# 9. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot([x, y]); ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["automático", "manual"])
for j, g in enumerate([x, y], 1): ax[0].scatter(np.full(len(g), j), g, alpha=0.6)
ax[0].set_ylabel("Consumo (mpg)")
ax[1].errorbar([0, 1], [m1, m2], yerr=[tc*s1/np.sqrt(n1), tc*s2/np.sqrt(n2)], fmt="o", capsize=8)
ax[1].set_xlim(-0.5, 1.5); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["automático", "manual"])
ax[1].set_ylabel("Média com IC de 95%")
plt.show()
