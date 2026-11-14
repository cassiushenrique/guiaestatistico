# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: tempo de reação (ms) de 12 participantes em três condições de ruído
dados = np.array([
    [412, 428, 471], [385, 402, 455], [440, 447, 462], [398, 421, 502],
    [455, 460, 489], [372, 395, 468], [420, 431, 444], [401, 426, 517],
    [433, 441, 470], [390, 409, 446], [447, 452, 495], [408, 430, 483]], dtype=float)
condicoes = ["Silêncio", "Ruído moderado", "Ruído intenso"]
n, k = dados.shape

# 1. Médias e variâncias das diferenças entre pares de condições
print("Médias por condição:", np.round(dados.mean(axis=0), 2))
for i in range(k):
    for j in range(i + 1, k):
        d = dados[:, i] - dados[:, j]
        print(f"Var({condicoes[i]} - {condicoes[j]}) = {d.var(ddof=1):.2f}")

# 2. Contrastes ortonormais (Helmert) e matriz de covariâncias transformada
C = np.array([[1, -1, 0], [1, 1, -2]], dtype=float)
C = C / np.sqrt((C**2).sum(axis=1, keepdims=True))
S = np.cov(dados, rowvar=False)
Sc = C @ S @ C.T
print("\nMatriz C S C':\n", np.round(Sc, 2))

# 3. Estatística W de Mauchly e aproximação qui-quadrado
p = k - 1
W = np.linalg.det(Sc) / (np.trace(Sc) / p)**p
gl = p * (p + 1) // 2 - 1
rho = 1 - (2*p**2 + p + 2) / (6 * p * (n - 1))
X2 = -(n - 1) * rho * np.log(W)
valor_p = stats.chi2.sf(X2, gl)
print(f"\nMauchly: W = {W:.5f}, qui-quadrado = {X2:.4f}, gl = {gl}, p-valor = {valor_p:.2e}")

# 4. Épsilons de correção dos graus de liberdade
eps_gg = np.trace(Sc)**2 / (p * np.trace(Sc @ Sc))
eps_hf = min(1, (n * p * eps_gg - 2) / (p * (n - 1 - p * eps_gg)))
print(f"Greenhouse-Geisser = {eps_gg:.4f} | Huynh-Feldt = {eps_hf:.4f} | limite inferior = {1/p:.4f}")

# 5. ANOVA de medidas repetidas com e sem correção
gm = dados.mean()
SQ_cond = n * np.sum((dados.mean(axis=0) - gm)**2)
SQ_suj = k * np.sum((dados.mean(axis=1) - gm)**2)
SQ_res = np.sum((dados - gm)**2) - SQ_cond - SQ_suj
gl1, gl2 = k - 1, (k - 1) * (n - 1)
F = (SQ_cond / gl1) / (SQ_res / gl2)
print(f"\nANOVA: F({gl1}, {gl2}) = {F:.3f}, p-valor = {stats.f.sf(F, gl1, gl2):.2e}")
print(f"Com Greenhouse-Geisser: gl = ({gl1*eps_gg:.2f}, {gl2*eps_gg:.2f}), p-valor = {stats.f.sf(F, gl1*eps_gg, gl2*eps_gg):.2e}")

# 6. Gráfico: perfis individuais e diferenças entre condições
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(dados.T, color="gray", alpha=0.6, marker="o")
ax[0].plot(dados.mean(axis=0), color="red", lw=3, marker="o")
ax[0].set_xticks(range(k)); ax[0].set_xticklabels(condicoes); ax[0].set_ylabel("Tempo de reação (ms)")
difs = [dados[:, 0] - dados[:, 1], dados[:, 0] - dados[:, 2], dados[:, 1] - dados[:, 2]]
ax[1].boxplot(difs); ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels(["S - M", "S - I", "M - I"])
ax[1].set_ylabel("Diferença (ms)")
plt.show()
