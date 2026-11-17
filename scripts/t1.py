# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: peso líquido de 25 pacotes de um lote (g). Valor nominal declarado: 500 g (dados fictícios)
peso = np.array([492.2, 493.3, 500.6, 490.3, 499.8, 499.2, 501.3, 496.2, 500.0, 492.3,
                 501.6, 500.2, 496.3, 497.2, 493.8, 500.1, 494.6, 504.2, 494.4, 496.0,
                 501.6, 496.6, 500.2, 507.5, 496.5])
mu0 = 500.0
n = len(peso); media = peso.mean(); dp = peso.std(ddof=1)
print(f"n = {n} | média = {media:.4f} g | desvio-padrão = {dp:.4f} g | mediana = {np.median(peso):.1f} g")
print(f"Mínimo = {peso.min()} | máximo = {peso.max()}")

# 1. Erro-padrão e estatística de teste
ep = dp / np.sqrt(n)
t = (media - mu0) / ep
gl = n - 1
p_bilateral = 2 * stats.t.sf(abs(t), gl)
print(f"\nErro-padrão da média = {dp:.4f} / raiz({n}) = {ep:.4f}")
print(f"t = ({media:.4f} - {mu0}) / {ep:.4f} = {t:.4f} com {gl} gl")
print(f"Valor-p bilateral = {p_bilateral:.5f} | unilateral à esquerda = {stats.t.cdf(t, gl):.5f}")

# 2. Conferência com a função pronta
print("scipy.stats.ttest_1samp:", stats.ttest_1samp(peso, mu0))

# 3. Intervalo de confiança de 95% para a média
tc = stats.t.ppf(0.975, gl)
ic = media + np.array([-1, 1]) * tc * ep
print(f"\nValor crítico t(0,975; {gl}) = {tc:.4f}")
print(f"IC 95% = {media:.4f} +/- {tc:.4f} x {ep:.4f} = [{ic[0]:.4f}; {ic[1]:.4f}]")
print(f"O valor nominal de {mu0} g está dentro do intervalo? {'sim' if ic[0] <= mu0 <= ic[1] else 'não'}")

# 4. Tamanho do efeito (d de Cohen) e diferença média
print(f"\nDiferença média = {media - mu0:.4f} g | d de Cohen = {(media - mu0) / dp:.4f}")

# 5. Pressuposto de normalidade
W, pw = stats.shapiro(peso)
print(f"\nShapiro-Wilk: W = {W:.4f}, p-valor = {pw:.4f}")

# 6. Alternativas não paramétricas, para comparação
print("Wilcoxon (posto sinalizado):", stats.wilcoxon(peso - mu0).pvalue.round(5))
abaixo = int(np.sum(peso < mu0)); acima = int(np.sum(peso > mu0))
p_sinal = min(1.0, 2 * stats.binom.cdf(min(abaixo, acima), abaixo + acima, 0.5))
print(f"Teste do sinal: {abaixo} pesos abaixo de 500 g e {acima} acima | p-valor = {p_sinal:.5f}")

# 7. Poder do teste para a diferença observada
delta = abs(media - mu0) / dp * np.sqrt(n)
poder = stats.nct.sf(tc, gl, delta) + stats.nct.cdf(-tc, gl, delta)
print(f"\nPoder do teste para o efeito observado (alfa = 0,05): {poder:.4f}")
n_nec = 1
while True:
    d_ = abs(media - mu0) / dp * np.sqrt(n_nec); tc_ = stats.t.ppf(0.975, n_nec - 1)
    if n_nec > 2 and stats.nct.sf(tc_, n_nec - 1, d_) + stats.nct.cdf(-tc_, n_nec - 1, d_) >= 0.80: break
    n_nec += 1
print(f"n necessário para poder de 80% com esse tamanho de efeito: {n_nec}")

# 8. Gráficos
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].hist(peso, bins=8, color="lightgray", edgecolor="white")
ax[0].axvline(mu0, color="red", lw=2); ax[0].axvline(media, color="darkgreen", lw=2, ls="--")
ax[0].set_xlabel("Peso (g)"); ax[0].set_ylabel("Frequência")
stats.probplot(peso, dist="norm", plot=ax[1]); ax[1].set_title("Gráfico Q-Q")
ax[2].errorbar([0], [media], yerr=tc * ep, fmt="o", capsize=8)
ax[2].axhline(mu0, color="red", lw=2); ax[2].set_xlim(-1, 1); ax[2].set_xticks([])
ax[2].set_ylabel("Peso médio (g)")
plt.show()
